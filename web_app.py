import os
import sys
import time
import json
import collections
import glob
import re
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, send_file
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    BASE_DIR, SEQUENCE_LENGTH, FEATURE_DIM, CONFIDENCE_THRESHOLD, DEBOUNCE_FRAMES,
    SUPPORTED_LANGUAGES, IDLE_VELOCITY_THRESHOLD, DATA_DIR, REPORTS_DIR, UPLOADS_DIR,
    GESTURE_TRANSLATIONS
)
from src.feature_extractor import HandFeatureExtractor
from src.model import ISLGestureClassifier
from src.tts_engine import MultilingualTTSEngine
from src.utils import calculate_hand_velocity

from src.sentence_translator import translate_isl_sequence, close_sentence
from src.isl_sentence_lm import ISLSentenceLanguageModel

# ─────────────────────────────────────────────────────────────
# LANGUAGE MODEL — Load or skip gracefully
# ─────────────────────────────────────────────────────────────
_LM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "isl_sentence_lm.json")
sentence_lm = ISLSentenceLanguageModel()
if os.path.exists(_LM_PATH):
    sentence_lm.load(_LM_PATH)
    print("[LM] ISL Sentence Language Model loaded.", flush=True)
else:
    print("[LM] No pre-built LM found. Run build_sentence_lm.py to enable LM re-scoring.", flush=True)

# ─────────────────────────────────────────────────────────────
# SENTENCE BUILDER — Pause-aware, LM-enhanced sentence assembly
# ─────────────────────────────────────────────────────────────
class SentenceBuilder:
    """
    Manages continuous ISL gesture words into complete sentences.

    - Groups gesture words signed without a gap into one sentence unit.
    - After PAUSE_SECONDS of no gesture (idle / no hand detected), 
      the current word buffer is 'closed' as a complete sentence.
    - Keeps the last MAX_HISTORY completed sentences for display.
    - Integrates the Language Model to smooth ambiguous predictions.
    """
    PAUSE_SECONDS   = 1.8   # Silence duration to trigger sentence boundary
    MAX_WORD_BUFFER = 10    # Max words in one sentence
    MAX_HISTORY     = 5     # How many completed sentences to keep

    def __init__(self, lm: ISLSentenceLanguageModel):
        self.lm              = lm
        self.word_buffer     = []          # Current in-progress sentence words
        self.completed       = []          # List of completed sentence dicts
        self.last_word_time  = 0.0
        self.last_raw_word   = None        # Last added raw gesture label
        self._sentence_just_closed = False # True for 1 poll cycle after closure

    def add_word(self, raw_label: str, candidate_labels: list = None):
        """
        Add a newly confirmed gesture word.
        If multiple candidates given, use LM to pick the best one.
        """
        now = time.time()

        # Use LM to rescore if we have candidates and context
        if candidate_labels and len(candidate_labels) > 1 and self.lm._trained:
            best = self.lm.rescore_candidates(self.word_buffer, candidate_labels)
        else:
            best = raw_label

        # Don't duplicate consecutive identical words
        if best == self.last_raw_word:
            return

        self.word_buffer.append(best)
        self.last_raw_word = best
        self.last_word_time = now
        self._sentence_just_closed = False

        # Auto-close if buffer hits max length
        if len(self.word_buffer) >= self.MAX_WORD_BUFFER:
            self._close_sentence()

    def tick(self):
        """
        Called periodically. Closes the sentence if a pause has been detected.
        Returns True if a sentence was just closed this tick.
        """
        if (self.word_buffer and
                time.time() - self.last_word_time > self.PAUSE_SECONDS):
            self._close_sentence()
            return True
        return False

    def _close_sentence(self):
        """Finalises the current word buffer as a complete sentence."""
        if not self.word_buffer:
            return
        words = list(self.word_buffer)
        en = close_sentence(words, "en")
        hi = close_sentence(words, "hi")
        self.completed.append({
            "words":  words,
            "en":     en,
            "hi":     hi,
            "time":   time.time()
        })
        # Trim history
        if len(self.completed) > self.MAX_HISTORY:
            self.completed.pop(0)
        self.word_buffer.clear()
        self.last_raw_word = None
        self._sentence_just_closed = True
        print(f"[SentenceBuilder] Closed sentence: {words} => {en}", flush=True)

    def reset(self):
        self.word_buffer.clear()
        self.completed.clear()
        self.last_raw_word = None
        self._sentence_just_closed = False

    @property
    def current_words(self):
        return list(self.word_buffer)

    @property
    def latest_sentence(self):
        return self.completed[-1] if self.completed else None

    @property
    def sentence_just_closed(self):
        val = self._sentence_just_closed
        self._sentence_just_closed = False   # auto-reset after one read
        return val

sentence_builder = SentenceBuilder(sentence_lm)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload size

# Shared System State
camera_id = 0
extractor = HandFeatureExtractor(max_num_hands=1)
classifier = ISLGestureClassifier()
tts = MultilingualTTSEngine(default_lang="en")

sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
recent_predictions = collections.deque(maxlen=DEBOUNCE_FRAMES)
prev_frame_feat = None
no_hand_consecutive_frames = 0

system_state = {
    "input_mode": "webcam",
    "active_video_filename": None,
    "prediction": "idle",
    "confidence": 0.0,
    "sentence": [],
    "video_sentence": [],
    "video_done": False,
    "video_lang": "en",
    "language": "en",
    "fps": 0.0,
    "last_speech_time": 0,
    "last_spoken_gesture": None,
    # Sentence-builder enhanced fields
    "completed_sentences": [],   # list of {words, en, hi}
    "sentence_closed": False,    # True for 1 poll after a sentence is finalized
    "lm_next_predictions": [],   # LM-predicted next words
}

def open_working_camera():
    """Tries opening available camera devices (0, 1) using DirectShow or default backend."""
    for idx in [0, 1]:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, test_f = cap.read()
                if ret and test_f is not None and test_f.mean() > 2:
                    return cap
                cap.release()
        except:
            pass

        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, test_f = cap.read()
                if ret and test_f is not None:
                    return cap
                cap.release()
        except:
            pass

    return cv2.VideoCapture(0)

def generate_webcam_stream():
    """Generates real-time MJPEG video stream from live webcam with MediaPipe HUD annotations."""
    global prev_frame_feat
    cap = open_working_camera()
    fps_start = time.time()
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        now = time.time()
        if now - fps_start >= 1.0:
            system_state["fps"] = round(frame_count / (now - fps_start), 1)
            frame_count = 0
            fps_start = now

        frame = cv2.flip(frame, 1)
        feat, annotated_frame, detected = extractor.process_frame(frame)

        if detected:
            velocity = calculate_hand_velocity(prev_frame_feat, feat)
            prev_frame_feat = feat.copy()
            sequence_buffer.append(feat)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                if velocity < IDLE_VELOCITY_THRESHOLD:
                    pred_label, conf = "idle", 1.0
                else:
                    seq_arr = np.array(sequence_buffer, dtype=np.float32)
                    pred_label, conf, _ = classifier.predict(seq_arr)

                system_state["prediction"] = pred_label
                system_state["confidence"] = round(conf * 100, 1)
                recent_predictions.append(pred_label)

                if len(recent_predictions) == DEBOUNCE_FRAMES:
                    if len(set(recent_predictions)) == 1 and conf >= CONFIDENCE_THRESHOLD:
                        stable = recent_predictions[0]
                        if stable != "idle" and stable != system_state["last_spoken_gesture"]:
                            if now - system_state["last_speech_time"] > 1.8:
                                system_state["last_spoken_gesture"] = stable
                                system_state["last_speech_time"] = now

                                translated = tts.get_translated_text(stable)
                                system_state["sentence"].append(translated)
                                if len(system_state["sentence"]) > 8:
                                    system_state["sentence"].pop(0)

                                tts.speak(translated)
        else:
            sequence_buffer.clear()
            recent_predictions.clear()
            system_state["prediction"] = "idle"
            system_state["confidence"] = 0.0

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_video_file_stream(filename, lang="en"):
    """
    Streams selected video file frame-by-frame with live MediaPipe landmark tracking,
    GRU gesture classification, on-screen subtitles, and audio speech triggering.
    Stops when video ends (no loop). Speaks full sentence summary on completion.
    """
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(file_path):
        return

    # Check if video contains an unindexed gesture label and auto-learn it on the fly!
    base_label = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(filename)[0]).strip('_').lower()
    if base_label and base_label not in classifier.labels:
        print(f"[ON-THE-FLY LEARNER] Learning new video sign: '{base_label}'...", flush=True)
        ok, learned_lbl, _ = auto_index_and_learn_video(file_path, base_label)
        if ok:
            classifier.load_model()

    # Reset video_done flag, sentence builder, and system state at start
    sentence_builder.reset()
    system_state["video_done"] = False
    system_state["video_sentence"] = []
    system_state["video_lang"] = lang
    system_state["sentence"] = []
    system_state["completed_sentences"] = []
    system_state["prediction"] = "idle"
    system_state["confidence"] = 0.0

    cap = cv2.VideoCapture(file_path)
    seq_buf = collections.deque(maxlen=SEQUENCE_LENGTH)
    rec_preds = collections.deque(maxlen=DEBOUNCE_FRAMES)
    prev_f = None

    current_pred = "idle"
    current_conf = 0.0
    video_sentence = []      # words accumulated during this video
    last_spoken = None
    last_speak_t = 0
    frame_w, frame_h = 640, 360

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # Video ended — DO NOT loop
            break

        frame_h, frame_w, _ = frame.shape
        feat, annotated_frame, detected = extractor.process_frame(frame)
        now = time.time()

        if detected:
            velocity = calculate_hand_velocity(prev_f, feat)
            prev_f = feat.copy()
            seq_buf.append(feat)

            if len(seq_buf) == SEQUENCE_LENGTH:
                if velocity < IDLE_VELOCITY_THRESHOLD:
                    pred_label, conf = "idle", 1.0
                else:
                    seq_arr = np.array(seq_buf, dtype=np.float32)
                    pred_label, conf, _ = classifier.predict(seq_arr)

                current_pred = pred_label
                current_conf = round(conf * 100, 1)
                rec_preds.append(pred_label)

                if len(rec_preds) == DEBOUNCE_FRAMES:
                    if len(set(rec_preds)) == 1 and conf >= CONFIDENCE_THRESHOLD:
                        stable = rec_preds[0]
                        if stable != "idle" and stable != last_spoken:
                            if now - last_speak_t > 1.8:
                                last_spoken = stable
                                last_speak_t = now

                                translated = tts.get_translated_text(stable, lang)
                                if stable not in video_sentence:
                                    video_sentence.append(stable)

                                # Speak each word as it's recognized
                                tts.speak(translated, lang)

        # Update global state for API polling
        system_state["prediction"] = current_pred
        system_state["confidence"] = current_conf
        system_state["sentence"] = list(video_sentence)
        system_state["video_sentence"] = list(video_sentence)

        # Draw Top HUD Overlay Banner
        cv2.rectangle(annotated_frame, (0, 0), (frame_w, 60), (20, 20, 20), -1)
        cv2.putText(annotated_frame, f"ISL SIGN: {current_pred.upper()}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"CONF: {current_conf:.1f}%", (frame_w - 220, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        # Draw Bottom Subtitle Banner
        cv2.rectangle(annotated_frame, (0, frame_h - 60), (frame_w, frame_h), (15, 15, 15), -1)
        sub_str = (f"TRANSLATION ({lang.upper()}): " + " | ".join([w.replace("_", " ") for w in video_sentence]))\
            if video_sentence else f"TRANSLATION ({lang.upper()}): [Recognizing ISL gestures...]"
        cv2.putText(annotated_frame, sub_str, (20, frame_h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

    cap.release()

    # ===== VIDEO ENDED: Generate final summary frame =====
    raw_sentence = " | ".join([w.upper().replace("_", " ") for w in video_sentence]) if video_sentence else "No ISL signs detected"
    fluent_sentence = translate_isl_sequence(video_sentence, lang) if video_sentence else "No ISL signs detected"

    summary_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    summary_frame[:] = (15, 23, 42)   # dark blue background

    # Title bar
    cv2.rectangle(summary_frame, (0, 0), (frame_w, 65), (30, 80, 40), -1)
    cv2.putText(summary_frame, "VIDEO PROCESSING COMPLETE", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 255, 120), 2)

    # Always use English for the OpenCV frame — cv2.putText cannot render Devanagari (Hindi)
    fluent_en = translate_isl_sequence(video_sentence, 'en') if video_sentence else "No ISL signs detected"

    # Summary labels
    cv2.putText(summary_frame, "RAW ISL SIGNS:", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (150, 200, 255), 2)
    cv2.putText(summary_frame, raw_sentence, (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 220, 255), 2)

    cv2.putText(summary_frame, "FLUENT TRANSLATION (English):", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (100, 255, 180), 2)
    cv2.putText(summary_frame, fluent_en, (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    lang_label = "English" if lang == "en" else "Hindi (see side panel)"
    cv2.putText(summary_frame, f"Language: {lang_label}", (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 1)

    total_signs = len(video_sentence)
    cv2.putText(summary_frame, f"Total unique signs detected: {total_signs}", (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.putText(summary_frame, "Audio playback triggered. Check translation panel.", (20, frame_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)

    # Hold the summary frame for 5 seconds (streaming 30 copies at 0.16s each)
    ret, buffer = cv2.imencode('.jpg', summary_frame)
    if ret:
        for _ in range(30):
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.16)

    # Speak the complete fluent sentence summary
    if video_sentence:
        fluent_text = translate_isl_sequence(video_sentence, lang)
        tts.speak(fluent_text, lang)

    # Mark video as done so frontend can react
    system_state["video_done"] = True
    system_state["video_sentence"] = list(video_sentence)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera_test')
def camera_test():
    return render_template('camera_test.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_webcam_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream_feed')
def video_stream_feed():
    filename = request.args.get('filename', '')
    lang = request.args.get('lang', 'en')
    return Response(generate_video_file_stream(filename, lang), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    Receives a base64-encoded JPEG frame from the browser's getUserMedia webcam,
    runs MediaPipe hand landmark extraction + BiGRU gesture classification,
    and returns the prediction in real-time.
    """
    global prev_frame_feat, no_hand_consecutive_frames
    try:
        data = request.get_json(force=True) or {}
        frame_b64 = data.get('frame', '')
        if not frame_b64:
            return jsonify({"prediction": "idle", "confidence": 0.0, "sentence": [],
                            "translation_en": "", "translation_hi": "", "sentence_closed": False})

        # Decode base64 image from browser
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',', 1)[1]
        img_bytes = np.frombuffer(__import__('base64').b64decode(frame_b64), dtype=np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"prediction": "idle", "confidence": 0.0, "sentence": [],
                            "translation_en": "", "translation_hi": "", "sentence_closed": False})

        # Flip horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Run MediaPipe hand landmark extraction
        feat, _, detected = extractor.process_frame(frame)

        pred_label = "idle"
        conf = 0.0
        sentence_just_closed = False

        if detected:
            no_hand_consecutive_frames = 0
            velocity = calculate_hand_velocity(prev_frame_feat, feat)
            prev_frame_feat = feat.copy()
            sequence_buffer.append(feat)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                if velocity < IDLE_VELOCITY_THRESHOLD:
                    pred_label, conf = "idle", 1.0
                else:
                    seq_arr = np.array(sequence_buffer, dtype=np.float32)
                    pred_label, conf, proba_dict = classifier.predict(seq_arr)

            system_state["prediction"] = pred_label
            system_state["confidence"] = round(conf * 100, 1)
            recent_predictions.append(pred_label)

            now = time.time()
            if len(recent_predictions) == DEBOUNCE_FRAMES:
                if len(set(recent_predictions)) == 1 and conf >= CONFIDENCE_THRESHOLD:
                    stable = recent_predictions[0]
                    if stable != "idle" and stable != system_state["last_spoken_gesture"]:
                        if now - system_state["last_speech_time"] > 1.0:   # reduced from 1.8s -> 1.0s
                            system_state["last_spoken_gesture"] = stable
                            system_state["last_speech_time"] = now

                            # Add to SentenceBuilder (with LM rescoring)
                            sentence_builder.add_word(stable)

                            # Update flat sentence list for backwards-compat with UI
                            system_state["sentence"] = sentence_builder.current_words
                            if len(system_state["sentence"]) > 10:
                                system_state["sentence"] = system_state["sentence"][-10:]

                            # Speak the raw word immediately (browser TTS handles fluent output)
                            translated = tts.get_translated_text(stable)
                            tts.speak(translated)

            # Run SentenceBuilder tick to check for pause-based sentence closure
            sentence_just_closed = sentence_builder.tick()
            if sentence_just_closed:
                system_state["sentence"] = []  # Reset flat buffer after closure
                system_state["last_spoken_gesture"] = None
                system_state["completed_sentences"] = [
                    {"en": s["en"], "hi": s["hi"], "words": s["words"]}
                    for s in sentence_builder.completed
                ]
                system_state["sentence_closed"] = True

        else:
            # No hand detected — increment consecutive missing frame counter
            no_hand_consecutive_frames += 1
            system_state["prediction"] = "idle"
            system_state["confidence"] = 0.0

            # Only clear the buffers if hand is missing for more than 5 consecutive frames (grace period)
            if no_hand_consecutive_frames >= 5:
                sequence_buffer.clear()
                recent_predictions.clear()

            sentence_just_closed = sentence_builder.tick()
            if sentence_just_closed:
                system_state["sentence"] = []
                system_state["last_spoken_gesture"] = None
                system_state["completed_sentences"] = [
                    {"en": s["en"], "hi": s["hi"], "words": s["words"]}
                    for s in sentence_builder.completed
                ]
                system_state["sentence_closed"] = True

        # Get current in-progress words for translation preview
        current_words = sentence_builder.current_words
        en_trans = translate_isl_sequence(current_words, "en") if current_words else ""
        hi_trans = translate_isl_sequence(current_words, "hi") if current_words else ""

        # LM next word predictions
        lm_next = []
        if sentence_lm._trained and current_words:
            lm_preds = sentence_lm.predict_next(current_words, top_k=3)
            lm_next = [w for w, _ in lm_preds]
        system_state["lm_next_predictions"] = lm_next

        # Check if sentence was just closed this cycle
        closed_flag = system_state.get("sentence_closed", False)
        if closed_flag:
            system_state["sentence_closed"] = False  # Reset after first read

        latest = sentence_builder.latest_sentence
        return jsonify({
            "prediction":        system_state["prediction"],
            "confidence":        system_state["confidence"],
            "sentence":          sentence_builder.current_words,
            "translation_en":    en_trans,
            "translation_hi":    hi_trans,
            "sentence_closed":   closed_flag,
            "completed_sentence_en": latest["en"] if latest else "",
            "completed_sentence_hi": latest["hi"] if latest else "",
            "completed_words":   latest["words"] if latest else [],
            "lm_next":           lm_next,
        })

    except Exception as e:
        print(f"process_frame error: {e}", flush=True)
        return jsonify({"prediction": "idle", "confidence": 0.0, "sentence": [],
                        "translation_en": "", "translation_hi": "", "sentence_closed": False})



@app.route('/api/status', methods=['GET'])
def get_status():
    pred = system_state["prediction"]

    # For video mode: use video_sentence. For live: use in-progress words from builder
    if system_state.get("video_done"):
        words_sequence = system_state.get("video_sentence", [])
    else:
        words_sequence = sentence_builder.current_words or system_state["sentence"]

    if not words_sequence and pred != "idle":
        words_sequence = [pred]

    # Fluent translation for current in-progress words
    en_trans = translate_isl_sequence(words_sequence, lang="en")
    hi_trans = translate_isl_sequence(words_sequence, lang="hi")

    # Latest completed sentence
    latest = sentence_builder.latest_sentence
    latest_en = latest["en"] if latest else ""
    latest_hi = latest["hi"] if latest else ""

    # LM next word predictions
    lm_next = []
    if sentence_lm._trained and words_sequence:
        lm_preds = sentence_lm.predict_next(words_sequence, top_k=3)
        lm_next = [w for w, _ in lm_preds]

    return jsonify({
        "input_mode":            system_state["input_mode"],
        "active_video":          system_state["active_video_filename"],
        "prediction":            pred,
        "translation_en":        en_trans,
        "translation_hi":        hi_trans,
        "confidence":            system_state["confidence"],
        "sentence":              words_sequence,
        "video_done":            system_state.get("video_done", False),
        "video_sentence":        system_state.get("video_sentence", []),
        "language":              tts.current_lang,
        "language_name":         SUPPORTED_LANGUAGES.get(tts.current_lang, "English"),
        "fps":                   system_state["fps"],
        "vocabulary":            classifier.labels,
        # Sentence builder fields
        "completed_sentence_en": latest_en,
        "completed_sentence_hi": latest_hi,
        "completed_sentences":   system_state.get("completed_sentences", []),
        "lm_next":               lm_next,
    })

from src.auto_indexer import auto_index_and_learn_video

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'video_file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        file = request.files['video_file']
        if not file or file.filename == '':
            return jsonify({"status": "error", "message": "Empty file name"}), 400

        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"status": "error", "message": "Invalid filename"}), 400

        os.makedirs(UPLOADS_DIR, exist_ok=True)
        save_path = os.path.join(UPLOADS_DIR, filename)
        file.save(save_path)

        system_state["active_video_filename"] = filename

        # Only run the auto-indexer/learner if explicitly requested (e.g. via a learn parameter)
        # to prevent corrupting the baseline model weights when users upload videos just to translate them.
        should_learn = request.form.get('learn') == 'true'
        learned_label = ""
        num_seqs = 0
        if should_learn:
            custom_label = request.form.get('label')
            ok, learned_label, num_seqs = auto_index_and_learn_video(save_path, custom_label)
            if ok:
                classifier.load_model()

        return jsonify({
            "status": "success",
            "filename": filename,
            "learned_label": learned_label,
            "samples_indexed": num_seqs,
            "original_url": f"/api/video_file/{filename}",
            "message": f"Uploaded and indexed '{filename}' as sign '{learned_label}'."
        })
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/learn_video', methods=['POST'])
def learn_video():
    """Allows user to explicitly index and train ANY video file into the model vocabulary."""
    try:
        data = request.get_json() or {}
        filename = data.get('filename') or system_state.get('active_video_filename')
        custom_label = data.get('label')

        if not filename:
            return jsonify({"status": "error", "message": "No video file specified"}), 400

        file_path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(BASE_DIR, filename)

        if not os.path.exists(file_path):
            return jsonify({"status": "error", "message": f"File '{filename}' not found"}), 404

        ok, label, num_seqs = auto_index_and_learn_video(file_path, custom_label)
        if ok:
            classifier.load_model()
            return jsonify({
                "status": "success",
                "label": label,
                "samples_saved": num_seqs,
                "message": f"Successfully learned sign '{label}' from video '{filename}'!"
            })
        else:
            return jsonify({"status": "error", "message": "Failed to extract hand landmarks from video"}), 400

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/video_file/<path:filename>')
def serve_video_file(filename):
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(BASE_DIR, filename)

    if os.path.exists(file_path):
        return send_file(file_path, mimetype='video/mp4')
    return jsonify({"status": "error", "message": "File not found"}), 404

@app.route('/api/list_videos', methods=['GET'])
def list_videos():
    videos = []
    if os.path.exists(UPLOADS_DIR):
        for f in os.listdir(UPLOADS_DIR):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append({"filename": f, "type": "uploaded"})

    for f in os.listdir(BASE_DIR):
        if f.lower().endswith('.mp4') and not f.endswith('_translated.mp4') and not f.endswith('_translated_hindi.mp4'):
            videos.append({"filename": f, "type": "sample"})

    return jsonify({"videos": videos})

@app.route('/api/set_language', methods=['POST'])
def set_language():
    data = request.get_json() or {}
    lang = data.get("language", "en")
    if lang in SUPPORTED_LANGUAGES:
        tts.set_language(lang)
        system_state["language"] = lang
        return jsonify({"status": "success", "language": lang})
    return jsonify({"status": "error", "message": "Invalid language code"}), 400

@app.route('/api/speak_audio', methods=['POST'])
def speak_audio():
    data = request.get_json() or {}
    text = data.get("text", "")
    lang = data.get("language", tts.current_lang)
    if text:
        tts.speak(text, lang)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Empty text"}), 400

@app.route('/api/reset_sentence', methods=['POST'])
def reset_sentence():
    system_state["sentence"] = []
    system_state["last_spoken_gesture"] = None
    system_state["completed_sentences"] = []
    system_state["sentence_closed"] = False
    sentence_builder.reset()
    return jsonify({"status": "success"})

@app.route('/api/record_webcam_sign', methods=['POST'])
def record_webcam_sign():
    """
    Accepts a list of 30 base64-encoded frames from the browser, extracts
    MediaPipe hand joint landmarks, saves the gesture sequence, and fine-tunes the model.
    """
    try:
        data = request.get_json() or {}
        raw_label = data.get("label", "").strip()
        frames_list = data.get("frames", [])

        if not raw_label:
            return jsonify({"status": "error", "message": "Please enter a sign name"}), 400
        if not frames_list or len(frames_list) < 5:
            return jsonify({"status": "error", "message": "No frames captured. Please ensure your camera is active."}), 400

        # Clean label name
        sign_label = re.sub(r'[^a-zA-Z0-9_]', '_', raw_label).strip('_').lower()
        if not sign_label:
            sign_label = "custom_sign"

        print(f"[LIVE RECORD] Processing {len(frames_list)} browser frames for gesture: '{sign_label}'...", flush=True)

        captured_features = []
        import base64

        for f_b64 in frames_list:
            try:
                if ',' in f_b64:
                    f_b64 = f_b64.split(',', 1)[1]
                img_bytes = np.frombuffer(base64.b64decode(f_b64), dtype=np.uint8)
                frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Mirror/flip frame to align with standard webcam mirroring
                    frame = cv2.flip(frame, 1)
                    feat, _, detected = extractor.process_frame(frame)
                    if detected:
                        captured_features.append(feat)
            except Exception as fe:
                print(f"Error processing frame in list: {fe}")
                continue

        if len(captured_features) < 3:
            return jsonify({
                "status": "error", 
                "message": "No hand detected in the recorded video frames. Please hold your hand clearly in front of the camera and try again."
            }), 400

        # Pad to exactly 30 frames
        seq_arr = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        n = len(captured_features)
        arr = np.array(captured_features, dtype=np.float32)
        seq_arr[:n] = arr
        for i in range(n, SEQUENCE_LENGTH):
            seq_arr[i] = arr[-1]

        # Save sequence .npy files to data/<sign_label>/
        sign_dir = os.path.join(DATA_DIR, sign_label)
        os.makedirs(sign_dir, exist_ok=True)
        existing = len(glob.glob(os.path.join(sign_dir, "*.npy")))

        # Save original + 2 augmented samples
        for s in range(3):
            noise = np.random.normal(0, 0.003, seq_arr.shape).astype(np.float32) if s > 0 else 0
            fname = f"live_webcam_{existing + s + 1}.npy"
            np.save(os.path.join(sign_dir, fname), (seq_arr + noise).astype(np.float32))

        # Fine-tune model with Replay Buffer to prevent catastrophic forgetting.
        # When learning a new sign, we also include samples from ALL other custom-learned
        # signs and a set of anchor classes, so the fc layer doesn't over-bias toward
        # the new sign and start misclassifying older ones.
        if sign_label not in classifier.labels:
            classifier.labels.append(sign_label)
            classifier.label_to_idx[sign_label] = len(classifier.labels) - 1
            classifier.idx_to_label[len(classifier.labels) - 1] = sign_label
            classifier.model.update_class_count(len(classifier.labels))

        target_idx = classifier.label_to_idx[sign_label]

        # ── Build Replay Buffer ──────────────────────────────────────────────
        replay_X = []
        replay_y = []

        # 1. Include files of the new sign (target class) - Max 3 files to stay balanced
        webcam_files = glob.glob(os.path.join(sign_dir, "live_webcam_*.npy"))
        target_files = webcam_files[:3] if len(webcam_files) >= 3 else glob.glob(os.path.join(sign_dir, "*.npy"))[:3]
        for f in target_files:
            replay_X.append(np.load(f))
            replay_y.append(target_idx)

        # 2. Include files from every other custom-learned sign folder - Max 3 files
        #    (any folder that contains live_webcam_*.npy files)
        for folder in os.listdir(DATA_DIR):
            if folder == sign_label:
                continue
            folder_path = os.path.join(DATA_DIR, folder)
            if not os.path.isdir(folder_path):
                continue
            webcam_files = glob.glob(os.path.join(folder_path, "live_webcam_*.npy"))
            if webcam_files:
                other_idx = classifier.label_to_idx.get(folder)
                if other_idx is None:
                    continue
                for f in webcam_files[:3]:
                    replay_X.append(np.load(f))
                    replay_y.append(other_idx)

        # 3. Add a small set of anchor class samples so baseline classes stay accurate - Max 3 files
        anchor_classes = ["hello", "thank_you", "yes", "no", "help", "water", "food",
                          "father", "mother", "good", "bad", "name", "work", "eat", "drink"]
        for anchor in anchor_classes:
            if anchor == sign_label:
                continue
            anchor_dir = os.path.join(DATA_DIR, anchor)
            anchor_idx = classifier.label_to_idx.get(anchor)
            if anchor_idx is None or not os.path.isdir(anchor_dir):
                continue
            anchor_files = glob.glob(os.path.join(anchor_dir, "*.npy"))
            for f in anchor_files[:3]:
                replay_X.append(np.load(f))
                replay_y.append(anchor_idx)

        replay_X = np.array(replay_X, dtype=np.float32)
        replay_y = np.array(replay_y, dtype=np.int64)
        print(f"[LIVE RECORD] Replay buffer: {len(replay_X)} samples across {len(np.unique(replay_y))} classes", flush=True)

        # Fine-tune fc layer only (GRU backbone stays frozen)
        classifier.fine_tune_transfer_learning(replay_X, replay_y, [sign_label], epochs=45, lr=0.005)
        classifier.save_model()
        classifier.load_model()

        print(f"[LIVE RECORD] SUCCESS: Learned sign '{sign_label}'!", flush=True)

        return jsonify({
            "status": "success",
            "label": sign_label,
            "samples_captured": len(captured_features),
            "message": f"Successfully learned sign '{sign_label}'! You can now perform this gesture in front of the camera to translate it."
        })

    except Exception as e:
        print(f"Record webcam sign error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/delete_custom_sign', methods=['POST'])
def delete_custom_sign():
    """Deletes a custom sign class from the model and disk."""
    try:
        data = request.get_json(force=True) or {}
        label = data.get('label', '').strip().lower()
        if not label:
            return jsonify({"status": "error", "message": "No label provided."}), 400

        success = classifier.delete_custom_class(label)
        if not success:
            return jsonify({"status": "error", "message": f"Sign '{label}' is not found."}), 400

        classifier.load_model()
        system_state["prediction"] = "idle"
        system_state["confidence"] = 0.0

        return jsonify({
            "status": "success",
            "message": f"Successfully deleted sign '{label}' from model and disk."
        })
    except Exception as e:
        print(f"Delete custom sign error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/get_custom_signs', methods=['GET'])
def get_custom_signs():
    """Returns the list of custom-learned signs currently loaded on disk."""
    try:
        custom_labels = classifier.get_custom_labels()
        return jsonify({
            "status": "success",
            "custom_signs": custom_labels
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/reports/<filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename)

if __name__ == '__main__':
    print("\n=======================================================")
    print("  Full-Stack ISL Recognition Web Application")
    print("  Access at: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
