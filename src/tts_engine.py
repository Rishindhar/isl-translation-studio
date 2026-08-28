import os
import threading
import time
import hashlib
import pyttsx3
from gtts import gTTS
from src.config import AUDIO_CACHE_DIR, GESTURE_TRANSLATIONS

class MultilingualTTSEngine:
    """
    Multilingual Text-to-Speech Engine for English ('en') & Hindi ('hi').
    Uses gTTS for high-quality audio with pyttsx3 offline fallback,
    cached locally to prevent playback latency and API throttle.
    """
    def __init__(self, default_lang="en"):
        self.current_lang = default_lang
        self.lock = threading.Lock()
        self.pyttsx_engine = None
        self._init_pyttsx3()

    def _init_pyttsx3(self):
        try:
            self.pyttsx_engine = pyttsx3.init()
            self.pyttsx_engine.setProperty('rate', 140)
        except Exception:
            self.pyttsx_engine = None

    def set_language(self, lang_code):
        if lang_code in ["en", "hi"]:
            self.current_lang = lang_code

    def get_translated_text(self, gesture_name, lang_code=None):
        lang = lang_code or self.current_lang
        if gesture_name in GESTURE_TRANSLATIONS and lang in GESTURE_TRANSLATIONS[gesture_name]:
            return GESTURE_TRANSLATIONS[gesture_name][lang]
        return gesture_name.replace("_", " ").title()

    def speak(self, text, lang_code=None):
        """Asynchronously plays text speech in background thread without blocking UI."""
        lang = lang_code or self.current_lang
        t = threading.Thread(target=self._speak_worker, args=(text, lang), daemon=True)
        t.start()

    def _speak_worker(self, text, lang):
        with self.lock:
            if not text or text.lower() in ["idle", "आराम"]:
                return

            text_hash = hashlib.md5(f"{text}_{lang}".encode('utf-8')).hexdigest()
            cache_file = os.path.join(AUDIO_CACHE_DIR, f"{text_hash}.mp3")

            # Try gTTS online speech generation
            try:
                if not os.path.exists(cache_file):
                    tts = gTTS(text=text, lang=lang, slow=False)
                    tts.save(cache_file)
                
                # Play audio using Windows System command or PowerShell
                os.system(f'powershell -c "(New-Object Media.SoundPlayer \'{cache_file}\').PlaySync()" > NUL 2>&1')
                return
            except Exception:
                pass

            # Fallback to offline pyttsx3 engine
            if self.pyttsx_engine:
                try:
                    self.pyttsx_engine.say(text)
                    self.pyttsx_engine.runAndWait()
                except Exception:
                    pass
