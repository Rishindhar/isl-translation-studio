// Full-Stack ISL Web Application Client Script
// Audio: Uses Browser Web Speech API (speechSynthesis) — guaranteed audible in Chrome/Edge

let activeInputFilename = null;
let currentTranslationEN = "Waiting for gesture...";
let currentTranslationHI = "इंतजार कर रहा हूं...";
let videoProcessingDone = false;
let hasShownComplete = false;
let lastSpokenText = "";
let isSpeaking = false;

// ─────────────────────────────────────────────────
// WEB SPEECH API — Plays audio directly in browser
// ─────────────────────────────────────────────────
function speakInBrowser(text, lang) {
    if (!text || text === 'idle' || text === 'Idle') return;
    if (!window.speechSynthesis) {
        console.warn('Web Speech API not supported in this browser.');
        return;
    }

    // Cancel any currently speaking utterance
    window.speechSynthesis.cancel();

    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = lang === 'hi' ? 'hi-IN' : 'en-US';
    utter.rate = 0.85;
    utter.pitch = 1.0;
    utter.volume = 1.0;

    // Pick best available voice
    const voices = window.speechSynthesis.getVoices();
    const targetLang = lang === 'hi' ? 'hi' : 'en';
    const match = voices.find(v => v.lang.startsWith(targetLang));
    if (match) utter.voice = match;

    isSpeaking = true;
    utter.onend = () => { isSpeaking = false; };
    utter.onerror = () => { isSpeaking = false; };

    window.speechSynthesis.speak(utter);
}

// Preload voices (required by some browsers)
if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = () => { window.speechSynthesis.getVoices(); };
}

// ─────────────────────────────────────────────────
// TAB SWITCHING
// ─────────────────────────────────────────────────
function switchTab(tabId) {
    // Stop camera automatically if switching tabs to prevent locking
    try {
        stopWebcam();
    } catch (e) {
        console.warn("Failed to stop webcam cleanly:", e);
    }

    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    const targetEl = document.getElementById(tabId);
    if (targetEl) targetEl.classList.add('active');
    
    // Find index of the clicked tab button
    let btnIndex = 0;
    if (tabId === 'videoTab') btnIndex = 0;
    else if (tabId === 'liveTab') btnIndex = 1;
    else if (tabId === 'learnTab') btnIndex = 2;

    const targetBtn = document.querySelectorAll('.tab-btn')[btnIndex];
    if (targetBtn) targetBtn.classList.add('active');
}

// ─────────────────────────────────────────────────
// LIVE WEBCAM — Browser Native Camera (getUserMedia)
// ─────────────────────────────────────────────────
let webcamStream = null;
let frameProcessingInterval = null;

async function startWebcam() {
    // Detect active tab to target the correct video/placeholder elements
    const isLearnActive = document.getElementById('learnTab').classList.contains('active');
    
    const video = isLearnActive 
        ? document.getElementById('learnWebcamVideo')
        : document.getElementById('webcamVideo');
        
    const placeholder = isLearnActive 
        ? document.getElementById('learnWebcamPlaceholder')
        : document.getElementById('webcamPlaceholder');
        
    const startBtn = isLearnActive 
        ? document.getElementById('learnStartCamBtn')
        : document.getElementById('startCamBtn');
        
    const stopBtn = isLearnActive 
        ? document.getElementById('learnStopCamBtn')
        : document.getElementById('stopCamBtn');

    const capCanvas = document.getElementById('captureCanvas');
    const ovCanvas  = document.getElementById('overlayCanvas');
    const hudOverlay = document.getElementById('hudOverlay');

    try {
        // Ask browser to open camera — this is GUARANTEED to show your actual webcam
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false
        });

        // Connect stream to video element
        video.srcObject = webcamStream;
        await video.play();

        // Size canvases to match video
        video.onloadedmetadata = () => {
            if (capCanvas) {
                capCanvas.width  = video.videoWidth;
                capCanvas.height = video.videoHeight;
            }
            if (ovCanvas) {
                ovCanvas.width   = video.videoWidth;
                ovCanvas.height  = video.videoHeight;
            }
        };

        // Show video, hide placeholder
        video.style.display = 'block';
        video.style.transform = 'scaleX(-1)'; // Mirror for natural view
        
        if (placeholder) placeholder.style.display = 'none';
        if (startBtn)    startBtn.style.display    = 'none';
        if (stopBtn)     stopBtn.style.display     = 'inline-block';

        // Start real-time recognition loop ONLY if on the Live tab
        if (!isLearnActive) {
            if (hudOverlay) hudOverlay.style.display = 'flex';
            frameProcessingInterval = setInterval(captureAndProcess, 65);
            speakInBrowser("Camera opened. Perform Indian Sign Language gestures to translate.", "en");
        } else {
            speakInBrowser("Camera opened. Enter a gesture name and click record to start.", "en");
        }

    } catch (err) {
        console.error('Camera access error:', err);
        let msg = 'Could not access camera. ';
        if (err.name === 'NotAllowedError') {
            msg += 'Please allow camera permission in your browser.';
        } else if (err.name === 'NotFoundError') {
            msg += 'No camera detected on this device.';
        } else {
            msg += err.message;
        }
        alert(msg);
    }
}

function stopWebcam() {
    // Stop all camera tracks
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    // Stop frame-sending loop
    if (frameProcessingInterval) {
        clearInterval(frameProcessingInterval);
        frameProcessingInterval = null;
    }

    // Reset both live and learn element groups
    const liveVid = document.getElementById('webcamVideo');
    const learnVid = document.getElementById('learnWebcamVideo');
    const livePlaceholder = document.getElementById('webcamPlaceholder');
    const learnPlaceholder = document.getElementById('learnWebcamPlaceholder');
    const liveStartBtn = document.getElementById('startCamBtn');
    const learnStartBtn = document.getElementById('learnStartCamBtn');
    const liveStopBtn = document.getElementById('stopCamBtn');
    const learnStopBtn = document.getElementById('learnStopCamBtn');
    const hudOverlay = document.getElementById('hudOverlay');

    if (liveVid) { liveVid.srcObject = null; liveVid.style.display = 'none'; }
    if (learnVid) { learnVid.srcObject = null; learnVid.style.display = 'none'; }
    
    if (livePlaceholder) livePlaceholder.style.display = 'flex';
    if (learnPlaceholder) learnPlaceholder.style.display = 'flex';
    
    if (liveStartBtn) liveStartBtn.style.display = 'inline-block';
    if (learnStartBtn) learnStartBtn.style.display = 'inline-block';
    
    if (liveStopBtn) liveStopBtn.style.display = 'none';
    if (learnStopBtn) learnStopBtn.style.display = 'none';
    
    if (hudOverlay) hudOverlay.style.display = 'none';
}

async function captureAndProcess() {
    const video = document.getElementById('webcamVideo');
    const capCanvas = document.getElementById('captureCanvas');
    if (!video || !capCanvas || !video.videoWidth) return;

    // Draw current video frame (un-mirrored) onto canvas for server
    const ctx = capCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);

    // Get base64 JPEG
    const frameB64 = capCanvas.toDataURL('image/jpeg', 0.6);

    try {
        const resp = await fetch('/api/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: frameB64 })
        });
        if (!resp.ok) return;
        const data = await resp.json();

        // ── Update HUD ────────────────────────────────────────────────
        const pred = data.prediction || 'idle';
        const conf = data.confidence || 0;
        const predEl  = document.getElementById('livePredLabel');
        const confEl  = document.getElementById('liveConfVal');
        const meterEl = document.getElementById('liveMeterFill');

        if (predEl) predEl.innerText = pred.toUpperCase().replace(/_/g, ' ');
        if (confEl) confEl.innerText = conf.toFixed(1) + '%';
        if (meterEl) meterEl.style.width = Math.min(conf, 100) + '%';

        // ── Update in-progress sentence word chips ──────────────────
        if (data.sentence && data.sentence.length > 0) {
            const box = document.getElementById('liveSentenceBox');
            if (box) {
                box.innerHTML = data.sentence.map(w =>
                    `<span class="word-chip">${w.replace(/_/g,' ')}</span>`
                ).join('');
            }
        }

        // ── Live translation preview ─────────────────────────────────
        if (data.translation_en) {
            const enEl = document.getElementById('liveTransEN');
            if (enEl) enEl.innerText = data.translation_en;
        }
        if (data.translation_hi) {
            const hiEl = document.getElementById('liveTransHI');
            if (hiEl) hiEl.innerText = data.translation_hi;
        }

        // ── LM Next-word hints ───────────────────────────────────────
        const lmHintEl = document.getElementById('lmNextHints');
        if (lmHintEl && data.lm_next && data.lm_next.length > 0) {
            lmHintEl.innerHTML = '🔮 Next: ' + data.lm_next.map(w =>
                `<span class="lm-hint-chip">${w.replace(/_/g,' ')}</span>`
            ).join('');
            lmHintEl.style.display = 'block';
        } else if (lmHintEl) {
            lmHintEl.style.display = 'none';
        }

        // ── Sentence CLOSED: show completed sentence banner ──────────
        if (data.sentence_closed && data.completed_sentence_en) {
            showCompletedSentenceBanner(
                data.completed_sentence_en,
                data.completed_sentence_hi,
                data.completed_words || []
            );
            // Clear in-progress word chips
            const box = document.getElementById('liveSentenceBox');
            if (box) box.innerHTML = '';
            const lmH = document.getElementById('lmNextHints');
            if (lmH) lmH.style.display = 'none';
        }

    } catch (err) {
        // Silently ignore transient network errors
    }
}

// ─────────────────────────────────────────────────
// SAMPLE VIDEO LOADER
// ─────────────────────────────────────────────────
async function loadSampleVideos() {
    try {
        const res = await fetch('/api/list_videos');
        const data = await res.json();
        const select = document.getElementById('sampleVideoSelect');
        select.innerHTML = '<option value="">-- Choose Sample Video --</option>';
        if (data.videos && data.videos.length > 0) {
            data.videos.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.filename;
                opt.innerText = `🎬 ${v.filename}`;
                select.appendChild(opt);
            });
        }
    } catch (err) { console.error('Failed to load video list:', err); }
}

function selectSampleVideo(filename) {
    if (!filename) return;
    activeInputFilename = filename;
    loadInputVideoPreview(filename);
}

function loadInputVideoPreview(filename) {
    const previewPlayer = document.getElementById('inputVideoPreview');
    const placeholder = document.getElementById('previewPlaceholder');
    previewPlayer.src = `/api/video_file/${filename}`;
    previewPlayer.style.display = 'block';
    placeholder.style.display = 'none';
    previewPlayer.load();
    document.getElementById('uploadStatusText').innerText = `Loaded '${filename}'. Click 'Proceed & Translate Video' to process.`;
}

function triggerFileInput() {
    document.getElementById('videoFileInput').click();
}

// ─────────────────────────────────────────────────
// FILE UPLOAD
// ─────────────────────────────────────────────────
async function handleFileUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    const formData = new FormData();
    formData.append('video_file', files[0]);
    document.getElementById('uploadStatusText').innerText = 'Uploading video file...';
    try {
        const response = await fetch('/api/upload_video', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.status === 'success') {
            activeInputFilename = data.filename;
            loadInputVideoPreview(data.filename);
            await loadSampleVideos();
            document.getElementById('sampleVideoSelect').value = data.filename;
        } else {
            alert('Upload failed: ' + data.message);
            document.getElementById('uploadStatusText').innerText = '';
        }
    } catch (err) {
        console.error('Upload error:', err);
        alert('Error uploading video file.');
        document.getElementById('uploadStatusText').innerText = '';
    }
}

// ─────────────────────────────────────────────────
// PROCEED & TRANSLATE
// ─────────────────────────────────────────────────
function proceedAndTranslateVideo() {
    if (!activeInputFilename) {
        alert('Please select a sample video or upload a video file first.');
        return;
    }
    const selectedLang = document.getElementById('videoLangSelect').value;
    const playerImg = document.getElementById('translatedStreamPlayer');
    const placeholder = document.getElementById('streamPlaceholder');
    const completeBanner = document.getElementById('videoCompleteBanner');
    const vocabWarning = document.getElementById('vocabWarningBanner');

    // Reset state
    videoProcessingDone = false;
    hasShownComplete = false;
    lastSpokenText = "";
    if (completeBanner) completeBanner.style.display = 'none';
    if (vocabWarning) vocabWarning.style.display = 'none';

    const streamUrl = `/video_stream_feed?filename=${encodeURIComponent(activeInputFilename)}&lang=${selectedLang}&t=${Date.now()}`;
    playerImg.src = streamUrl;
    playerImg.style.display = 'block';
    placeholder.style.display = 'none';
    document.getElementById('uploadStatusText').innerText = `Processing '${activeInputFilename}'...`;
}

// ─────────────────────────────────────────────────
// PLAY AUDIO BUTTONS — Browser Speech API
// ─────────────────────────────────────────────────
function playAudio(lang) {
    const text = lang === 'en' ? currentTranslationEN : currentTranslationHI;
    if (!text || text.includes('Waiting') || text.includes('इंतजार')) {
        speakInBrowser('No translation available yet. Please process a video first.', 'en');
        return;
    }
    speakInBrowser(text, lang);
}

// ─────────────────────────────────────────────────
// STATUS POLLING — Updates UI every 400ms
// ─────────────────────────────────────────────────
let prevSentenceLen = 0;

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) return;
        const data = await response.json();

        // Translation cards
        if (data.translation_en) {
            currentTranslationEN = data.translation_en;
            const textEnEl = document.getElementById('transTextEN');
            if (textEnEl) textEnEl.innerText = data.translation_en;
        }
        if (data.translation_hi) {
            currentTranslationHI = data.translation_hi;
            const textHiEl = document.getElementById('transTextHI');
            if (textHiEl) textHiEl.innerText = data.translation_hi;
        }

        // Sentence chips + real-time browser speech
        if (data.sentence && data.sentence.length > 0) {
            const html = data.sentence.map(w => `<span class="word-chip">${w}</span>`).join('');
            const sb = document.getElementById('sentenceBox');
            const lb = document.getElementById('liveSentenceBox');
            if (sb) sb.innerHTML = html;
            if (lb) lb.innerHTML = html;

            // Speak new words as they appear (browser TTS)
            if (data.sentence.length > prevSentenceLen) {
                const newWord = data.sentence[data.sentence.length - 1];
                const selectedLang = document.getElementById('videoLangSelect')
                    ? document.getElementById('videoLangSelect').value : 'en';
                if (newWord && newWord !== lastSpokenText && !isSpeaking) {
                    lastSpokenText = newWord;
                    speakInBrowser(newWord, selectedLang);
                }
            }
            prevSentenceLen = data.sentence.length;
        }

        // Vocabulary tags
        if (data.vocabulary) {
            const vc = document.getElementById('vocabCount');
            const vt = document.getElementById('vocabTags');
            if (vc) vc.innerText = data.vocabulary.length;
            if (vt) vt.innerHTML = data.vocabulary.map(v => `<span class="tag">${v}</span>`).join('');

            // Also update the warning banner dynamically
            const warnCount = document.getElementById('warnVocabCount');
            const warnDetail = document.getElementById('warnVocabDetail');
            if (warnCount) warnCount.innerText = data.vocabulary.length;
            if (warnDetail && data.vocabulary.length > 5) {
                // Show first 6 gestures and total count
                const sampleGestures = data.vocabulary.slice(0, 6).join(', ');
                const othersCount = data.vocabulary.length - 6;
                warnDetail.innerHTML = `Includes: <strong>${sampleGestures}</strong>, and ${othersCount} other gestures.`;
            }
        }

        // Video done
        if (data.video_done && !hasShownComplete) {
            hasShownComplete = true;
            prevSentenceLen = 0;
            showVideoCompletePanel(data.video_sentence || data.sentence || [], data.language);
        }

    } catch (err) { console.warn('Status poll error:', err); }
}

// ─────────────────────────────────────────────────
// VIDEO COMPLETE PANEL + FULL SENTENCE AUDIO
// ─────────────────────────────────────────────────
function showVideoCompletePanel(sentenceWords, lang) {
    const banner = document.getElementById('videoCompleteBanner');
    const summaryEl = document.getElementById('completeSummaryText');
    const langEl = document.getElementById('completeLang');
    const selectedLang = document.getElementById('videoLangSelect')
        ? document.getElementById('videoLangSelect').value : (lang || 'en');

    if (banner && summaryEl) {
        const summaryText = sentenceWords.length > 0
            ? sentenceWords.map(w => w.replace(/_/g, ' ')).join(' | ')
            : 'No ISL signs recognized in this video';
        summaryEl.innerText = summaryText;
        if (langEl) langEl.innerText = selectedLang === 'hi' ? 'Hindi (हिन्दी)' : 'English';
        banner.style.display = 'block';

        // Speak full sentence via browser TTS after 1 second
        setTimeout(() => {
            if (sentenceWords.length > 0) {
                const fullText = sentenceWords.map(w => w.replace(/_/g, ' ')).join(', ');
                speakInBrowser(fullText, selectedLang);
            } else {
                speakInBrowser('Video processing complete. No signs detected.', 'en');
            }
        }, 800);
    }

    // Show vocabulary warning if no signs detected
    const vocabWarning = document.getElementById('vocabWarningBanner');
    if (vocabWarning && sentenceWords.length === 0) {
        vocabWarning.style.display = 'block';
    }

    document.getElementById('uploadStatusText').innerText = 'Processing complete.';
}

async function changeLanguage(langCode) {
    try {
        await fetch('/api/set_language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: langCode })
        });
    } catch (err) { console.error('Failed to change language:', err); }
}

async function resetSentence() {
    try {
        prevSentenceLen = 0;
        lastSpokenText = "";
        hasShownComplete = false;
        const banner = document.getElementById('videoCompleteBanner');
        const warn = document.getElementById('vocabWarningBanner');
        if (banner) banner.style.display = 'none';
        if (warn) warn.style.display = 'none';
        await fetch('/api/reset_sentence', { method: 'POST' });
        updateStatus();
    } catch (err) { console.error('Failed to reset sentence:', err); }
}

// ─────────────────────────────────────────────────
// CUSTOM SIGN LEARNER — Record & Fine-Tune
// ─────────────────────────────────────────────────
async function recordWebcamSign() {
    const signInput = document.getElementById('customSignInput');
    const statusEl = document.getElementById('customLearnStatus');
    const labelName = signInput ? signInput.value.trim() : '';

    if (!labelName) {
        alert("Please enter a name for the custom sign (e.g. namaste, water, help).");
        return;
    }

    // 1. Ensure camera is running
    if (!webcamStream) {
        if (statusEl) {
            statusEl.style.color = '#3b82f6';
            statusEl.innerText = "⏳ Opening camera first...";
        }
        await startWebcam();
        await new Promise(r => setTimeout(r, 1500)); // wait for camera to warm up
    }

    const isLearnActive = document.getElementById('learnTab').classList.contains('active');
    const video = isLearnActive 
        ? document.getElementById('learnWebcamVideo')
        : document.getElementById('webcamVideo');
    const capCanvas = document.getElementById('captureCanvas');
    if (!video || !capCanvas || !video.videoWidth) {
        alert("Webcam is not ready yet. Please try to 'Open Camera' first.");
        return;
    }

    // Temporarily pause the real-time inference loop while recording
    if (frameProcessingInterval) {
        clearInterval(frameProcessingInterval);
        frameProcessingInterval = null;
    }

    // 2. Countdown loop (3, 2, 1) so the user gets ready
    for (let c = 3; c > 0; c--) {
        if (statusEl) {
            statusEl.style.color = '#eab308';
            statusEl.innerText = `⏳ Get ready to perform sign '${labelName}' in ${c} seconds...`;
        }
        speakInBrowser(c.toString(), 'en');
        await new Promise(r => setTimeout(r, 1000));
    }

    if (statusEl) {
        statusEl.style.color = '#ef4444';
        statusEl.innerText = `🔴 RECORDING NOW! Perform gesture for '${labelName}'...`;
    }
    speakInBrowser("Record now!", 'en');

    // 3. Capture 30 frames at 15 FPS (one frame every 70ms)
    const recordedFrames = [];
    const ctx = capCanvas.getContext('2d');
    
    for (let i = 0; i < 30; i++) {
        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
        // Save as base64 JPEG
        const b64 = capCanvas.toDataURL('image/jpeg', 0.6);
        recordedFrames.push(b64);
        
        // Show progress bar
        if (statusEl) {
            const pct = Math.round(((i + 1) / 30) * 100);
            const progress = "█".repeat(Math.round(pct / 10)) + "░".repeat(10 - Math.round(pct / 10));
            statusEl.innerText = `🔴 Capturing: [${progress}] ${pct}%`;
        }
        
        await new Promise(r => setTimeout(r, 70));
    }

    // Restart the real-time inference loop if on live tab
    if (!isLearnActive) {
        frameProcessingInterval = setInterval(captureAndProcess, 300);
    }

    if (statusEl) {
        statusEl.style.color = '#3b82f6';
        statusEl.innerText = "⏳ Processing landmarks and training AI model...";
    }

    // 4. Send base64 frames to server for learning
    try {
        const response = await fetch('/api/record_webcam_sign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                label: labelName,
                frames: recordedFrames
            })
        });
        const data = await response.json();

        if (data.status === 'success') {
            if (statusEl) {
                statusEl.style.color = '#22c55e';
                statusEl.innerText = `✅ SUCCESS: Learned sign '${data.label}'! Model updated.`;
            }
            speakInBrowser(`Successfully learned custom sign ${data.label}!`, 'en');
            updateStatus();
            loadCustomSigns();
        } else {
            if (statusEl) {
                statusEl.style.color = '#ef4444';
                statusEl.innerText = `❌ Error: ${data.message}`;
            }
            alert(data.message);
        }
    } catch (err) {
        console.error('Record webcam sign error:', err);
        if (statusEl) {
            statusEl.style.color = '#ef4444';
            statusEl.innerText = `❌ Connection error during model training.`;
        }
    }
}

// ─────────────────────────────────────────────────
// COMPLETED SENTENCE BANNER & HISTORY
// ─────────────────────────────────────────────────
const completedSentenceHistory = [];

function showCompletedSentenceBanner(enText, hiText, words) {
    // ── Live Camera tab banner ────────────────────────────────────
    const banner = document.getElementById('liveCompletedBanner');
    const bannerEN = document.getElementById('liveCompletedEN');
    const bannerHI = document.getElementById('liveCompletedHI');
    const bannerWords = document.getElementById('liveCompletedWords');

    if (banner && enText) {
        if (bannerEN) bannerEN.innerText = enText;
        if (bannerHI && hiText) bannerHI.innerText = hiText;
        if (bannerWords && words.length > 0) {
            bannerWords.innerHTML = words.map(w =>
                `<span class="word-chip completed-chip">${w.replace(/_/g,' ')}</span>`
            ).join('');
        }
        banner.style.display = 'block';
        banner.classList.add('sentence-flash');
        setTimeout(() => {
            banner.classList.remove('sentence-flash');
        }, 1500);

        // Speak the completed sentence
        speakInBrowser(enText, 'en');
    }

    // ── Add to history panel ─────────────────────────────────────
    completedSentenceHistory.unshift({ en: enText, hi: hiText, words });
    if (completedSentenceHistory.length > 5) completedSentenceHistory.pop();

    const historyEl = document.getElementById('sentenceHistoryPanel');
    if (historyEl) {
        historyEl.innerHTML = completedSentenceHistory.map((s, i) =>
            `<div class="history-entry ${i===0 ? 'latest' : ''}">
                <span class="history-idx">#${completedSentenceHistory.length - i}</span>
                <span class="history-en">${s.en}</span>
                ${s.hi ? `<span class="history-hi">${s.hi}</span>` : ''}
             </div>`
        ).join('');
    }
}

// ─────────────────────────────────────────────────
// CUSTOM SIGNS MANAGEMENT (LIST & DELETE)
// ─────────────────────────────────────────────────
async function loadCustomSigns() {
    const listEl = document.getElementById('manageCustomSignsList');
    if (!listEl) return;

    try {
        const resp = await fetch('/api/get_custom_signs');
        if (!resp.ok) return;
        const data = await resp.json();

        if (data.status === 'success' && data.custom_signs && data.custom_signs.length > 0) {
            listEl.innerHTML = data.custom_signs.map(sign =>
                `<span onclick="document.getElementById('deleteSignInput').value='${sign}'"
                    style="background: rgba(147,51,234,0.15); border: 1px solid var(--accent-color);
                           padding: 8px 14px; display: inline-flex; align-items: center; gap: 6px;
                           border-radius: var(--radius-sm); cursor: pointer; font-weight: 600;
                           color: var(--text-primary); font-size: 13px;"
                    title="Click to fill delete box">
                    ${sign.toUpperCase().replace(/_/g, ' ')}
                </span>`
            ).join('');
        } else {
            listEl.innerHTML = `<span style="font-size:0.8rem; color: var(--text-muted);">No custom signs found.</span>`;
        }
    } catch (err) {
        console.error('Failed to load custom signs:', err);
    }
}

async function deleteTypedCustomSign() {
    const input = document.getElementById('deleteSignInput');
    if (!input) return;
    const label = input.value.trim().toLowerCase();
    if (!label) {
        alert('Please type the name of the sign you want to delete.');
        return;
    }
    if (!confirm(`Are you sure you want to permanently delete the custom sign "${label}" from the AI model and disk?`)) {
        return;
    }
    await deleteCustomSign(label);
    input.value = '';
}

async function deleteCustomSign(label) {
    try {
        const resp = await fetch('/api/delete_custom_sign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: label })
        });
        const data = await resp.json();

        if (resp.ok && data.status === 'success') {
            alert(`✅ ${data.message}`);
            loadCustomSigns();
            updateStatus();
        } else {
            alert('❌ Error: ' + data.message);
        }
    } catch (err) {
        alert('Failed to connect to server: ' + err.message);
    }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    loadSampleVideos();
    updateStatus();
    loadCustomSigns();
    if (window.speechSynthesis) window.speechSynthesis.getVoices();
});
setInterval(updateStatus, 400);

