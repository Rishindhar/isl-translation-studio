import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "audio_cache")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Feature Extraction & Model Hyperparameters
LANDMARK_POINTS = 21
COORDINATE_DIM = 3
FEATURE_DIM = LANDMARK_POINTS * COORDINATE_DIM  # 63 features per frame
SEQUENCE_LENGTH = 30                              # 30 frames (~1 sec motion window)

# Classification & Threshold Parameters
CONFIDENCE_THRESHOLD = 0.70                        # 70% probability threshold for multi-class
DEBOUNCE_FRAMES = 3                                # Require 3 consecutive frames before firing prediction
IDLE_VELOCITY_THRESHOLD = 0.005                   # Velocity threshold to skip inference when idle

# Expanded Indian Sign Language (ISL) Vocabulary (Alphabets, Numbers, Words)
ALPHABETS = [f"letter_{c}" for c in "abcdefghijklmnopqrstuvwxyz"]
NUMBERS = [f"number_{n}" for n in range(10)]
COMMON_WORDS = [
    "hello", "thank_you", "yes", "no", "help", "idle",
    "namaste", "welcome", "good_job", "please", "sorry", "goodbye",
    "gram", "loud_speaker", "water", "food", "father", "mother",
    "family", "home", "school", "friend", "love", "work",
    "time", "day", "night", "stop", "go", "read", "write"
]

DEFAULT_GESTURES = sorted(list(set(COMMON_WORDS + ALPHABETS + NUMBERS)))

# Multilingual Text-to-Speech Settings
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi"
}

# Multilingual Translations mapping for all gestures
GESTURE_TRANSLATIONS = {
    "hello": {"en": "Hello", "hi": "नमस्ते"},
    "thank_you": {"en": "Thank You", "hi": "धन्यवाद"},
    "yes": {"en": "Yes", "hi": "हाँ"},
    "no": {"en": "No", "hi": "नहीं"},
    "help": {"en": "Help", "hi": "मदद"},
    "idle": {"en": "Idle", "hi": "आराम"},
    "namaste": {"en": "Namaste", "hi": "नमस्ते"},
    "welcome": {"en": "Welcome", "hi": "स्वागत हे"},
    "good_job": {"en": "Good Job", "hi": "शाबाश"},
    "please": {"en": "Please", "hi": "कृपया"},
    "sorry": {"en": "Sorry", "hi": "क्षमा करें"},
    "goodbye": {"en": "Goodbye", "hi": "अलविदा"},
    "gram": {"en": "Gram / Unit", "hi": "ग्राम"},
    "loud_speaker": {"en": "Loudspeaker / Sound", "hi": "लाउडस्पीकर"},
    "water": {"en": "Water", "hi": "पानी"},
    "food": {"en": "Food / Eat", "hi": "खाना"},
    "father": {"en": "Father", "hi": "पिताजी"},
    "mother": {"en": "Mother", "hi": "माताजी"},
    "family": {"en": "Family", "hi": "परिवार"},
    "home": {"en": "Home", "hi": "घर"},
    "school": {"en": "School", "hi": "स्कूल"},
    "friend": {"en": "Friend", "hi": "मित्र"},
    "love": {"en": "Love", "hi": "प्यार"},
    "work": {"en": "Work / Job", "hi": "काम"},
    "time": {"en": "Time", "hi": "समय"},
    "day": {"en": "Day", "hi": "दिन"},
    "night": {"en": "Night", "hi": "रात"},
    "stop": {"en": "Stop", "hi": "रुको"},
    "go": {"en": "Go", "hi": "जाओ"},
    "read": {"en": "Read", "hi": "पढ़ो"},
    "write": {"en": "Write", "hi": "लिखो"},
    "hug": {"en": "Hug", "hi": "गले लगाना"},
    "little": {"en": "Little", "hi": "थोड़ा"},
    "above": {"en": "Above", "hi": "ऊपर"},
    "below": {"en": "Below", "hi": "नीचे"},
    "nice": {"en": "Nice", "hi": "अच्छा"},
    "hello_bro": {"en": "Hello Bro", "hi": "नमस्ते भाई"},
    "interview": {"en": "Interview", "hi": "साक्षात्कार"},
    "how_much": {"en": "How Much", "hi": "कितना"},
    "leave_preparatory_to_retirement": {"en": "Leave Preparatory to Retirement", "hi": "सेवानिवृत्ति पूर्व अवकाश"},
    "grandson": {"en": "Grandson", "hi": "पोता"},
    "granddaughter": {"en": "Granddaughter", "hi": "पोती"},
    "handkerchief": {"en": "Handkerchief", "hi": "रूमाल"},
    "gram_kilogram": {"en": "Gram Kilogram", "hi": "ग्राम किलोग्राम"}
}

# Populate translations for alphabets and numbers dynamically
for c in "abcdefghijklmnopqrstuvwxyz":
    key = f"letter_{c}"
    GESTURE_TRANSLATIONS[key] = {"en": f"Letter {c.upper()}", "hi": f"अक्षर {c.upper()}"}

for n in range(10):
    key = f"number_{n}"
    GESTURE_TRANSLATIONS[key] = {"en": f"Number {n}", "hi": f"संख्या {n}"}

MODEL_PATH = os.path.join(MODELS_DIR, "isl_gru_model.pt")
LABELS_PATH = os.path.join(MODELS_DIR, "gesture_labels.json")
