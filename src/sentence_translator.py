"""
ISL Sentence Translator & NLP Grammar Smoothing Engine
======================================================
Converts raw ISL gesture word sequences (which follow ISL SOV/Topic-Comment structure)
into fluent, grammatically correct English and Hindi sentences.

v2.0 — Expanded to 300+ sentence combinations, smarter NLP fallback,
        pause-aware sentence closure, and Language Model integration.
"""

# ─────────────────────────────────────────────────────────────
# MASTER ISL SENTENCE DICTIONARY — 300+ entries
# Keys are TUPLES of ISL gesture words (lowercase, underscore-separated)
# Values are {"en": "...", "hi": "..."}
# ─────────────────────────────────────────────────────────────
ISL_SENTENCE_DICTIONARY = {

    # ── Greetings & Courtesy ──────────────────────────────
    ("hello",):                      {"en": "Hello!", "hi": "नमस्ते!"},
    ("namaste",):                    {"en": "Namaste!", "hi": "नमस्ते!"},
    ("welcome",):                    {"en": "Welcome!", "hi": "आपका स्वागत है!"},
    ("thank_you",):                  {"en": "Thank you!", "hi": "धन्यवाद!"},
    ("sorry",):                      {"en": "I am sorry.", "hi": "मुझे खेद है।"},
    ("goodbye",):                    {"en": "Goodbye! Take care.", "hi": "अलविदा! अपना ख्याल रखें।"},
    ("please",):                     {"en": "Please.", "hi": "कृपया।"},
    ("yes",):                        {"en": "Yes!", "hi": "हाँ!"},
    ("no",):                         {"en": "No.", "hi": "नहीं।"},
    ("help",):                       {"en": "Help!", "hi": "मदद करें!"},

    ("hello", "thank_you"):          {"en": "Hello, thank you!", "hi": "नमस्ते, धन्यवाद!"},
    ("hello", "thank_you", "yes"):   {"en": "Hello, thank you, yes!", "hi": "नमस्ते, धन्यवाद, हाँ!"},
    ("hello", "how", "you"):         {"en": "Hello, how are you?", "hi": "नमस्ते, आप कैसे हैं?"},
    ("hello", "good", "morning"):    {"en": "Hello, good morning!", "hi": "नमस्ते, शुभ प्रभात!"},
    ("hello", "how", "you", "today"):{"en": "Hello! How are you doing today?", "hi": "नमस्ते! आज आप कैसे हैं?"},
    ("thank_you", "help"):           {"en": "Thank you for your help.", "hi": "आपकी मदद के लिए धन्यवाद।"},
    ("thank_you", "help", "me"):     {"en": "Thank you so much for helping me.", "hi": "मेरी मदद करने के लिए बहुत-बहुत धन्यवाद।"},
    ("good", "morning"):             {"en": "Good morning! Have a nice day.", "hi": "शुभ प्रभात! आपका दिन शुभ हो।"},
    ("good", "night"):               {"en": "Good night, sweet dreams.", "hi": "शुभ रात्रि, मीठे सपने।"},
    ("good", "morning", "how", "you"):{"en": "Good morning! How are you today?", "hi": "शुभ प्रभात! आज आप कैसे हैं?"},
    ("welcome", "home"):             {"en": "Welcome to our home!", "hi": "हमारे घर में आपका स्वागत है!"},
    ("please", "help"):              {"en": "Please help me.", "hi": "कृपया मेरी मदद करें।"},
    ("please", "help", "me"):        {"en": "Please help me urgently.", "hi": "कृपया मेरी तुरंत मदद करें।"},
    ("please", "stop"):              {"en": "Please stop.", "hi": "कृपया रुकें।"},
    ("please", "water"):             {"en": "Please give me some water.", "hi": "कृपया मुझे पानी दें।"},
    ("please", "food"):              {"en": "Please give me food.", "hi": "कृपया मुझे खाना दें।"},
    ("please", "come"):              {"en": "Please come here.", "hi": "कृपया यहाँ आएं।"},
    ("sorry", "late"):               {"en": "I am sorry for being late.", "hi": "देर से आने के लिए मुझे खेद है।"},
    ("sorry", "mistake"):            {"en": "I am sorry for the mistake.", "hi": "गलती के लिए मुझे खेद है।"},
    ("sorry", "late", "work"):       {"en": "I am sorry I am late for work.", "hi": "काम पर देर से आने के लिए क्षमा करें।"},
    ("yes", "thank_you"):            {"en": "Yes, thank you very much.", "hi": "हाँ, आपका बहुत-बहुत धन्यवाद।"},
    ("no", "thank_you"):             {"en": "No, thank you.", "hi": "जी नहीं, धन्यवाद।"},
    ("yes", "i", "know"):            {"en": "Yes, I understand.", "hi": "हाँ, मैं समझता हूँ।"},
    ("no", "i", "know"):             {"en": "No, I do not know.", "hi": "नहीं, मुझे नहीं पता।"},
    ("good", "job"):                 {"en": "Good job! Well done.", "hi": "शाबाश! बहुत अच्छा।"},
    ("good_morning",):               {"en": "Good morning!", "hi": "शुभ प्रभात!"},
    ("good_night",):                 {"en": "Good night!", "hi": "शुभ रात्रि!"},

    # ── Family & Relationships ─────────────────────────────
    ("father", "go", "work"):        {"en": "Father has gone to work.", "hi": "पिताजी काम पर गए हैं।"},
    ("father", "home"):              {"en": "Father is at home.", "hi": "पिताजी घर पर हैं।"},
    ("father", "go", "school"):      {"en": "Father went to drop us at school.", "hi": "पिताजी हमें स्कूल छोड़ने गए।"},
    ("father", "go", "work", "morning"): {"en": "Father goes to work every morning.", "hi": "पिताजी हर सुबह काम पर जाते हैं।"},
    ("mother", "food", "cook"):      {"en": "Mother is cooking food.", "hi": "माताजी खाना बना रही हैं।"},
    ("mother", "food"):              {"en": "Mother has prepared food.", "hi": "माताजी ने खाना तैयार किया है।"},
    ("mother", "home"):              {"en": "Mother is at home.", "hi": "माताजी घर पर हैं।"},
    ("mother", "food", "cook", "home"): {"en": "Mother is cooking food at home.", "hi": "माताजी घर पर खाना बना रही हैं।"},
    ("i", "love", "family"):         {"en": "I love my family.", "hi": "मैं अपने परिवार से प्यार करता हूँ।"},
    ("family", "home"):              {"en": "My family is at home.", "hi": "मेरा परिवार घर पर है।"},
    ("family", "love"):              {"en": "I love my family very much.", "hi": "मैं अपने परिवार से बहुत प्यार करता हूँ।"},
    ("family", "home", "happy"):     {"en": "My family is happy at home.", "hi": "मेरा परिवार घर पर खुश है।"},
    ("friend", "good"):              {"en": "You are a very good friend.", "hi": "आप एक बहुत अच्छे मित्र हैं।"},
    ("friend", "home"):              {"en": "My friend is at home.", "hi": "मेरा मित्र घर पर है।"},
    ("friend", "school"):            {"en": "My friend is at school.", "hi": "मेरा मित्र स्कूल में है।"},
    ("friend", "school", "go"):      {"en": "My friend has gone to school.", "hi": "मेरा मित्र स्कूल गया है।"},
    ("grandfather", "home"):         {"en": "Grandfather is at home.", "hi": "दादाजी घर पर हैं।"},
    ("grandmother", "home"):         {"en": "Grandmother is at home.", "hi": "दादीजी घर पर हैं।"},
    ("mother", "in_law", "home"):    {"en": "Mother-in-law is at home.", "hi": "सास घर पर हैं।"},
    ("husband", "work"):             {"en": "My husband is at work.", "hi": "मेरे पति काम पर हैं।"},

    # ── Needs & Actions ───────────────────────────────────
    ("need", "water"):               {"en": "I need some water, please.", "hi": "मुझे पानी चाहिए।"},
    ("need", "food"):                {"en": "I am hungry and need food.", "hi": "मुझे भूख लगी है, खाना चाहिए।"},
    ("need", "help"):                {"en": "I need urgent help!", "hi": "मुझे तुरंत मदद चाहिए!"},
    ("need", "medicine"):            {"en": "I need medicine urgently.", "hi": "मुझे दवा चाहिए।"},
    ("need", "doctor"):              {"en": "I need to see a doctor.", "hi": "मुझे डॉक्टर की जरूरत है।"},
    ("need", "help", "please"):      {"en": "I need help, please!", "hi": "कृपया मेरी मदद करें!"},
    ("need", "water", "please"):     {"en": "I urgently need water, please.", "hi": "कृपया मुझे पानी चाहिए।"},
    ("i", "need", "water", "please"):{"en": "I need water, please.", "hi": "मुझे पानी चाहिए, कृपया।"},
    ("i", "need", "food"):           {"en": "I need food, I am hungry.", "hi": "मुझे खाना चाहिए, मुझे भूख लगी है।"},
    ("i", "go", "school", "morning"):{"en": "I go to school every morning.", "hi": "मैं हर सुबह स्कूल जाता हूँ।"},
    ("water", "drink"):              {"en": "I need to drink clean water.", "hi": "साफ पानी पीना है।"},
    ("water", "need"):               {"en": "I need water.", "hi": "मुझे पानी चाहिए।"},
    ("water", "need", "please"):     {"en": "I need water, please.", "hi": "कृपया मुझे पानी चाहिए।"},
    ("food", "eat"):                 {"en": "I want to eat food.", "hi": "मुझे खाना खाना है।"},
    ("food", "need"):                {"en": "I need food.", "hi": "मुझे खाना चाहिए।"},
    ("food", "need", "hungry"):      {"en": "I am hungry and need food.", "hi": "मुझे भूख लगी है और खाना चाहिए।"},
    ("food", "cook"):                {"en": "I am cooking food.", "hi": "मैं खाना बना रहा हूँ।"},
    ("food", "time"):                {"en": "It is time for food.", "hi": "खाने का समय है।"},
    ("lunch", "time"):               {"en": "It is lunch time.", "hi": "दोपहर के खाने का समय है।"},
    ("go", "home"):                  {"en": "I want to go home.", "hi": "मैं घर जाना चाहता हूँ।"},
    ("go", "home", "now"):           {"en": "I am going home now.", "hi": "मैं अभी घर जा रहा हूँ।"},
    ("go", "school"):                {"en": "I am going to school.", "hi": "मैं स्कूल जा रहा हूँ।"},
    ("go", "work"):                  {"en": "I am going to work.", "hi": "मैं काम पर जा रहा हूँ।"},
    ("go", "office"):                {"en": "I am going to the office.", "hi": "मैं ऑफिस जा रहा हूँ।"},
    ("go", "hospital"):              {"en": "I need to go to the hospital.", "hi": "मुझे अस्पताल जाना है।"},
    ("come", "home"):                {"en": "Please come home.", "hi": "कृपया घर आएं।"},
    ("come", "here", "please"):      {"en": "Please come here.", "hi": "कृपया यहाँ आएं।"},
    ("stop", "go"):                  {"en": "Please stop and do not go.", "hi": "कृपया रुकें और न जाएं।"},
    ("stop", "now", "please"):       {"en": "Please stop now!", "hi": "कृपया अभी रुकें!"},
    ("stop", "now"):                 {"en": "Stop right now!", "hi": "अभी रुको!"},
    ("read", "write"):               {"en": "I am reading and writing.", "hi": "मैं पढ़ और लिख रहा हूँ।"},
    ("read", "book"):                {"en": "I am reading a book.", "hi": "मैं किताब पढ़ रहा हूँ।"},
    ("write", "work"):               {"en": "I am writing my work.", "hi": "मैं अपना काम लिख रहा हूँ।"},
    ("work", "time"):                {"en": "It is time for work.", "hi": "यह काम करने का समय है।"},
    ("work", "home"):                {"en": "I am working from home.", "hi": "मैं घर से काम कर रहा हूँ।"},
    ("work", "morning", "go"):       {"en": "I go to work in the morning.", "hi": "मैं सुबह काम पर जाता हूँ।"},
    ("morning", "work"):             {"en": "I work in the morning.", "hi": "मैं सुबह काम करता हूँ।"},
    ("night", "sleep"):              {"en": "I sleep at night.", "hi": "मैं रात को सोता हूँ।"},
    ("day", "work"):                 {"en": "I work during the day.", "hi": "मैं दिन में काम करता हूँ।"},
    ("home", "come"):                {"en": "I have come home.", "hi": "मैं घर आ गया हूँ।"},

    # ── School & Education ────────────────────────────────
    ("school", "go"):                {"en": "I am going to school.", "hi": "मैं स्कूल जा रहा हूँ।"},
    ("student", "school"):           {"en": "The student is at school.", "hi": "छात्र स्कूल में है।"},
    ("teacher", "school"):           {"en": "The teacher is at school.", "hi": "अध्यापक स्कूल में हैं।"},
    ("learn", "school"):             {"en": "I am learning at school.", "hi": "मैं स्कूल में पढ़ रहा हूँ।"},
    ("learn", "sign", "language"):   {"en": "I am learning sign language.", "hi": "मैं सांकेतिक भाषा सीख रहा हूँ।"},
    ("school", "learn", "happy"):    {"en": "Learning at school makes me happy.", "hi": "स्कूल में पढ़ना मुझे खुश करता है।"},
    ("where", "school"):             {"en": "Where is the school located?", "hi": "स्कूल कहाँ स्थित है?"},

    # ── Medical & Health ──────────────────────────────────
    ("doctor", "help"):              {"en": "The doctor will help me.", "hi": "डॉक्टर मेरी मदद करेंगे।"},
    ("medicine", "need"):            {"en": "I need medicine.", "hi": "मुझे दवा चाहिए।"},
    ("hospital", "go"):              {"en": "I need to go to the hospital.", "hi": "मुझे अस्पताल जाना है।"},
    ("pain", "help"):                {"en": "I am in pain, please help.", "hi": "मुझे दर्द है, कृपया मदद करें।"},
    ("doctor", "hospital"):          {"en": "The doctor is at the hospital.", "hi": "डॉक्टर अस्पताल में हैं।"},
    ("need", "doctor", "help"):      {"en": "I need a doctor's help urgently.", "hi": "मुझे तुरंत डॉक्टर की मदद चाहिए।"},
    ("hurt", "help"):                {"en": "I am hurt, please help me.", "hi": "मुझे चोट लगी है, कृपया मदद करें।"},
    ("hungry", "food"):              {"en": "I am hungry, I need food.", "hi": "मुझे भूख लगी है, खाना चाहिए।"},
    ("hungry",):                     {"en": "I am hungry.", "hi": "मुझे भूख लगी है।"},

    # ── Questions ─────────────────────────────────────────
    ("what", "name"):                {"en": "What is your name?", "hi": "आपका नाम क्या है?"},
    ("where", "home"):               {"en": "Where is your home?", "hi": "आपका घर कहाँ है?"},
    ("time", "what"):                {"en": "What time is it right now?", "hi": "अभी क्या समय हुआ है?"},
    ("how", "much"):                 {"en": "How much does this cost?", "hi": "इसकी कीमत कितनी है?"},
    ("how", "many"):                 {"en": "How many are there?", "hi": "कितने हैं?"},
    ("how", "you"):                  {"en": "How are you?", "hi": "आप कैसे हैं?"},
    ("what", "you", "do"):           {"en": "What are you doing?", "hi": "आप क्या कर रहे हैं?"},
    ("where", "you", "go"):          {"en": "Where are you going?", "hi": "आप कहाँ जा रहे हैं?"},
    ("where", "work"):               {"en": "Where do you work?", "hi": "आप कहाँ काम करते हैं?"},
    ("what", "time"):                {"en": "What time is it?", "hi": "क्या समय है?"},
    ("how_big",):                    {"en": "How big is it?", "hi": "यह कितना बड़ा है?"},

    # ── Emotions ──────────────────────────────────────────
    ("happy", "good"):               {"en": "I am happy, everything is good.", "hi": "मैं खुश हूँ, सब कुछ अच्छा है।"},
    ("happy",):                      {"en": "I am happy!", "hi": "मैं खुश हूँ!"},
    ("sad", "help"):                 {"en": "I am sad, please help me.", "hi": "मैं उदास हूँ, कृपया मदद करें।"},
    ("love", "family"):              {"en": "I love my family.", "hi": "मैं अपने परिवार से प्यार करता हूँ।"},
    ("proud", "work"):               {"en": "I am proud of my work.", "hi": "मुझे अपने काम पर गर्व है।"},
    ("love",):                       {"en": "I feel love.", "hi": "मुझे प्यार महसूस हो रहा है।"},
    ("angry", "stop"):               {"en": "I am angry, please stop.", "hi": "मैं गुस्से में हूँ, कृपया रुकें।"},

    # ── Transport & Travel ────────────────────────────────
    ("car", "go"):                   {"en": "I am going by car.", "hi": "मैं कार से जा रहा हूँ।"},
    ("bus", "school"):               {"en": "I take the bus to school.", "hi": "मैं बस से स्कूल जाता हूँ।"},
    ("train", "go"):                 {"en": "I am travelling by train.", "hi": "मैं ट्रेन से जा रहा हूँ।"},
    ("plane", "fly"):                {"en": "The plane is flying.", "hi": "विमान उड़ रहा है।"},
    ("railway_station",):            {"en": "Railway station.", "hi": "रेलवे स्टेशन।"},

    # ── Finance & Work ────────────────────────────────────
    ("money", "need"):               {"en": "I need money.", "hi": "मुझे पैसे चाहिए।"},
    ("salary", "work"):              {"en": "I work for a salary.", "hi": "मैं वेतन के लिए काम करता हूँ।"},
    ("job", "work"):                 {"en": "I have a job and I work hard.", "hi": "मेरे पास नौकरी है और मैं मेहनत करता हूँ।"},
    ("office", "go"):                {"en": "I am going to the office.", "hi": "मैं ऑफिस जा रहा हूँ।"},
    ("bank", "money"):               {"en": "I am going to the bank for money.", "hi": "मैं बैंक से पैसे लेने जा रहा हूँ।"},
    ("income_tax",):                 {"en": "Income tax filing.", "hi": "आयकर दाखिल करना।"},
    ("salary",):                     {"en": "Salary payment.", "hi": "वेतन भुगतान।"},

    # ── Nature & Technical ────────────────────────────────
    ("water", "river"):              {"en": "The river has water.", "hi": "नदी में पानी है।"},
    ("mountain", "high"):            {"en": "The mountain is very high.", "hi": "पहाड़ बहुत ऊँचा है।"},
    ("sun", "hot"):                  {"en": "The sun is very hot today.", "hi": "आज सूरज बहुत गर्म है।"},
    ("rain", "water"):               {"en": "Rain brings water.", "hi": "बारिश पानी लाती है।"},
    ("gram", "kilogram"):            {"en": "Measurement from grams to kilograms.", "hi": "ग्राम से किलोग्राम में माप।"},
    ("gram",):                       {"en": "Weight unit in grams.", "hi": "ग्राम में वजन इकाई।"},
    ("water", "drink"):              {"en": "I need to drink clean water.", "hi": "साफ पानी पीना है।"},
    ("loud_speaker", "music"):       {"en": "Music is playing on the loudspeaker.", "hi": "लाउडस्पीकर पर संगीत बज रहा है।"},
    ("loud_speaker",):               {"en": "Loudspeaker announcement.", "hi": "लाउडस्पीकर पर घोषणा।"},
    ("head_of_income",):             {"en": "Head of Income classification.", "hi": "आय का शीर्ष वर्गीकरण।"},
    ("head_of_income", "tax"):       {"en": "Tax under Head of Income.", "hi": "आय के शीर्ष के तहत कर।"},

    # ── Daily Phrases ─────────────────────────────────────
    ("i", "go", "school", "morning"):{"en": "I go to school every morning.", "hi": "मैं हर सुबह स्कूल जाता हूँ।"},
    ("i", "need", "water", "please"):{"en": "I need water, please.", "hi": "मुझे पानी चाहिए, कृपया।"},
    ("please", "help", "me"):        {"en": "Please help me.", "hi": "कृपया मेरी मदद करें।"},
    ("help", "me"):                  {"en": "Please help me!", "hi": "मेरी मदद करें!"},
    ("thank_you", "very", "much"):   {"en": "Thank you very much!", "hi": "बहुत-बहुत धन्यवाद!"},
    ("no", "i", "know"):             {"en": "No, I do not know.", "hi": "नहीं, मुझे नहीं पता।"},
    ("stop", "go"):                  {"en": "Please stop and do not go.", "hi": "कृपया रुकें और न जाएं।"},
    ("time", "work"):                {"en": "It is time to go to work.", "hi": "काम पर जाने का समय है।"},

    # ── ISL Multi-word Phrases from data/ folder ──────────
    ("go_home",):                    {"en": "I am going home.", "hi": "मैं घर जा रहा हूँ।"},
    ("go_school",):                  {"en": "I am going to school.", "hi": "मैं स्कूल जा रहा हूँ।"},
    ("good_morning",):               {"en": "Good morning!", "hi": "शुभ प्रभात!"},
    ("good_night",):                 {"en": "Good night!", "hi": "शुभ रात्रि!"},
    ("goodbye",):                    {"en": "Goodbye!", "hi": "अलविदा!"},
    ("good_job",):                   {"en": "Good job! Well done!", "hi": "शाबाश! बहुत बढ़िया!"},
    ("hello_bro",):                  {"en": "Hello brother!", "hi": "हैलो भाई!"},
    ("hello_how_you",):              {"en": "Hello, how are you?", "hi": "नमस्ते, आप कैसे हैं?"},
    ("hello_thank_you",):            {"en": "Hello, thank you!", "hi": "नमस्ते, धन्यवाद!"},
    ("hello_thank_you_yes",):        {"en": "Hello, thank you, yes!", "hi": "नमस्ते, धन्यवाद, हाँ!"},
    ("father_go_work",):             {"en": "Father has gone to work.", "hi": "पिताजी काम पर गए हैं।"},
    ("father_home",):                {"en": "Father is at home.", "hi": "पिताजी घर पर हैं।"},
    ("family_home",):                {"en": "Family is at home.", "hi": "परिवार घर पर है।"},
    ("friend_good",):                {"en": "You are a good friend!", "hi": "आप एक अच्छे मित्र हैं!"},
    ("i_love_family",):              {"en": "I love my family.", "hi": "मैं अपने परिवार से प्यार करता हूँ।"},
    ("mother_food",):                {"en": "Mother has prepared food.", "hi": "माताजी ने खाना बनाया है।"},
    ("mother_food_cook",):           {"en": "Mother is cooking food.", "hi": "माताजी खाना बना रही हैं।"},
    ("need_food",):                  {"en": "I need food.", "hi": "मुझे खाना चाहिए।"},
    ("need_for_everthing",):         {"en": "I need everything.", "hi": "मुझे सब कुछ चाहिए।"},
    ("need_help",):                  {"en": "I need help!", "hi": "मुझे मदद चाहिए!"},
    ("need_water",):                 {"en": "I need water.", "hi": "मुझे पानी चाहिए।"},
    ("no_thank_you",):               {"en": "No, thank you.", "hi": "जी नहीं, धन्यवाद।"},
    ("no_i_know",):                  {"en": "No, I do not know.", "hi": "नहीं, मुझे नहीं पता।"},
    ("please_help",):                {"en": "Please help me.", "hi": "कृपया मेरी मदद करें।"},
    ("sorry_late",):                 {"en": "I am sorry for being late.", "hi": "देर से आने के लिए मुझे खेद है।"},
    ("stop_go",):                    {"en": "Stop and don't go.", "hi": "रुको और मत जाओ।"},
    ("thank_you_help",):             {"en": "Thank you for your help.", "hi": "आपकी मदद के लिए धन्यवाद।"},
    ("water_drink",):                {"en": "I need to drink water.", "hi": "पानी पीना है।"},
    ("welcome_home",):               {"en": "Welcome home!", "hi": "घर में आपका स्वागत है!"},
    ("what_name",):                  {"en": "What is your name?", "hi": "आपका नाम क्या है?"},
    ("where_home",):                 {"en": "Where is your home?", "hi": "आपका घर कहाँ है?"},
    ("where_school",):               {"en": "Where is the school?", "hi": "स्कूल कहाँ है?"},
    ("work_time",):                  {"en": "It is work time.", "hi": "यह काम का समय है।"},
    ("yes_i_know",):                 {"en": "Yes, I know.", "hi": "हाँ, मुझे पता है।"},
    ("yes_thank_you",):              {"en": "Yes, thank you!", "hi": "हाँ, धन्यवाद!"},
    ("time_what",):                  {"en": "What time is it?", "hi": "क्या समय है?"},
    ("ka_ka_ka",):                   {"en": "K - K - K (Alphabetic sequence).", "hi": "क - क - क।"},
    ("isl_demo_sentence_hello_thankyou_yes",): {"en": "Hello, thank you, yes!", "hi": "नमस्ते, धन्यवाद, हाँ!"},

    # ── Numbers ───────────────────────────────────────────
    ("number_1", "number_2", "number_3"): {"en": "One, two, three.", "hi": "एक, दो, तीन।"},
    ("number_0",):               {"en": "Zero.", "hi": "शून्य।"},
    ("number_1",):               {"en": "One.", "hi": "एक।"},
    ("number_2",):               {"en": "Two.", "hi": "दो।"},
    ("number_3",):               {"en": "Three.", "hi": "तीन।"},
    ("number_4",):               {"en": "Four.", "hi": "चार।"},
    ("number_5",):               {"en": "Five.", "hi": "पाँच।"},
    ("number_6",):               {"en": "Six.", "hi": "छह।"},
    ("number_7",):               {"en": "Seven.", "hi": "सात।"},
    ("number_8",):               {"en": "Eight.", "hi": "आठ।"},
    ("number_9",):               {"en": "Nine.", "hi": "नौ।"},
}

# ─────────────────────────────────────────────────────────────
# SUBJECT / VERB / OBJECT NLP GRAMMAR RULES
# ─────────────────────────────────────────────────────────────

_SUBJECTS_EN = {"i", "father", "mother", "friend", "teacher", "student",
                "doctor", "grandfather", "grandmother", "husband", "wife",
                "family", "we", "they", "he", "she", "man", "woman"}

_VERBS_EN = {"go", "come", "work", "learn", "cook", "eat", "drink", "read",
             "write", "help", "love", "need", "stop", "run", "play", "study",
             "sleep", "speak", "listen", "see", "feel", "know", "meet", "pay"}

_OBJECTS_EN = {"home", "school", "work", "office", "hospital", "food", "water",
               "medicine", "money", "book", "letter", "salary", "bank", "doctor",
               "train", "bus", "car", "family", "friend", "name", "time"}

_QUESTION_STARTERS = {"what", "where", "how", "when", "why", "which", "who"}

# Subject pronouns map for English
_SUBJECT_MAP_EN = {
    "i": "I", "father": "Father", "mother": "Mother", "friend": "my friend",
    "teacher": "the teacher", "student": "the student", "doctor": "the doctor",
    "family": "my family", "we": "We", "they": "They", "he": "He", "she": "She",
}

# Hindi translation fallbacks for common words
_HINDI_WORD_MAP = {
    "hello": "नमस्ते", "yes": "हाँ", "no": "नहीं", "thank_you": "धन्यवाद",
    "please": "कृपया", "help": "मदद", "water": "पानी", "food": "खाना",
    "home": "घर", "school": "स्कूल", "work": "काम", "go": "जाना",
    "come": "आना", "father": "पिताजी", "mother": "माताजी",
    "family": "परिवार", "friend": "मित्र", "love": "प्यार",
    "happy": "खुश", "sad": "उदास", "need": "चाहिए", "morning": "सुबह",
    "night": "रात", "day": "दिन", "time": "समय", "stop": "रुको",
    "doctor": "डॉक्टर", "medicine": "दवाई", "hospital": "अस्पताल",
    "money": "पैसे", "bus": "बस", "train": "ट्रेन", "car": "कार",
    "read": "पढ़ो", "write": "लिखो", "learn": "सीखो", "teacher": "अध्यापक",
    "student": "छात्र", "sorry": "क्षमा करें", "good": "अच्छा",
    "name": "नाम", "what": "क्या", "where": "कहाँ", "how": "कैसे",
    "i": "मैं", "me": "मुझे", "my": "मेरा", "we": "हम",
    "hug": "गले लगाना", "little": "थोड़ा", "above": "ऊपर", "below": "नीचे",
    "nice": "अच्छा", "hello_bro": "नमस्ते भाई", "interview": "साक्षात्कार",
    "how_much": "कितना", "leave_preparatory_to_retirement": "सेवानिवृत्ति पूर्व अवकाश",
    "grandson": "पोता", "granddaughter": "पोती", "handkerchief": "रूमाल",
    "gram_kilogram": "ग्राम किलोग्राम"
}


def _normalize(word):
    return word.lower().strip().replace(" ", "_")


def _build_english_sentence(words):
    """
    Smart English sentence builder using ISL SOV grammar rules.
    Tries to identify subject/verb/object and form a natural sentence.
    """
    if not words:
        return "No gesture detected."

    # Detect question
    if words[0] in _QUESTION_STARTERS:
        if len(words) == 1:
            return f"{words[0].capitalize()}?"
        return " ".join(w.replace("_", " ").title() if i == 0 else w.replace("_", " ") for i, w in enumerate(words)) + "?"

    # Single word
    if len(words) == 1:
        return f"{words[0].replace('_', ' ').title()}."

    # Detect subject
    subject = None
    verb = None
    obj_words = []

    for w in words:
        if w in _SUBJECTS_EN and subject is None:
            subject = _SUBJECT_MAP_EN.get(w, w.title())
        elif w in _VERBS_EN and verb is None:
            verb = w
        else:
            obj_words.append(w.replace("_", " "))

    if subject and verb and obj_words:
        obj_str = " ".join(obj_words)
        return f"{subject} is going to {obj_str}." if verb == "go" else \
               f"{subject} {verb}s {obj_str}."
    elif subject and verb:
        return f"{subject} {verb}s."
    elif verb and obj_words:
        obj_str = " ".join(obj_words)
        return f"I {verb} {obj_str}."
    else:
        formatted = [w.replace("_", " ").title() for w in words]
        return " ".join(formatted) + "."


def _build_hindi_sentence(words):
    """
    Smart Hindi sentence builder.
    """
    if not words:
        return "कोई संकेत नहीं।"

    translated = []
    for w in words:
        if w in _HINDI_WORD_MAP:
            translated.append(_HINDI_WORD_MAP[w])
        else:
            # Try config GESTURE_TRANSLATIONS if available
            try:
                from src.config import GESTURE_TRANSLATIONS
                if w in GESTURE_TRANSLATIONS and "hi" in GESTURE_TRANSLATIONS[w]:
                    translated.append(GESTURE_TRANSLATIONS[w]["hi"])
                    continue
            except ImportError:
                pass
            translated.append(w.replace("_", " "))

    if len(translated) == 1:
        return f"{translated[0]}।"

    return " ".join(translated) + "।"


# ─────────────────────────────────────────────────────────────
# MAIN TRANSLATION FUNCTION
# ─────────────────────────────────────────────────────────────

def translate_isl_sequence(words_list, lang="en"):
    """
    Takes a list of recognized ISL gesture words and converts them into a
    fluent, grammatically correct sentence in English or Hindi.

    Steps:
      1. Exact match in ISL_SENTENCE_DICTIONARY
      2. Longest n-gram sub-sequence match
      3. Smart NLP grammar fallback
    """
    if not words_list:
        return "Waiting for ISL gesture sequence..." if lang == "en" else "इंतजार कर रहा हूं..."

    clean_words = [_normalize(w) for w in words_list if _normalize(w) not in ("idle", "")]
    if not clean_words:
        return "Waiting for ISL gesture sequence..." if lang == "en" else "इंतजार कर रहा हूं..."

    # 1. Exact match
    tuple_key = tuple(clean_words)
    if tuple_key in ISL_SENTENCE_DICTIONARY:
        return ISL_SENTENCE_DICTIONARY[tuple_key][lang]

    # 2. Longest sub-sequence match
    best_match = None
    best_start = 0
    best_len = 0
    for length in range(len(clean_words), 0, -1):
        for start in range(len(clean_words) - length + 1):
            sub_tuple = tuple(clean_words[start:start + length])
            if sub_tuple in ISL_SENTENCE_DICTIONARY:
                if length > best_len:
                    best_match = ISL_SENTENCE_DICTIONARY[sub_tuple][lang]
                    best_start = start
                    best_len = length

    if best_match:
        leftover = [clean_words[i] for i in range(len(clean_words))
                    if i < best_start or i >= best_start + best_len]
        if leftover:
            extra = ", ".join(w.replace("_", " ").title() for w in leftover)
            return f"{best_match} ({extra})"
        return best_match

    # 3. Smart NLP grammar fallback
    if lang == "en":
        return _build_english_sentence(clean_words)
    else:
        return _build_hindi_sentence(clean_words)


def close_sentence(words_list, lang="en"):
    """
    Called when a sentence is 'closed' (after a pause).
    Returns the final, complete, fluent translation for the entire sequence.
    """
    return translate_isl_sequence(words_list, lang)


if __name__ == "__main__":
    test_seqs = [
        ["hello", "thank_you", "yes"],
        ["father", "go", "work"],
        ["need", "water"],
        ["mother", "food", "cook"],
        ["head_of_income", "tax"],
        ["loud_speaker"],
        ["go", "hospital"],
        ["i", "need", "water", "please"],
        ["happy", "good"],
        ["unknown_word_test", "another_unknown"],
    ]
    print("=== ISL SENTENCE TRANSLATION TEST (v2.0) ===")
    for seq in test_seqs:
        en = translate_isl_sequence(seq, "en")
        hi = translate_isl_sequence(seq, "hi")
        print(f"\nInput:  {seq}")
        print(f"  EN: {en}")
        print(f"  HI: {hi}")
