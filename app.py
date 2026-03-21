from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import json
import re
import os

app = Flask(__name__)

# -------------------------------------------------------
# LOAD VOCABULARY
# -------------------------------------------------------
print("Loading vocabulary...")
with open("vocabulary.json", "r") as f:
    vocab = json.load(f)
print(f"✅ Vocabulary loaded — {len(vocab):,} words")

# -------------------------------------------------------
# LOAD TFLITE MODEL
# -------------------------------------------------------
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="spam_model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ TFLite model loaded")
print(f"   Input  : {input_details[0]['shape']} {input_details[0]['dtype']}")
print(f"   Output : {output_details[0]['shape']} {output_details[0]['dtype']}")

MAX_LEN = 300

# -------------------------------------------------------
# VECTORIZE TEXT USING SAVED VOCAB
# Replicates what TextVectorization did during training
# -------------------------------------------------------
# def vectorize_text(text, max_len=MAX_LEN):
#     # Basic cleaning — lowercase, split on whitespace
#     tokens = text.lower().split()
#     ids = []
#     for token in tokens:
#         # 0 = padding, 1 = [UNK], rest = vocab words
#         idx = vocab.get(token, 1)
#         ids.append(idx)
#     # Truncate if too long
#     ids = ids[:max_len]
#     # Pad if too short
#     ids = ids + [0] * (max_len - len(ids))
#     return np.array(ids, dtype=np.int32)

# -------------------------------------------------------
# TFLITE INFERENCE
# -------------------------------------------------------
def predict_email(email_text):
    # Input is raw string — model handles vectorization internally
    x = np.array([email_text], dtype=object)  # shape (1,) string array

    # Resize input tensor to match
    interpreter.resize_tensor_input(input_details[0]['index'], [1])
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    return prob

# -------------------------------------------------------
# SPAM KEYWORDS
# -------------------------------------------------------
SPAM_KEYWORDS = [
    "act now", "limited time offer", "today only", "expires soon",
    "urgent action required", "immediate action",
    "you won", "you have won", "claim your prize", "free money",
    "free gift", "cash prize", "lottery winner", "selected winner",
    "million dollars", "billion dollars", "wire transfer",
    "click here", "click now", "visit now", "buy now", "order now",
    "call now", "call free",
    "no prescription", "weight loss", "lose weight fast",
    "cheap medications", "online pharmacy",
    "online casino", "free casino", "poker online", "win big",
    "verify your account", "confirm your account", "account suspended",
    "account compromised", "unusual activity detected",
    "update your billing", "update your password",
    "your payment failed", "billing information",
    "dear friend", "dear winner", "dear beneficiary",
    "undisclosed recipients", "this is not spam",
    "remove from list", "to be removed",
    "congratulations you", "you have been selected",
    "you have been chosen",
    "nigerian prince", "inheritance funds", "bank transfer",
    "investment opportunity", "risk free investment",
    "guaranteed returns", "double your money",
    "no credit check", "debt relief", "loan approved",
    "viagra", "cialis", "enlarge", "adult content",
    "xxx", "meet singles", "hot singles"
]

def find_spam_keywords(text):
    text_lower = text.lower()
    found = []
    for keyword in SPAM_KEYWORDS:
        if " " in keyword:
            if keyword.lower() in text_lower:
                found.append(keyword)
        else:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found.append(keyword)
    return found

def highlight_keywords(text, keywords):
    highlighted = text
    for keyword in keywords:
        if " " in keyword:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        else:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        highlighted = pattern.sub(
            f'<mark class="highlight">{keyword}</mark>',
            highlighted
        )
    return highlighted

def get_risk_level(probability):
    if probability >= 0.85:
        return {"level": "HIGH",   "color": "#e74c3c", "emoji": "🔴"}
    elif probability >= 0.5:
        return {"level": "MEDIUM", "color": "#e67e22", "emoji": "🟡"}
    elif probability >= 0.25:
        return {"level": "LOW",    "color": "#f1c40f", "emoji": "🟡"}
    else:
        return {"level": "SAFE",   "color": "#2ecc71", "emoji": "🟢"}

# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json()
        email_text = data.get("email", "").strip()

        if not email_text or len(email_text) < 5:
            return jsonify({"error": "Email too short"}), 400

        # Fast TFLite inference
        probability      = predict_email(email_text)
        is_spam          = probability >= 0.5
        found_keywords   = find_spam_keywords(email_text)
        highlighted_text = highlight_keywords(email_text, found_keywords)
        risk             = get_risk_level(probability)

        return jsonify({
            "probability"      : round(probability * 100, 2),
            "is_spam"          : is_spam,
            "label"            : "SPAM" if is_spam else "SAFE",
            "risk_level"       : risk["level"],
            "risk_color"       : risk["color"],
            "risk_emoji"       : risk["emoji"],
            "found_keywords"   : found_keywords,
            "keyword_count"    : len(found_keywords),
            "highlighted_text" : highlighted_text,
            "email_length"     : len(email_text.split()),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "TFLite SpamShield"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
