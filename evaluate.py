import joblib
from utils import preprocess

def predict(message: str) -> dict:
    vectorizer = joblib.load("models/vectorizer.pkl")
    model      = joblib.load("models/model.pkl")

    vec   = vectorizer.transform([message])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    return {
        "message": message,
        "prediction": "spam" if label == 1 else "ham",
        "confidence": round(float(max(proba)), 4),
    }

if __name__ == "__main__":
    tests = [
        "Win a FREE iPhone now! Click here!!!",
        "Hey, are we still on for lunch tomorrow?",
        "URGENT: Your account has been compromised. Verify now.",
        "Can you send me the report before the meeting?",
    ]
    for msg in tests:
        result = predict(msg)
        tag = "SPAM" if result["prediction"] == "spam" else "  ok"
        print(f"[{tag}] ({result['confidence']:.0%}) {result['message'][:60]}")