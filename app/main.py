from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib
import os

# ── Schemas ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, example="Win a free iPhone now!")

class PredictResponse(BaseModel):
    message: str
    prediction: str          # "spam" | "ham"
    confidence: float        # probabilidad del label ganador
    spam_probability: float  # siempre la prob de spam, útil para umbrales

# ── Estado global (cargado una sola vez al iniciar) ───────────────────────────

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    model_path      = os.getenv("MODEL_PATH", "models/model.pkl")
    vectorizer_path = os.getenv("VECTORIZER_PATH", "models/vectorizer.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise RuntimeError(f"Vectorizer no encontrado: {vectorizer_path}")

    ml_models["model"]      = joblib.load(model_path)
    ml_models["vectorizer"] = joblib.load(vectorizer_path)
    print("Modelos cargados correctamente")
    yield
    # --- shutdown ---
    ml_models.clear()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Spam Classifier API",
    description="Detecta spam en mensajes de texto",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Verifica que la API y los modelos están listos."""
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Clasifica un mensaje como spam o ham."""
    try:
        vectorizer = ml_models["vectorizer"]
        model      = ml_models["model"]

        vec   = vectorizer.transform([req.message])
        label = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0]   # [prob_ham, prob_spam]

        spam_prob  = float(proba[1])
        confidence = float(max(proba))

        return PredictResponse(
            message          = req.message,
            prediction       = "spam" if label == 1 else "ham",
            confidence       = round(confidence, 4),
            spam_probability = round(spam_prob, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(messages: list[PredictRequest]):
    """Clasifica varios mensajes en una sola llamada (máx 100)."""
    if len(messages) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 mensajes por batch")

    vectorizer = ml_models["vectorizer"]
    model      = ml_models["model"]

    texts  = [r.message for r in messages]
    vecs   = vectorizer.transform(texts)
    labels = model.predict(vecs)
    probas = model.predict_proba(vecs)

    results = []
    for msg, label, proba in zip(texts, labels, probas):
        results.append(PredictResponse(
            message          = msg,
            prediction       = "spam" if label == 1 else "ham",
            confidence       = round(float(max(proba)), 4),
            spam_probability = round(float(proba[1]), 4),
        ))
    return results