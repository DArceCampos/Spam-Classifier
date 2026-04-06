import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from utils import preprocess

# ── 1. Cargar datos ──────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    df["label_bin"] = (df["label"] == "spam").astype(int)  # spam=1, ham=0
    print(f"Dataset cargado: {len(df)} filas")
    print(df["label"].value_counts())
    return df

# ── 3. Vectorizar ────────────────────────────────────────────────────────────
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=preprocess,
        ngram_range=(1, 2),   # unigramas + bigramas
        max_features=10_000,
        sublinear_tf=True,    # log(tf) en lugar de tf crudo
    )

# ── 4. Entrenar ──────────────────────────────────────────────────────────────
def train(X_train, y_train, model_type: str = "naive_bayes"):
    if model_type == "naive_bayes":
        model = MultinomialNB(alpha=0.1)
    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")

    model.fit(X_train, y_train)
    print(f"Modelo entrenado: {model_type}")
    return model

# ── 5. Evaluar ───────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\n── Reporte de clasificación ──")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("── Matriz de confusión ──")
    print(confusion_matrix(y_test, y_pred))

# ── 6. Guardar artefactos ────────────────────────────────────────────────────
def save_artifacts(model, vectorizer, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model,      f"{model_dir}/model.pkl")
    joblib.dump(vectorizer, f"{model_dir}/vectorizer.pkl")
    print(f"\nArtefactos guardados en {model_dir}/")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Carga
    df = load_data("data/spam.csv")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_bin"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_bin"],   # mantiene proporción spam/ham en ambos sets
    )

    # Vectoriza — fit SOLO sobre train, transform en ambos
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Entrena
    model = train(X_train_vec, y_train, model_type="naive_bayes")

    # Evalúa
    evaluate(model, X_test_vec, y_test)

    # Guarda
    save_artifacts(model, vectorizer)