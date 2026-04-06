# ── Stage 1: dependencias ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# instala dependencias en un directorio aislado
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: imagen final ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# copia solo las dependencias ya instaladas del stage anterior
COPY --from=builder /install /usr/local

# copia el código y los artefactos del modelo
COPY app/        ./app/
COPY models/     ./models/
COPY utils.py    .

# usuario no-root por seguridad
RUN useradd -m appuser
USER appuser

# variables de entorno
ENV MODEL_PATH=models/model.pkl
ENV VECTORIZER_PATH=models/vectorizer.pkl
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]