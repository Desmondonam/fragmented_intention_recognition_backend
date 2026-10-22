FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image ~1.5GB smaller than the default CUDA wheel
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY model/ model/
COPY app/ app/

# Weights are downloaded from the Hugging Face Hub on first startup and cached
# here; mount a volume at this path to keep them across restarts.
ENV HF_HOME=/app/.cache/huggingface \
    MODEL_REPO_ID=Desmondonam/intent-emotion-mtl-distilroberta \
    MODEL_REVISION=main

EXPOSE 8000

# Allow PORT env var override (Render / Railway / HF Spaces inject this)
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
