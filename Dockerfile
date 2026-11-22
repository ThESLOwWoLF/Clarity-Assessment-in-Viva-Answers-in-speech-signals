# CPU image. The trained model is NOT baked in (it contains embeddings of private clips): mount it.
#   docker build -t clarity .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" clarity
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN grep -vE "^(torch|torchaudio)==" requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt
# fetch WavLM at build time so the container starts offline
RUN python -c "from transformers import AutoFeatureExtractor, WavLMModel as M; n='microsoft/wavlm-base-plus'; AutoFeatureExtractor.from_pretrained(n); M.from_pretrained(n)"
COPY config.yaml .
COPY src src
ENV DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
