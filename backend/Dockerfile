FROM python:3.12-slim

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# HuggingFace caches into HF_HOME. Point it at the persistent volume so model
# weights survive container restarts.
ENV HF_HOME=/app/models/hf
ENV TRANSFORMERS_CACHE=/app/models/hf
ENV HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
