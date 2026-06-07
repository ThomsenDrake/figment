FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FIGMENT_MODE=hosted \
    MODEL_STACK=omni_native \
    AUDIO_BACKEND=none \
    ENABLE_AUDIO_INTAKE=false \
    ALLOW_LOCAL_ASR=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
