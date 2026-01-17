FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY backend /app/backend
COPY bots /app/bots
COPY neural /app/neural
COPY models /app/models

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

WORKDIR /app/backend
CMD ["python", "-c", "from api_server import run; run('0.0.0.0', 8000)"]
