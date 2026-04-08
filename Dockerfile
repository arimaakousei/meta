FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py          ./models.py
COPY server/            ./server/
COPY __init__.py        ./__init__.py

# Hugging Face Spaces uses port 7860
ENV PORT=7860
ENV EMAIL_TRIAGE_TASK=basic_triage

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
