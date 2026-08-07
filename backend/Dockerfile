FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models

WORKDIR /srv

# Dependencies first so a code change does not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. ~130 MB, and it means the container
# starts without reaching out to HuggingFace — which also makes an air-gapped
# or rate-limited deploy work.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY app ./app
COPY content ./content

# Run unprivileged; /srv/var holds the database and the vector index.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /srv/var \
    && chown -R appuser:appuser /srv /opt/models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
