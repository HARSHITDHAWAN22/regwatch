FROM python:3.11-slim

WORKDIR /app

# build-essential: needed to build some deps' native extensions
# libgomp1: required at runtime by faiss-cpu on debian-slim (missing it causes
#           an import-time crash that's easy to miss until you actually run
#           the container, not just build it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads data/faiss_index

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
