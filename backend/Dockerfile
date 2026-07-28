FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# One install, one index order: requirements.txt pins an exact torch
# version, so a separate `pip install torch` step before this one could
# resolve a different version there's no guarantee this second command can
# actually find again (the default index doesn't have CPU-only wheels).
# CPU-only wheel explicitly: the default PyPI torch build pulls in CUDA
# support (multi-GB), wasted space on a CPU-only Cloud Run instance.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY . .

EXPOSE 8080
# Cloud Run injects $PORT (8080 by default) and expects the container to
# listen on it; shell form (not exec-form CMD) so that expansion happens.
# The :-8080 fallback keeps `docker run` usable locally without setting it.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
