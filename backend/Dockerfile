# The C++ engine is built in its own stage so that the compiler and the
# build tools stay out of the final image. Both stages use the same base
# image, so the extension matches the interpreter and the C library that run it.
FROM python:3.12-slim AS extension

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pybind11==3.1.0

WORKDIR /src
COPY cpp ./cpp
RUN cmake -S cpp -B cpp/build -DPython3_EXECUTABLE=$(which python) \
    && cmake --build cpp/build --target othello_cpp -j $(nproc) \
    && PYTHONPATH=cpp/build python -c "import othello_cpp"


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
# engine/cpp_ai.py looks for the module in cpp/build. Without it, main.py
# serves requests with the Python engine.
COPY --from=extension /src/cpp/build/othello_cpp*.so cpp/build/

EXPOSE 8080
# Cloud Run injects $PORT (8080 by default) and expects the container to
# listen on it; shell form (not exec-form CMD) so that expansion happens.
# The :-8080 fallback keeps `docker run` usable locally without setting it.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
