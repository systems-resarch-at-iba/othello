# Othello backend

FastAPI server wrapping the MCTS+CNN engine, deployed to Google Cloud Run.

## Endpoints

| Method & path | Request body | Response body |
|---|---|---|
| `GET /health` |  | `{ "status": "ok" \| "model not loaded", "model_version": string, "architecture": string, "parameter_count": number }` |
| `POST /move`  | `{ "board": number[64], "player": 1 \| -1, "num_mcts_sims"?: number, "cpuct_type"?: "static" \| "increment" \| "decrement", "c_puct"?: number }` | `{ "move": number, "elapsed_ms": number }` |
| `POST /hints` | same as `/move` | `{ "probs": number[65], "elapsed_ms": number }` |

`board` is row-major flat, index = `row*8 + col`. `move` is 0-63 for a board
square, or 64 for pass. `probs` is the MCTS visit-count distribution over
all 65 actions for the given position, same indexing as `move`.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt
uvicorn main:app --reload
```

Place the trained checkpoint at `models/model-{version}.pth.tar`, where
`{version}` is `arch.py`'s own `args.version`. Without it, the server still
starts, but `/health` reports `"model not loaded"` and `/move`/`/hints`
return 503 until it's there.

## Deployment

Deployed via `cloudbuild.yaml` to Cloud Run (project `srai-othello`, region
`asia-south1`, service `othello-backend`). The checkpoint is never committed
to this repo (see `.gitignore`); it's uploaded to a GCS bucket and pulled
into the build context by `cloudbuild.yaml`'s first step, so it ends up
baked into the image rather than fetched at request time.

```bash
gcloud builds submit --config=backend/cloudbuild.yaml .   # from the repo root
```

## Tests

```bash
pip install pytest httpx
pytest
```

Endpoint tests run against a randomly-initialized network rather than the
real checkpoint: they're checking the request/response contract, not move
quality, so the real weights aren't needed and CI runs the exact same tests
as a local checkout with the checkpoint installed.
