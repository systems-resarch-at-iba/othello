# Othello

An Othello (Reversi) engine (Monte Carlo Tree Search + a CNN, AlphaZero-style)
with a web frontend, built by Systems Research @ IBA. Played live at
[systems-research @ IBA's playground](https://github.com/systems-resarch-at-iba/site).

## Structure

- `frontend/`: the web UI (`@othello/frontend`), mounted as a submodule and
  npm workspace package inside [`site`](https://github.com/systems-resarch-at-iba/site)
  at `/playground/othello`.
- `backend/`: a FastAPI server wrapping the MCTS+CNN engine, deployed to
  Google Cloud Run. See `backend/README.md`.

## Provenance

The board representation, MCTS, and CNN architecture originated in
[`othello-engine`](https://github.com/syedtaha22/othello-engine), including
the training loop the deployed model was produced by. This repo is a fresh
start built around them for the web: no git history was carried over, and
the gameplay GUI/training/benchmarking tooling from that repo isn't part of
this one, which only needs to serve moves.

## License

MIT, see [LICENSE](LICENSE).
