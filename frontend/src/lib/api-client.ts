import type { Board, EngineSettings, Player } from './types'

export class OthelloApiError extends Error {}

/**
 * `fetch` itself throwing (backend unreachable -- offline, wrong URL, CORS,
 * the process just isn't running) surfaces as a raw browser exception like
 * "NetworkError when attempting to fetch resource" or "Failed to fetch":
 * technically accurate, meaningless to a player. OthelloApiError means the
 * backend *did* respond, just with an error, so its message is kept as-is.
 */
export function friendlyErrorMessage(err: unknown): string {
  if (err instanceof OthelloApiError) return err.message
  return "Couldn't reach the model."
}

function engineParams(settings: EngineSettings) {
  return {
    num_mcts_sims: settings.numMctsSims,
    cpuct_type: settings.cpuctType,
    c_puct: settings.cPuct,
  }
}

export interface AiMove {
  move: number
  elapsedMs: number
}

/**
 * Asks the backend for the AI's move. Callers should only invoke this when
 * it's actually the AI's turn and the client-side rules engine has already
 * confirmed a legal move exists (see hasValidMoves): auto-passing locally
 * is free, since that check is already needed for turn/pass logic anyway,
 * and it avoids a network round-trip for a move that can't happen.
 */
export async function requestAiMove(
  apiBaseUrl: string,
  board: Board,
  player: Player,
  settings: EngineSettings
): Promise<AiMove> {
  const response = await fetch(`${apiBaseUrl}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ board, player, ...engineParams(settings) }),
  })

  if (!response.ok) {
    throw new OthelloApiError(`Othello backend returned ${response.status}`)
  }

  const data = (await response.json()) as { move: number; elapsed_ms: number }
  return { move: data.move, elapsedMs: data.elapsed_ms }
}

export interface MoveHints {
  /** MCTS visit-count distribution over all 65 actions, indices 0-63 matching board squares. */
  probs: number[]
  elapsedMs: number
}

/**
 * Runs a fresh MCTS search from the given player's perspective and returns
 * the resulting visit-count distribution, so the UI can show "which moves
 * is the engine actually considering" rather than just its single best pick.
 * Not tied to whose turn it actually is server-side: the caller decides
 * when this is worth asking for (see useMoveHints).
 */
export async function requestHints(
  apiBaseUrl: string,
  board: Board,
  player: Player,
  settings: EngineSettings
): Promise<MoveHints> {
  const response = await fetch(`${apiBaseUrl}/hints`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ board, player, ...engineParams(settings) }),
  })

  if (!response.ok) {
    throw new OthelloApiError(`Othello backend returned ${response.status}`)
  }

  const data = (await response.json()) as { probs: number[]; elapsed_ms: number }
  return { probs: data.probs, elapsedMs: data.elapsed_ms }
}

export interface ModelHealth {
  status: string
  modelVersion: string
  architecture: string
  parameterCount: number
}

export async function checkHealth(apiBaseUrl: string): Promise<ModelHealth> {
  const response = await fetch(`${apiBaseUrl}/health`)
  if (!response.ok) throw new OthelloApiError(`Othello backend returned ${response.status}`)
  const data = (await response.json()) as {
    status: string
    model_version: string
    architecture: string
    parameter_count: number
  }
  return {
    status: data.status,
    modelVersion: data.model_version,
    architecture: data.architecture,
    parameterCount: data.parameter_count,
  }
}
