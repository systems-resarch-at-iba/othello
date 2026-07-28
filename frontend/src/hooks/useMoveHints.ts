import { useEffect, useState } from 'react'
import { requestHints } from '../lib/api-client'
import type { Board, EngineSettings, Player } from '../lib/types'

/**
 * Fetches the engine's move-suggestion distribution for the given board
 * whenever `enabled` is true, and clears it otherwise (e.g. hints toggled
 * off, or it's no longer the player's turn to decide). Returns null while
 * disabled or between requests, so callers never render a stale probability
 * from a previous, now-irrelevant board state.
 */
export function useMoveHints(
  apiBaseUrl: string,
  board: Board,
  player: Player,
  enabled: boolean,
  settings: EngineSettings
): number[] | null {
  const [probs, setProbs] = useState<number[] | null>(null)

  useEffect(() => {
    if (!enabled) {
      setProbs(null)
      return
    }

    let cancelled = false
    setProbs(null)

    requestHints(apiBaseUrl, board, player, settings)
      .then((result) => {
        if (!cancelled) setProbs(result.probs)
      })
      .catch(() => {
        // Hints are a visual aid, not core gameplay: a failed request just
        // means no highlight shows, not a surfaced game error.
      })

    return () => {
      cancelled = true
    }
  }, [apiBaseUrl, board, player, enabled, settings])

  return probs
}
