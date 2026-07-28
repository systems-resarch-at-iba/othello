import { useCallback, useEffect, useState } from 'react'
import { advanceTurn, createInitialState, getValidMoves, hasValidMoves } from '../lib/othello-rules'
import { friendlyErrorMessage, requestAiMove } from '../lib/api-client'
import type { EngineSettings, GameState, Move, Player } from '../lib/types'

export function useOthelloGame(
  apiBaseUrl: string,
  humanPlayer: Player,
  settings: EngineSettings,
  onLatency: (elapsedMs: number) => void
) {
  const [state, setState] = useState<GameState>(createInitialState)
  const [error, setError] = useState<string | null>(null)

  const aiPlayer = -humanPlayer as Player
  // Derived rather than its own state: true exactly while it's the AI's
  // unresolved turn. Once a move succeeds, advanceTurn flips currentPlayer
  // away from aiPlayer, which naturally turns this false too, so there's no
  // separate "done loading" signal to set on the success path. A failed
  // request (error !== null) also counts as no-longer-thinking, since the
  // request is not in flight anymore, even though currentPlayer hasn't moved.
  const aiThinking = !state.gameOver && state.currentPlayer === aiPlayer && error === null

  useEffect(() => {
    if (state.gameOver || state.currentPlayer !== aiPlayer) return
    // advanceTurn already auto-passes a player with no legal move, so by the
    // time currentPlayer === aiPlayer here, the AI is guaranteed to have at
    // least one legal move (except possibly the very first state, before any
    // move has been played), this check is cheap insurance against that
    // edge case rather than load-bearing logic.
    if (!hasValidMoves(state.board, aiPlayer)) return

    let cancelled = false

    requestAiMove(apiBaseUrl, state.board, aiPlayer, settings)
      .then(({ move: moveIndex, elapsedMs }) => {
        if (cancelled) return
        onLatency(elapsedMs)
        if (moveIndex === 64) return
        const move: Move = { row: Math.floor(moveIndex / 8), col: moveIndex % 8 }
        setState((prev) => advanceTurn(prev, move))
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(friendlyErrorMessage(err))
      })

    return () => {
      cancelled = true
    }
    // Changing settings mid-thought (e.g. cranking sims up on the slider
    // while the AI is already searching) cancels the in-flight request and
    // re-fires with the new settings, rather than finishing the stale one.
    // onLatency is expected to be stable (useCallback'd by the caller); it's
    // an event notification, not a value this effect reads to decide what to do.
  }, [apiBaseUrl, aiPlayer, state, settings, onLatency])

  const playHumanMove = useCallback(
    (move: Move) => {
      setState((prev) => {
        if (prev.gameOver || prev.currentPlayer !== humanPlayer) return prev
        const legal = getValidMoves(prev.board, humanPlayer)
        const isLegal = legal.some((m) => m.row === move.row && m.col === move.col)
        return isLegal ? advanceTurn(prev, move) : prev
      })
    },
    [humanPlayer]
  )

  const restart = useCallback(() => {
    setState(createInitialState())
    setError(null)
  }, [])

  return { state, humanPlayer, aiPlayer, aiThinking, error, playHumanMove, restart }
}
