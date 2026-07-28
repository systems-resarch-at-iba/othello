import type { GameState, Player } from '../lib/types'

interface GameStatusBarProps {
  state: GameState
  humanPlayer: Player
  aiThinking: boolean
  error: string | null
}

function ScoreBadge({ color, count, label }: { color: string; count: number; label: string }) {
  return (
    <div
      className="flex items-center gap-2 rounded-full border border-hairline-strong bg-paper-raised px-3 py-1.5 font-sans text-sm"
      aria-label={`${label}: ${count}`}
    >
      <span
        className="h-3 w-3 shrink-0 rounded-full border border-hairline-strong"
        style={{ background: color }}
        aria-hidden="true"
      />
      <span className="font-semibold text-ink">{count}</span>
    </div>
  )
}

export function GameStatusBar({ state, humanPlayer, aiThinking, error }: GameStatusBarProps) {
  const blackCount = state.board.filter((c) => c === -1).length
  const whiteCount = state.board.filter((c) => c === 1).length

  let status: string
  if (error) {
    status = `Error: ${error}`
  } else if (state.gameOver) {
    if (blackCount === whiteCount) status = "It's a draw."
    else status = `${blackCount > whiteCount ? 'Black' : 'White'} wins.`
  } else if (aiThinking) {
    status = 'AI is thinking...'
  } else if (state.currentPlayer === humanPlayer) {
    status = 'Your move.'
  } else {
    status = "AI's move."
  }

  return (
    <div className="flex items-center justify-between gap-4">
      <ScoreBadge color="#15161b" count={blackCount} label="Black" />
      <span className="font-sans text-sm font-medium text-ink">{status}</span>
      <ScoreBadge color="#f5f5f4" count={whiteCount} label="White" />
    </div>
  )
}
