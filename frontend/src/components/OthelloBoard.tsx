'use client'

import { useCallback, useState } from 'react'
import { useOthelloGame } from '../hooks/useOthelloGame'
import { useMoveHints } from '../hooks/useMoveHints'
import { getValidMoves } from '../lib/othello-rules'
import { DEFAULT_ENGINE_SETTINGS } from '../lib/types'
import type { EngineSettings, Player } from '../lib/types'
import { Cell } from './Cell'
import { GameStatusBar } from './GameStatusBar'
import { GameSettings } from './GameSettings'
import { TimingChart } from './TimingChart'
import { CpuctChart } from './CpuctChart'
import { ModelCard } from './ModelCard'

interface OthelloBoardProps {
  apiBaseUrl: string
}

export function OthelloBoard({ apiBaseUrl }: OthelloBoardProps) {
  const [humanPlayer, setHumanPlayer] = useState<Player>(-1)
  const [settings, setSettings] = useState<EngineSettings>(DEFAULT_ENGINE_SETTINGS)
  const [showMoveHints, setShowMoveHints] = useState(true)
  const [showAiHints, setShowAiHints] = useState(true)
  const [latencyHistory, setLatencyHistory] = useState<number[]>([])

  const handleHumanPlayerChange = (player: Player) => {
    setHumanPlayer(player)
    setLatencyHistory([])
  }

  // Stable across renders (empty dep array, setState identity never
  // changes), which matters because it's a dependency of useOthelloGame's
  // AI-move effect -- an unstable reference here would re-fire that effect,
  // and therefore the network request, on every render.
  const recordLatency = useCallback((ms: number) => {
    setLatencyHistory((prev) => [...prev, ms])
  }, [])

  const settingsPanel = (
    <GameSettings
      humanPlayer={humanPlayer}
      onHumanPlayerChange={handleHumanPlayerChange}
      settings={settings}
      onSettingsChange={setSettings}
      showMoveHints={showMoveHints}
      onShowMoveHintsChange={setShowMoveHints}
      showAiHints={showAiHints}
      onShowAiHintsChange={setShowAiHints}
    />
  )

  return (
    <>
      {/* Three-column grid, the middle track fixed at the board's own
          width and the two flanking tracks both `1fr`: since equal-`fr`
          tracks are always equal width regardless of what's in them, the
          middle column -- the board -- is exactly centered on its own,
          the same as if the sidebar didn't exist. The sidebar just
          occupies the right track's leftover space; `items-center`
          vertically aligns it against the board's actual rendered height,
          so it stays centered automatically as either side's content
          changes, with no manual offsets to re-tune. */}
      <div className="mx-auto grid max-w-[1800px] grid-cols-1 items-center gap-10 px-6 pb-12 lg:grid-cols-[1fr_480px_1fr] lg:gap-20">
        {/* About the model (not configurable) on the left, mirroring
            Settings (which is) on the right -- kept as separate panels
            rather than stacked together so the two aren't visually
            conflated. */}
        <aside className="w-full lg:col-start-1 lg:w-72 lg:justify-self-end">
          <ModelCard apiBaseUrl={apiBaseUrl} />
        </aside>

        <div className="mx-auto w-full lg:col-start-2" style={{ maxWidth: 480 }}>
          {/* Keyed by humanPlayer: switching sides mid-game would otherwise leave
              stale, inconsistent turn state (whose turn is it now?), so remount
              the whole game fresh instead of trying to patch it in place. */}
          <Game
            key={humanPlayer}
            apiBaseUrl={apiBaseUrl}
            humanPlayer={humanPlayer}
            settings={settings}
            showMoveHints={showMoveHints}
            showAiHints={showAiHints}
            onLatency={recordLatency}
            onRestart={() => setLatencyHistory([])}
          />
        </div>

        <aside className="w-full lg:col-start-3 lg:w-72 lg:justify-self-start">{settingsPanel}</aside>
      </div>

      <section className="mx-auto max-w-[1000px] border-t border-hairline px-6 py-14">
        <h2 className="mb-6 font-display text-2xl font-semibold text-ink">Search diagnostics</h2>
        <div className={settings.cpuctType === 'static' ? '' : 'grid grid-cols-1 gap-10 lg:grid-cols-2'}>
          <div>
            <h3 className="mb-1 font-sans text-sm font-semibold text-ink">Response time</h3>
            <p className="mb-4 font-serif text-sm text-ink-muted">
              How long each of the AI&apos;s searches took, move by move.
            </p>
            <TimingChart history={latencyHistory} />
          </div>
          {settings.cpuctType !== 'static' && (
            <div>
              <h3 className="mb-1 font-sans text-sm font-semibold text-ink">c_puct schedule</h3>
              <p className="mb-4 font-serif text-sm text-ink-muted">
                How c_puct itself changes as a position gets visited more, given the current{' '}
                {settings.cpuctType} setting and c_puct value.
              </p>
              <CpuctChart cpuctType={settings.cpuctType} cPuct={settings.cPuct} />
            </div>
          )}
        </div>
      </section>
    </>
  )
}

interface GameProps {
  apiBaseUrl: string
  humanPlayer: Player
  settings: EngineSettings
  showMoveHints: boolean
  showAiHints: boolean
  onLatency: (elapsedMs: number) => void
  onRestart: () => void
}

function Game({
  apiBaseUrl,
  humanPlayer,
  settings,
  showMoveHints,
  showAiHints,
  onLatency,
  onRestart,
}: GameProps) {
  const { state, aiThinking, error, playHumanMove, restart } = useOthelloGame(
    apiBaseUrl,
    humanPlayer,
    settings,
    onLatency
  )

  const isHumanTurn = !state.gameOver && state.currentPlayer === humanPlayer
  const legalMoves = isHumanTurn ? getValidMoves(state.board, humanPlayer) : []
  const isLegal = (row: number, col: number) => legalMoves.some((m) => m.row === row && m.col === col)

  // "AI hints" is specifically what the AI recommends for *your* move, so
  // it only ever runs during your turn -- there's nothing to recommend
  // while the AI is deciding its own.
  const hintProbs = useMoveHints(apiBaseUrl, state.board, humanPlayer, showAiHints && isHumanTurn, settings)

  const handleRestart = () => {
    restart()
    onRestart()
  }

  return (
    <>
      <GameStatusBar state={state} humanPlayer={humanPlayer} aiThinking={aiThinking} error={error} />

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(8, 1fr)',
          gap: 2,
          marginTop: '1rem',
          marginBottom: '1rem',
        }}
      >
        {state.board.map((value, index) => {
          const row = Math.floor(index / 8)
          const col = index % 8
          return (
            <Cell
              key={index}
              value={value as -1 | 0 | 1}
              isLegalMove={isLegal(row, col)}
              showMoveDot={showMoveHints}
              hintProb={hintProbs ? hintProbs[index] : 0}
              onClick={() => playHumanMove({ row, col })}
            />
          )
        })}
      </div>

      <div className="flex justify-center">
        <button
          type="button"
          onClick={handleRestart}
          className="cursor-pointer rounded-sm bg-signal px-4 py-2 font-sans text-sm font-medium text-white transition-colors hover:bg-signal-ink"
        >
          New game
        </button>
      </div>
    </>
  )
}
