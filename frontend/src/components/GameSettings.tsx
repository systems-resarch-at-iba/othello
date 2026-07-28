import type { CpuctType, EngineSettings, Player } from '../lib/types'
import { Dropdown } from './Dropdown'
import { NumberField } from './NumberField'
import { Toggle } from './Toggle'

interface GameSettingsProps {
  humanPlayer: Player
  onHumanPlayerChange: (player: Player) => void
  settings: EngineSettings
  onSettingsChange: (settings: EngineSettings) => void
  showMoveHints: boolean
  onShowMoveHintsChange: (show: boolean) => void
  showAiHints: boolean
  onShowAiHintsChange: (show: boolean) => void
}

const PLAYER_OPTIONS: { value: Player; label: string }[] = [
  { value: -1, label: 'Black (moves first)' },
  { value: 1, label: 'White' },
]

const CPUCT_TYPES: { value: CpuctType; label: string }[] = [
  { value: 'static', label: 'Static' },
  { value: 'increment', label: 'Increment (starts low, rises toward c_puct)' },
  { value: 'decrement', label: 'Decrement (starts at c_puct, decays toward 0.5)' },
]

export function GameSettings({
  humanPlayer,
  onHumanPlayerChange,
  settings,
  onSettingsChange,
  showMoveHints,
  onShowMoveHintsChange,
  showAiHints,
  onShowAiHintsChange,
}: GameSettingsProps) {
  return (
    <div className="font-sans text-sm text-ink">
      <h3 className="mb-3 border-b border-hairline pb-3 font-sans text-xs font-semibold uppercase tracking-widest text-ink-muted">
        Settings
      </h3>

      <div className="flex flex-col gap-4">
        <label className="flex flex-col gap-1">
          Play as
          <Dropdown value={humanPlayer} options={PLAYER_OPTIONS} onChange={onHumanPlayerChange} />
        </label>

        <label className="flex flex-col gap-1">
          MCTS simulations
          <NumberField
            min={1}
            max={500}
            value={settings.numMctsSims}
            onChange={(n) => onSettingsChange({ ...settings, numMctsSims: n })}
          />
        </label>

        <label className="flex flex-col gap-1">
          c_puct scaling
          <Dropdown
            value={settings.cpuctType}
            options={CPUCT_TYPES}
            onChange={(v) => onSettingsChange({ ...settings, cpuctType: v })}
          />
        </label>

        <label className="flex flex-col gap-1">
          c_puct value
          <NumberField
            min={0.1}
            max={5}
            step={0.1}
            value={settings.cPuct}
            onChange={(n) => onSettingsChange({ ...settings, cPuct: n })}
          />
        </label>

        <div className="flex flex-col gap-3 border-t border-hairline pt-4">
          <div>
            <Toggle checked={showMoveHints} onChange={onShowMoveHintsChange}>
              Show move hints
            </Toggle>
            <p className="mt-1 font-sans text-xs text-ink-muted">
              Marks every square you can legally play.
            </p>
          </div>
          <div>
            <Toggle checked={showAiHints} onChange={onShowAiHintsChange}>
              Show AI hints
            </Toggle>
            <p className="mt-1 font-sans text-xs text-ink-muted">
              Shades squares by how strongly the AI recommends them for your move.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
