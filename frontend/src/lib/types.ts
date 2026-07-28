/**
 * Row-major flat board of length 64 (index = row*8 + col), matching the
 * backend's own OthelloBitBoard index convention exactly, so a board can be
 * sent over the wire with no reshaping.
 */
export type Board = number[]

export type Player = 1 | -1

export type Cell = -1 | 0 | 1

export type Move = { row: number; col: number }

export interface GameState {
  board: Board
  currentPlayer: Player
  gameOver: boolean
  lastMove: Move | null
}

/** See MCTS.get_dynamic_cpuct on the backend: how the exploration constant scales with visit count. */
export type CpuctType = 'static' | 'increment' | 'decrement'

export interface EngineSettings {
  numMctsSims: number
  cpuctType: CpuctType
  /** Base exploration constant (arch.py's own default is 1.5). See cpuctType for how it's scaled. */
  cPuct: number
}

export const DEFAULT_ENGINE_SETTINGS: EngineSettings = {
  numMctsSims: 500,
  cpuctType: 'static',
  cPuct: 1.5,
}
