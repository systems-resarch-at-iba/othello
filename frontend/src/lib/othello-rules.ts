import type { Board, GameState, Move, Player } from './types'

const SIZE = 8
const DIRECTIONS: [number, number][] = [
  [-1, 0], [1, 0], [0, -1], [0, 1],
  [-1, -1], [-1, 1], [1, -1], [1, 1],
]

function toIndex(row: number, col: number): number {
  return row * SIZE + col
}

function inBounds(row: number, col: number): boolean {
  return row >= 0 && row < SIZE && col >= 0 && col < SIZE
}

/**
 * Standard Othello starting position: white on (3,3)/(4,4), black on
 * (3,4)/(4,3). Confirmed against the backend's own starting bitboard values
 * (0x0000000810000000 for black, 0x0000001008000000 for white) rather than
 * assumed, so the client and server never disagree on where a fresh game
 * actually starts.
 */
export function createInitialBoard(): Board {
  const board = new Array(SIZE * SIZE).fill(0)
  board[toIndex(3, 3)] = 1
  board[toIndex(4, 4)] = 1
  board[toIndex(3, 4)] = -1
  board[toIndex(4, 3)] = -1
  return board
}

/** Cells that would flip to `player` if they moved at (row, col), across all 8 directions. */
function flipsForMove(board: Board, row: number, col: number, player: Player): number[] {
  if (board[toIndex(row, col)] !== 0) return []

  const flips: number[] = []
  for (const [dr, dc] of DIRECTIONS) {
    const lineFlips: number[] = []
    let r = row + dr
    let c = col + dc
    while (inBounds(r, c) && board[toIndex(r, c)] === -player) {
      lineFlips.push(toIndex(r, c))
      r += dr
      c += dc
    }
    if (lineFlips.length > 0 && inBounds(r, c) && board[toIndex(r, c)] === player) {
      flips.push(...lineFlips)
    }
  }
  return flips
}

export function getValidMoves(board: Board, player: Player): Move[] {
  const moves: Move[] = []
  for (let row = 0; row < SIZE; row++) {
    for (let col = 0; col < SIZE; col++) {
      if (flipsForMove(board, row, col, player).length > 0) {
        moves.push({ row, col })
      }
    }
  }
  return moves
}

export function hasValidMoves(board: Board, player: Player): boolean {
  for (let row = 0; row < SIZE; row++) {
    for (let col = 0; col < SIZE; col++) {
      if (flipsForMove(board, row, col, player).length > 0) return true
    }
  }
  return false
}

/** Returns a new board with the move applied and captured pieces flipped; does not mutate `board`. */
export function applyMove(board: Board, move: Move, player: Player): Board {
  const flips = flipsForMove(board, move.row, move.col, player)
  const next = board.slice()
  next[toIndex(move.row, move.col)] = player
  for (const index of flips) next[index] = player
  return next
}

export function createInitialState(): GameState {
  return {
    board: createInitialBoard(),
    currentPlayer: -1, // black moves first, matching game.py's OthelloGame
    gameOver: false,
    lastMove: null,
  }
}

/**
 * Applies a move and advances turn state, mirroring game.py's OthelloGame.step
 * exactly: switch to the other player; if they have no legal move, switch
 * back (the mover goes again); if neither player has a legal move, the game
 * is over. Assumes `move` is already known-legal (callers filter against
 * getValidMoves before calling this, e.g. only offering legal squares in the UI).
 */
export function advanceTurn(state: GameState, move: Move): GameState {
  if (state.gameOver) return state

  const board = applyMove(state.board, move, state.currentPlayer)
  let nextPlayer = -state.currentPlayer as Player
  let gameOver = false

  if (!hasValidMoves(board, nextPlayer)) {
    nextPlayer = -nextPlayer as Player
    if (!hasValidMoves(board, nextPlayer)) {
      gameOver = true
    }
  }

  return { board, currentPlayer: nextPlayer, gameOver, lastMove: move }
}
