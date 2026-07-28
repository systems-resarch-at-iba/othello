import { describe, expect, it } from 'vitest'
import { advanceTurn, applyMove, createInitialBoard, createInitialState, getValidMoves, hasValidMoves } from '../src/lib/othello-rules'

describe('createInitialBoard', () => {
  it('matches the backend bitboard starting layout exactly', () => {
    // Confirmed against the backend's own starting bitboard values
    // (black=0x0000000810000000, white=0x0000001008000000): white on
    // (3,3)/(4,4), black on (3,4)/(4,3).
    const board = createInitialBoard()
    expect(board[3 * 8 + 3]).toBe(1)
    expect(board[4 * 8 + 4]).toBe(1)
    expect(board[3 * 8 + 4]).toBe(-1)
    expect(board[4 * 8 + 3]).toBe(-1)
    expect(board.filter((c) => c !== 0)).toHaveLength(4)
  })
})

describe('getValidMoves', () => {
  it("gives black the 4 classic Othello opening moves", () => {
    const board = createInitialBoard()
    const moves = getValidMoves(board, -1)
    const asSet = new Set(moves.map((m) => `${m.row},${m.col}`))
    expect(asSet).toEqual(new Set(['2,3', '3,2', '4,5', '5,4']))
  })

  it('has no valid moves on an empty board', () => {
    const empty = new Array(64).fill(0)
    expect(getValidMoves(empty, -1)).toHaveLength(0)
    expect(hasValidMoves(empty, -1)).toBe(false)
  })
})

describe('applyMove', () => {
  it('places the piece and flips exactly the captured line, without mutating the input', () => {
    const board = createInitialBoard()
    const next = applyMove(board, { row: 2, col: 3 }, -1)

    expect(next[2 * 8 + 3]).toBe(-1) // the placed piece
    expect(board[3 * 8 + 3]).toBe(1) // original board untouched (still white)
    expect(next[3 * 8 + 3]).toBe(-1) // captured: was white, now black

    // Only the placed piece and the one captured piece should have changed.
    const changed = next.filter((v, i) => v !== board[i])
    expect(changed).toHaveLength(2)
  })
})

describe('advanceTurn', () => {
  it("switches to the other player after a normal move", () => {
    const state = createInitialState()
    const next = advanceTurn(state, { row: 2, col: 3 })
    expect(next.currentPlayer).toBe(1)
    expect(next.gameOver).toBe(false)
    expect(next.lastMove).toEqual({ row: 2, col: 3 })
  })

  it('is a no-op once the game is already over', () => {
    const state = { board: createInitialBoard(), currentPlayer: -1 as const, gameOver: true, lastMove: null }
    const next = advanceTurn(state, { row: 2, col: 3 })
    expect(next).toBe(state)
  })

  it('skips back to the same mover when the opponent has no legal move after that move', () => {
    // A real position reached via simulated play (not hand-constructed):
    // white plays (1,3), which leaves black with no legal reply anywhere on
    // the board, so play returns to white instead of switching to black.
    // prettier-ignore
    const board = [
      -1,  1,  1,  1,  1,  1,  1,  1,
      -1, -1,  1,  0,  1,  1,  1,  1,
      -1, -1, -1,  1, -1,  1,  1,  1,
      -1, -1, -1, -1, -1, -1,  1,  1,
       1, -1,  1,  1,  1,  1, -1, -1,
       1, -1,  1, -1, -1,  1,  1,  1,
      -1, -1, -1, -1, -1, -1, -1,  1,
      -1, -1, -1, -1,  0, -1,  0,  1,
    ]
    const state = { board, currentPlayer: 1 as const, gameOver: false, lastMove: null }
    const next = advanceTurn(state, { row: 1, col: 3 })

    expect(hasValidMoves(next.board, -1)).toBe(false)
    expect(next.currentPlayer).toBe(1) // stayed with white, didn't switch to black
    expect(next.gameOver).toBe(false)
  })
})
