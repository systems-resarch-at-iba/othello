from .othello_bit_board import OthelloBitBoard
import numpy as np

from configs import BOARD_SIZE


class Utils():

    @staticmethod
    def get_init_board():
        # return initial board as numpy array
        b = OthelloBitBoard()
        return b.get_board_state_as_numpy()

    @staticmethod
    def get_next_state(board, player, action):
        if action == BOARD_SIZE ** 2: return (board, -player)

        b = OthelloBitBoard()
        Utils._load_state(b, board)
        move = OthelloBitBoard._to_coord(action)
        b.apply_move(move, player)
        return (b.get_board_state_as_numpy(), -player)

    @staticmethod
    def has_valid_moves(board, player) -> bool:
        # Check if the player has any valid moves
        b = OthelloBitBoard()
        Utils._load_state(b, board)
        return b.has_valid_moves(player)

    @staticmethod
    def get_valid_moves(board, player) -> np.ndarray:
        b = OthelloBitBoard()
        Utils._load_state(b, board)

        valids = b.get_valid_move_indices(player)
        return np.array(valids, dtype=np.uint8)

    @staticmethod
    def get_game_ended(board, player):
        b = OthelloBitBoard()
        Utils._load_state(b, board)

        if b.has_valid_moves(player) or b.has_valid_moves(-player):
            return 0
        diff = b.count_diff(player)
        return 1 if diff > 0 else -1 if diff < 0 else 1e-4  # draw

    @staticmethod
    def get_canonical_form(board, player):
        return player * board

    @staticmethod
    def _load_state(b: OthelloBitBoard, state: np.ndarray):
        """
        Helper: Load a numpy 2D board into the bitboard.
        Used to convert input numpy board into bit representation.
        """
        b.black = 0
        b.white = 0
        # Get flat indices of black and white pieces
        black_indices = np.flatnonzero(state == -1)
        white_indices = np.flatnonzero(state == 1)

        # Set the bits for black and white
        for idx in black_indices: b.black |= (1 << int(idx))
        for idx in white_indices: b.white |= (1 << int(idx))
