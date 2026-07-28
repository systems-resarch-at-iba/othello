from typing import List, Tuple
import numpy as np
import random

class OthelloBitBoard:
    """
    Bitboard-based Othello board representation.
    Efficiently simulates game logic using 64-bit integers.
    """

    index_to_coord = None  # Static mapping from index to (row, col) coordinates
    coord_to_index = None  # Static mapping from (row, col) coordinates to index


    def __init__(self):
        """
        Initialize the board to the standard starting state.
        """
        if OthelloBitBoard.index_to_coord is None or OthelloBitBoard.coord_to_index is None:
            OthelloBitBoard._precompute_maps()
        self.reset()

    @property
    def representation(self) -> str:
        return "OthelloBitBoard"
    
    @classmethod
    def _precompute_maps(cls):
        """
        Precompute static mappings for index to coordinates and vice versa.
        This avoids repeated calculations during gameplay.
        Assumes column-major order: index = col * 8 + row
        """
        cls.index_to_coord = {}
        cls.coord_to_index = {}
        for index in range(64):
            row, col = index >> 3, index & 7
            coord = (row, col)
            cls.index_to_coord[index] = coord
            cls.coord_to_index[coord] = index
        cls.index_to_coord[64] = (-1, -1) # For Pass
        cls.coord_to_index[(-1, -1)] = 64  # For Pass action
    
    @staticmethod
    def _to_coord(index: int) -> Tuple[int, int]:
        """
        Convert a flat bit index to (row, col) coordinates on an 8x8 board.

        This replaces the slower `index // 8` and `index % 8` with faster bitwise
        operations:
            - index >> 3  is equivalent to index // 8
            - index & 7   is equivalent to index % 8
        These work because 8 is a power of 2 (2^3), allowing us to use bit tricks
        for integer division and modulo.

        Bitwise operations are significantly faster than division/modulo in tight
        loops, especially when this function is called frequently (e.g., during 
        board iteration in bitboard-based representations).

        Parameters
        ----------
        index : int
            Bit index (0 to 63), where 0 is top-left and 63 is bottom-right.

        Returns
        -------
        Tuple[int, int]
            Row and column coordinates on the board.
        """
        return OthelloBitBoard.index_to_coord[index]

    @staticmethod
    def _to_index(row: int, col: int) -> int:
        """
        Convert (row, col) coordinates to a flat bit index.

        Parameters
        ----------
        row : int
            Row index (0 to 7).
        col : int
            Column index (0 to 7).

        Returns
        -------
        int
            Bit index (0 to 63).
        """
        return OthelloBitBoard.coord_to_index[(row, col)]
    
    @staticmethod
    def _to_notation(row: int, col: int) -> str:
        """
        Convert (row, col) coordinates to Othello board notation (e.g., 'f5').

        Parameters
        ----------
        row : int
            Row index (0 to 7), where 0 is the top row.
        col : int
            Column index (0 to 7), where 0 is the leftmost column.

        Returns
        -------
        str
            Board position in standard notation (e.g., 'a1', 'h8').
        If the coordinates are (-1, -1), returns "ps" for pass.
        """
        if (row, col) == (-1, -1):
            return "ps"
        return chr(ord('a') + col) + str(row + 1)
    
    @staticmethod
    def _to_coords_from_notation(notation: str) -> Tuple[int, int]: 
        """
        Convert Othello board notation (e.g., 'f5') to (row, col) coordinates.

        Parameters
        ----------
        notation : str
            Board position in standard notation (e.g., 'a1', 'h8').

        Returns
        -------
        Tuple[int, int]
            Row and column indices (0 to 7).
        """
        col = ord(notation[0]) - ord('a')
        row = int(notation[1]) - 1
        return row, col
    
    @staticmethod 
    def _to_index_from_notation(notation: str) -> int:
        """
        Convert Othello board notation (e.g., 'f5') to a flat bit index.

        Parameters
        ----------
        notation : str
            Board position in standard notation (e.g., 'a1', 'h8').

        Returns
        -------
        int
            Bit index (0 to 63) corresponding to the position.
        """
        row, col = OthelloBitBoard._to_coords_from_notation(notation)
        return OthelloBitBoard.coord_to_index[(row, col)]

    def reset(self) -> None:
        """
        Reset the board to the initial Othello state.
        """
        self.black = 0x0000000810000000
        self.white = 0x0000001008000000

    def count_diff(self, color: int) -> int:
        """
        Compute the difference in piece count between player and opponent.

        Parameters
        ----------
        color : int
            1 for white, -1 for black.

        Returns
        -------
        int
            Number of own pieces - number of opponent pieces.
        """
        my_count = bin(self.white if color == 1 else self.black).count("1")
        opp_count = bin(self.black if color == 1 else self.white).count("1")
        return my_count - opp_count

    def score(self) -> Tuple[int, int]:
        """
        Count total pieces for black and white.

        Returns
        -------
        Tuple[int, int]
            (black_score, white_score)
        """
        white_score = bin(self.white).count("1")
        black_score = bin(self.black).count("1")
        return black_score, white_score
    
    def get_board_state(self) -> List[List[int]]:
        """
        Get board as a nested list with -1, 0, 1 for black, empty, white.

        Returns
        -------
        List[List[int]]
            8x8 board where:
            -1 = black, 0 = empty, 1 = white
        """
        flat = [0] * 64
        b, w = self.black, self.white
        combined = b | w

        while combined:
            lsb = combined & -combined
            i = lsb.bit_length() - 1
            combined ^= lsb

            flat[i] = -1 if (b >> i) & 1 else 1

        # Reshape manually to 8x8 nested list
        return [flat[i:i+8] for i in range(0, 64, 8)]

    def get_board_state_as_numpy(self) -> np.ndarray:
        """
        Get board as a NumPy array with -1, 0, 1 for black, empty, white.

        Returns
        -------
        np.ndarray
            8x8 integer array.
        """
        flat = np.zeros(64, dtype=np.int8)

        b, w = self.black, self.white
        combined = b | w

        while combined:
            lsb = combined & -combined
            i = lsb.bit_length() - 1
            combined ^= lsb

            flat[i] = -1 if (b >> i) & 1 else 1

        return flat.reshape((8, 8))
    
    def get_board_as_string(self, current_player: int) -> str:
        """
        Returns a string representation of the board.

        - 'B' for black pieces
        - 'W' for white pieces
        - '-' for empty squares
        - Final character indicates whose turn it is ('B' or 'W')

        Parameters
        ----------
        current_player : int
            -1 for black's turn, 1 for white's turn.

        Returns
        -------
        str
            Flattened board as a string with a final B/W for current player.
        """
        result = []
        for i in range(64):
            mask = 1 << i
            if self.black & mask:
                result.append('B')
            elif self.white & mask:
                result.append('W')
            else:
                result.append('-')
        result.append('B' if current_player == -1 else 'W')
        return ''.join(result)

    def get_valid_moves(self, color: int) -> List[Tuple[int, int]]:
        """
        Get list of valid (row, col) moves for the given color.

        Parameters
        ----------
        color : int
            1 for white, -1 for black.

        Returns
        -------
        List[Tuple[int, int]]
            List of legal moves as (row, col) tuples.
        """
        own = self.black if color == -1 else self.white
        opp = self.white if color == -1 else self.black
        moves = 0

        for shift, mask in self.__bit_directions():
            temp = self._shift(own, shift, mask) & opp
            potential = 0
            while temp:
                potential |= temp
                temp = self._shift(temp, shift, mask) & opp
            outflank = self._shift(potential, shift, mask) & ~(own | opp)
            moves |= outflank

        return [OthelloBitBoard._to_coord(i) for i in range(64) if (moves >> i) & 1]
    
    def get_valid_move_indices(self, color: int) -> List[int]:
        """
        Returns a list of length 65:
        - Indices 0 to 63: 1 if the move is legal, else 0.
        - Index 64: 1 if no legal moves exist, else 0.
        """
        own = self.black if color == -1 else self.white
        opp = self.white if color == -1 else self.black
        moves = 0

        for shift, mask in self.__bit_directions():
            temp = self._shift(own, shift, mask) & opp
            if not temp:
                continue
            potential = 0
            while temp:
                potential |= temp
                temp = self._shift(temp, shift, mask) & opp
            outflank = self._shift(potential, shift, mask) & ~(own | opp)
            moves |= outflank

        # Fast path: if no moves at all
        if moves == 0:
            result = [0] * 64
            result.append(1)  # index 64
            return result

        # If moves exist
        result = [(moves >> i) & 1 for i in range(64)]
        result.append(0)  # index 64
        return result

    def has_valid_moves(self, color: int) -> bool:
        """
        Check if player has at least one legal move.
        Optimized to return True as soon as the first move is found.

        Parameters
        ----------
        color : int
            1 for white, -1 for black.

        Returns
        -------
        bool
            True if any move is possible, False otherwise.
        """
        own = self.black if color == -1 else self.white
        opp = self.white if color == -1 else self.black

        for shift, mask in self.__bit_directions():
            temp = self._shift(own, shift, mask) & opp
            potential = 0
            while temp:
                potential |= temp
                temp = self._shift(temp, shift, mask) & opp
            outflank = self._shift(potential, shift, mask) & ~(own | opp)

            # If any bit is set in outflank, it means there's at least one valid move
            if outflank != 0:
                return True # Found a valid move, no need to check further

        # If the loop finishes, no valid moves were found
        return False

    def apply_move(self, move: Tuple[int, int], color: int) -> None:
        """
        Apply a move and flip captured pieces.

        Parameters
        ----------
        move : Tuple[int, int]
            Coordinates of the move (row, col).
        color : int
            1 for white, -1 for black.
        """
        row, col = move
        index = OthelloBitBoard._to_index(row, col)
        move_mask = 1 << index

        own = self.black if color == -1 else self.white
        opp = self.white if color == -1 else self.black

        to_flip = 0
        for shift, mask in self.__bit_directions():
            flipped = 0
            cur = self._shift(move_mask, shift, mask)
            while cur and (cur & opp):
                flipped |= cur
                cur = self._shift(cur, shift, mask)
            if cur & own:
                to_flip |= flipped

        if color == -1:
            self.black |= move_mask | to_flip
            self.white &= ~to_flip
        else:
            self.white |= move_mask | to_flip
            self.black &= ~to_flip

    def _shift(self, bb: int, direction: int, mask: int) -> int:
        """
        Shift a bitboard in a given direction with masking.

        Parameters
        ----------
        bb : int
            Bitboard to shift.
        direction : int
            Bit shift amount (positive = left, negative = right).
        mask : int
            Mask to apply after the shift.

        Returns
        -------
        int
            Shifted and masked bitboard.
        """
        if direction > 0:
            return (bb << direction) & mask
        else:
            return (bb >> -direction) & mask

    def random_position(self, max_moves: int = 20) -> int:
        """
        Play up to `max_moves` random legal moves and return the current player.

        Parameters
        ----------
        max_moves : int, optional
            Maximum number of moves to simulate (default is 20).

        Returns
        -------
        int
            Current player after simulation (1 or -1).
        """
        self.reset()
        player = -1
        for _ in range(max_moves):
            moves = self.get_valid_moves(player)
            if not moves:
                player = -player
                moves = self.get_valid_moves(player)
                if not moves:
                    break
            move = random.choice(moves)
            self.apply_move(move, player)
            player = -player
        return player

    @staticmethod
    def __bit_directions():
        """
        Directions and corresponding edge masks for Othello bitboard logic.

        Returns
        -------
        List[Tuple[int, int]]
            List of (shift, mask) tuples for each of 8 directions.
        """
        return [
            (8,  0xFFFFFFFFFFFFFFFF),  # N
            (-8, 0xFFFFFFFFFFFFFFFF),  # S
            (1,  0xFEFEFEFEFEFEFEFE),  # E
            (-1, 0x7F7F7F7F7F7F7F7F),  # W
            (9,  0xFEFEFEFEFEFEFEFE),  # NE
            (-9, 0x7F7F7F7F7F7F7F7F),  # SW
            (7,  0x7F7F7F7F7F7F7F7F),  # NW
            (-7, 0xFEFEFEFEFEFEFEFE),  # SE
        ]
