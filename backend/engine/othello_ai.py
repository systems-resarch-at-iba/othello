import numpy as np
from typing import Type
import torch
import torch.nn as nn
from .mcts import MCTS
from .othello_cnn import OthelloCNN

from utils import dotdict
from board import Utils as BU

class OthelloAI:
    """
    A class that wraps the neural network and MCTS to play Othello.

    Attributes
    ----------
    nnet : OthelloCNN
        The neural network model used for evaluating board positions.
    mcts : MCTS
        The Monte Carlo Tree Search object used to select actions.
    """

    def __init__(self, cnn: Type[nn.Module], args: dotdict, version: int = 1, model_dir: str | None = None, model_file: str | None = None):
        """
        Initialize the OthelloAI with a trained neural network and MCTS.

        Parameters
        ----------
        version : int, optional
            Version of the model to load. Default is 1.
        model_dir : str, optional
            Directory where the model checkpoint is stored. Default is None, which will use the default directory "./models/".
        model_file : str, optional
            Filename of the model checkpoint. Default is None, which will use the default naming convention based on the version.
        """
        self.nnet = OthelloCNN(cnn, args)

        if model_dir is None:
            self.model_dir = "./models/"
        else:
            self.model_dir = model_dir
        
        if model_file is None:
            self.model_file = f"model-{version}.pth.tar"
        else:
            self.model_file = model_file

        self.nnet.load_checkpoint(self.model_dir, self.model_file)

        self.mcts = MCTS(self.nnet, args)

    def select_move(self, board_state: np.ndarray, player: int = 1) -> int:
        """
        Select the best move from a given board state using MCTS and the neural network.

        Parameters
        ----------
        board_state : np.ndarray
            The current board state as an 8x8 NumPy array, with values:
            - 1 for white
            - -1 for black
            - 0 for empty
        player : int
            The player for whom to select the move (1 for white, -1 for black).

        Returns
        -------
        int
            The index of the best move (0 to 63), or 64 if the player chooses to pass.
        """
        canonical_board = BU.get_canonical_form(board_state, player)
        action_probs = self.mcts.get_action_probs(canonical_board, temperature=0)
        return int(np.argmax(action_probs))
