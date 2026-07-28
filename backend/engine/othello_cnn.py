import os
from typing import Type

import numpy as np
import torch
import torch.nn as nn

from utils import dotdict, LoggerSetup
from configs import BOARD_SIZE

default_args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 3,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'conv_channels': 512,
    'log_level': "INFO"
})


class OthelloCNN():
    """
    This class encapsulates the OthelloCNN model (defined as CNN in cnn.py)
    and provides methods for prediction and checkpoint loading.

    Parameters
    ----------
    cnn : nn.Module
        The CNN model to be used for Othello.
        Can be any CNN class with a compatible interface.
    args : dotdict | None, optional
        A dictionary object containing configuration parameters for the CNN.
        If None, the `default_args` variable will be used.

    Attributes
    ----------
    cnn : CNN
        The neural network model for Othello.
    board_size : int
        The size of one side of the Othello board (e.g., 8 for an 8x8 board).
    action_size : int
        The total number of possible actions (BOARD_SIZE * BOARD_SIZE + 1 for pass).
    """
    def __init__(self, cnn: Type[nn.Module], args: dotdict | None = None) -> None:
        """
        Initializes the OthelloCNN class.
        Sets up the neural network model and moves it to GPU if CUDA is available.
        """

        self.args = args
        if self.args is None:
            self.args = default_args

        self.cnn = cnn(self.args)

        self.board_size = BOARD_SIZE
        self.action_size = BOARD_SIZE * BOARD_SIZE + 1

        if self.args.cuda:
            self.cnn.cuda()

        setup = LoggerSetup(log_name=f"{self.__class__.__name__}", log_level=self.args.log_level, filename="othello_cnn.log")

        self.log = setup.get_logger()

    def predict(self, board: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Performs a prediction using the trained OthelloCNN model on a single board state.

        Parameters
        ----------
        board : numpy.ndarray
            The Othello board state as a NumPy array.
            Expected shape: (BOARD_SIZE, BOARD_SIZE).

        Returns
        -------
        tuple
            A tuple containing:
            - policy_probabilities (numpy.ndarray): The predicted policy probabilities over the action space.
            - value (float): The predicted win probability from the current board state.
        """
        # Convert board to PyTorch FloatTensor
        board = torch.FloatTensor(board.astype(np.float64))

        # Move board to GPU if CUDA is enabled
        if self.args.cuda:
            board = board.contiguous().cuda()

        # Reshape the board to (1, BOARD_SIZE, BOARD_SIZE) for batch processing
        board = board.view(1, self.board_size, self.board_size)

        self.cnn.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            policy, value = self.cnn(board)

        # Convert policy probabilities from log-softmax to actual probabilities and move to CPU
        # Convert value to numpy and move to CPU
        return torch.exp(policy).data.cpu().numpy()[0], value.data.cpu().numpy()[0]

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar') -> None:
        """
        Loads the model's weights from a saved checkpoint file.

        Parameters
        ----------
        folder : str, optional
            The directory where the checkpoint file is located. Defaults to 'checkpoint'.
        filename : str, optional
            The name of the checkpoint file. Defaults to 'checkpoint.pth.tar'.

        Raises
        ------
        FileNotFoundError
            If the specified checkpoint file does not exist.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found in path {filepath}")

        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.cnn.load_state_dict(checkpoint['state_dict'])
