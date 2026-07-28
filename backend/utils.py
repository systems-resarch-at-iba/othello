import logging
import coloredlogs
import os
import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and current value.

    This class is commonly used in deep learning training loops to track metrics
    like loss or accuracy over an epoch.

    Attributes
    ----------
    val : float
        The most recently updated value.
    avg : float
        The current running average of all updated values.
    sum : float
        The cumulative sum of all updated values.
    count : int
        The total number of updates (or samples seen).

    References
    ----------
    .. [1] https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        """
        Returns a string representation of the current average.

        Returns
        -------
        str
            A string showing the average value formatted in scientific notation
            with 2 decimal places.
        """
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Parameters
        ----------
        val : float
            The new value to add to the meter.
        n : int, optional
            The number of samples/observations associated with `val`.
            Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    """
    A dictionary subclass that allows accessing dictionary keys as attributes.

    This can make code more readable by allowing `obj.key` instead of `obj['key']`.

    Examples
    --------
    >>> d = dotdict({'a': 1, 'b': 2})
    >>> d.a
    1
    >>> d.b
    2
    """

    def __getattr__(self, name):
        """
        Allows accessing dictionary values using dot notation (e.g., `obj.key`).

        Parameters
        ----------
        name : str
            The name of the attribute (which corresponds to a dictionary key).

        Returns
        -------
        Any
            The value associated with the given key.

        Raises
        ------
        AttributeError
            If the `name` (key) does not exist in the dictionary.
        """
        return self[name]

    def __setattr__(self, name, value):
        """
        Allows setting dictionary values using dot notation (e.g., `obj.key = value`).

        Parameters
        ----------
        name : str
            The name of the attribute (which corresponds to a dictionary key).
        value : Any
            The value to set for the given key.
        """
        self[name] = value


class LoggerSetup:
    """
    A class to set up a logger with console (colored) and file logging.
    Logs are appended to a specified file within a 'logs/' directory.
    """
    def __init__(self, log_name: str, log_level: str = 'INFO', filename: str = 'app.log'):
        """
        Initializes the logger.

        Parameters
        ----------
        log_name : str, optional
            The name of the logger. Defaults to the name of the current module.
        log_level : str, optional
            The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
            Defaults to 'INFO'.
        filename : str, optional
            The base name of the log file (e.g., 'app.log').
            The file will be saved in a 'logs/' directory: 'logs/app.log'.
            Logs will be appended to this file across runs.
        """
        self.log = logging.getLogger(log_name)
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')

        self.log.setLevel(numeric_level)

        if not self.log.handlers:
            coloredlogs.install(level=numeric_level, logger=self.log)

            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_file_path = os.path.join(log_dir, filename)

            if os.path.exists(log_file_path):
                # If the log file already exists, append a newline to separate logs
                with open(log_file_path, 'a') as f:
                    f.write('\n')

            file_handler = logging.FileHandler(log_file_path, mode='a')
            file_handler.setLevel(numeric_level)

            file_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
            file_handler.setFormatter(file_formatter)

            self.log.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.
        """
        return self.log


class ZobristHasher:
    """
    64-bit Zobrist hashing for 8x8 Othello boards.

    The board is expected to contain values:
        -1 -> black
         0 -> empty
        +1 -> white
    """

    __slots__ = ("_table",)

    def __init__(self, seed: int = 0) -> None:
        """
        Initialize the Zobrist hasher with a fixed random seed.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator (default is 0).
        """
        rng = np.random.default_rng(seed)
        self._table: np.ndarray = rng.integers(
            0, 2**64, size=(8, 8, 3), dtype=np.uint64
        )

    def hash(self, board: np.ndarray) -> int:
        """
        Compute a 64-bit Zobrist hash for the given 8x8 board.

        Parameters
        ----------
        board : np.ndarray of shape (8, 8)
            The Othello board with entries -1 (black), 0 (empty), or +1 (white).

        Returns
        -------
        int
            A 64-bit hash value representing the board state.
        """
        idx = board + 1  # shift: -1 -> 0, 0 -> 1, 1 -> 2
        gathered = self._table[np.arange(8)[:, None], np.arange(8), idx]
        return int(np.bitwise_xor.reduce(gathered.flatten(), dtype=np.uint64))
