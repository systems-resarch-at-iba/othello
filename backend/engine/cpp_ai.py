import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from utils import dotdict

from .weight_export import pack_state_dict

CPP_BUILD_DIR = Path(__file__).resolve().parent.parent / "cpp" / "build"
DEFAULT_BATCH_SIZE = 4
# A search tree of this many positions takes about 45 MB.
MAX_TREE_NODES = 200_000


def load_cpp_module() -> ModuleType:
    """
    Imports the compiled C++ engine module.

    The module is searched in the Python path first and then in the build
    directory next to this package.

    Returns
    -------
    types.ModuleType
        The `othello_cpp` module.

    Raises
    ------
    ImportError
        When the module has not been built.
    """
    try:
        import othello_cpp
    except ImportError:
        if str(CPP_BUILD_DIR) not in sys.path:
            sys.path.insert(0, str(CPP_BUILD_DIR))
        import othello_cpp
    return othello_cpp


class CppSearch:
    """
    Search settings and a get_action_probs method over the C++ engine.

    Attributes
    ----------
    args : dotdict
        Search settings. `num_mcts_sims` and `c_puct` are read on every call.
    cpuct_type : str
        Scaling of the exploration constant: "static", "increment" or "decrement".
    use_dcpuct : bool
        Whether the exploration constant is scaled.
    batch_size : int
        Simulations whose leaves are evaluated together.
    """

    def __init__(self, engine, args: dotdict, batch_size: int):
        self.engine = engine
        self.args = args
        self.cpuct_type = "static"
        self.use_dcpuct = False
        self.batch_size = batch_size

    def get_action_probs(self, canonical_board: np.ndarray, temperature: float = 1) -> list[float]:
        """
        Searches a position and returns the action distribution.

        Parameters
        ----------
        canonical_board : numpy.ndarray
            8x8 board seen from the side to move, whose discs are 1.
        temperature : float, optional
            0 gives a one-hot most visited action with ties broken at random.
            Otherwise the visit counts are raised to 1 / temperature and normalized.

        Returns
        -------
        list of float
            One probability for each of the 64 squares and the pass action.
        """
        probs = self.engine.get_action_probs(
            canonical_board,
            1,
            int(self.args.num_mcts_sims),
            float(temperature),
            float(self.args.c_puct),
            self.cpuct_type,
            int(np.random.randint(0, 2**31)),
            self.batch_size,
        )
        return probs.tolist()

    def clear_memory(self) -> None:
        """
        Drops the search tree.
        """
        self.engine.clear()


class CppAI:
    """
    Plays Othello with the C++ search and the network of a trained checkpoint.

    `select_move` returns the most visited move, and `mcts` holds the search.

    Attributes
    ----------
    mcts : CppSearch
        The search object.
    parameter_count : int
        Number of parameters of the loaded network.
    isa : str
        Instruction set the network uses: "avx2" or "scalar".
    """

    def __init__(
        self,
        args: dotdict,
        version: str = "1",
        model_dir: str | None = None,
        model_file: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Loads a checkpoint and builds the C++ engine.

        Parameters
        ----------
        args : dotdict
            Search settings.
        version : str, optional
            Model version used to name the checkpoint file.
        model_dir : str, optional
            Directory of the checkpoint. Defaults to "./models/".
        model_file : str, optional
            Checkpoint file name. Defaults to "model-{version}.pth.tar".
        batch_size : int, optional
            Simulations whose leaves are evaluated together.

        Raises
        ------
        FileNotFoundError
            When the checkpoint does not exist.
        ImportError
            When the C++ module has not been built.
        """
        path = Path(model_dir or "./models/") / (model_file or f"model-{version}.pth.tar")
        if not path.exists():
            raise FileNotFoundError(f"No model found in path {path}")
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        self._build(state_dict, args, batch_size)

    @classmethod
    def from_state_dict(cls, state_dict: dict, args: dotdict, batch_size: int = DEFAULT_BATCH_SIZE) -> "CppAI":
        """
        Builds the engine from network parameters.

        Parameters
        ----------
        state_dict : dict
            Parameters of the network.
        args : dotdict
            Search settings.
        batch_size : int, optional
            Simulations whose leaves are evaluated together.

        Returns
        -------
        CppAI
            The engine.
        """
        ai = cls.__new__(cls)
        ai._build(state_dict, args, batch_size)
        return ai

    def _build(self, state_dict: dict, args: dotdict, batch_size: int) -> None:
        othello_cpp = load_cpp_module()
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "weights.bin"
            weights.write_bytes(pack_state_dict(state_dict))
            engine = othello_cpp.Engine(str(weights), MAX_TREE_NODES)
        self.isa = engine.isa
        self.parameter_count = sum(int(tensor.numel()) for tensor in state_dict.values())
        self.mcts = CppSearch(engine, dotdict(dict(args)), batch_size)

    def select_move(self, board_state: np.ndarray, player: int = 1) -> int:
        """
        Selects the most visited move for a player.

        Parameters
        ----------
        board_state : numpy.ndarray
            8x8 board with 1 for white, -1 for black and 0 for empty.
        player : int
            Player to move, 1 for white or -1 for black.

        Returns
        -------
        int
            Index of the move (0 to 63), or 64 to pass.
        """
        canonical_board = player * board_state
        return int(np.argmax(self.mcts.get_action_probs(canonical_board, temperature=0)))
