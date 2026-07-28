import math
import numpy as np
import logging

from typing import List, Tuple

EPS = 1e-8  # Small constant to avoid division by zero to prevent NaN or Inf

from .othello_cnn import OthelloCNN
from utils import dotdict, LoggerSetup, ZobristHasher
from board import Utils as BU

from configs import BOARD_SIZE


class MCTS:
    """
    This class implements the Monte Carlo Tree Search (MCTS) algorithm for Othello.
    It uses a Convolutional Neural Network (CNN) to guide its search, similar
    to the AlphaZero approach.

    Attributes
    ----------
    cnn : OthelloCNN
        The neural network for policy and value predictions.
    args : dotdict
        Configuration for MCTS, including number of simulations, c_puct, and temperature.
    action_size : int
        Total possible actions (BOARD_SIZE^2 + 1 for 'pass').
    Qsa, Nsa, Ns, Ps, Es, Vs : dicts
        Core MCTS memory structures.
    hasher : ZobristHasher
        Efficient hashing of board states.
    """

    def __init__(self, cnn: OthelloCNN, args: dotdict, cpuct_type: str = 'static'):
        """
        Initializes the MCTS agent with the game environment, neural network, and arguments.

        Parameters
        ----------
        cnn : OthelloCNN
            The neural network that provides policy and value predictions.
        args : dotdict
            Hyperparameters for MCTS.
        cpuct_type : str, optional
            Type of c_puct scaling to use. Options are 'static' for fixed c_puct,
            'increment' for dynamic incremental scaling, or 'decrement' for dynamic decremental scaling.
            Defaults to 'static'.
        """
        setup = LoggerSetup(log_name=f"{self.__class__.__name__}", log_level=args.log_level, filename="mcts.log")
        self.log = setup.get_logger()

        self.cnn = cnn
        self.args = args
        self.action_size = BOARD_SIZE * BOARD_SIZE + 1  # +1 for pass action
        self.Qsa = {}  # Q-values for state-action pairs (state_hash, action_index) -> Q_value
        self.Nsa = {}  # Visit counts for state-action pairs (state_hash, action_index) -> count
        self.Ns = {}  # Visit counts for states state_hash -> count
        self.Ps = {}  # Policy probabilities for states, predicted by CNN: state_hash -> policy_array
        self.Es = {}  # Game outcome for states: state_hash -> game_result (0 if not ended, 1 if win, -1 if loss/draw)
        self.Vs = {}  # Valid moves for board states: state_hash -> valid_moves_array (boolean)

        self.cpuct_type = cpuct_type
        if cpuct_type not in ['static', 'increment', 'decrement']:
            self.log.error(f"Invalid cpuct_type: {cpuct_type}. Defaulting to 'static'.")
            self.cpuct_type = 'static'

        self.use_dcpuct = False  # Flag to indicate if using dynamic c_puct scaling
        if self.cpuct_type != 'static':
            self.use_dcpuct = True

        self.hasher = ZobristHasher()  # Zobrist hashing for state representation

        self.total_depth = 0
        self.avg_depth = 0

    
    def compute_policy_entropy(self, masked_action_probs: List[float]) -> float:
        """
        Computes the normalized entropy of the action probabilities.

        Uses the formula:

            H(P) = -sum(p * log(p)) for all actions p in masked_action_probs

        Parameters
        ----------
        masked_action_probs : list of float
            A list of action probabilities, where each element represents the probability
            of taking a specific action. The list should be normalized (sum to 1) and masked
            to include only valid actions.

        Returns
        -------
        float
            The normalized entropy of the action probabilities, scaled to [0, 1].
        """
        masked_action_probs = np.array(masked_action_probs)
        entropy = -np.sum(masked_action_probs * np.log(masked_action_probs + EPS))  # Add EPS to avoid log(0)
        return entropy / np.log(self.action_size)  # Normalize by log(action_size) for scale [0, 1]

    def get_action_probs(self, canonical_board: np.ndarray, temperature: float = 1) -> List[float]:
        """
        Performs `num_mcts_sims` simulations of MCTS starting from the given
        canonical board state and returns the action probabilities based on visit counts.

        Parameters
        ----------
        canonical_board : numpy.ndarray
            The current canonical board state (from the perspective of the current player).
        temperature : float, optional
            The temperature parameter for controlling exploration. A higher temperature
            encourages more exploration (smoother distribution), while temp=0 selects
            the best action deterministically. Defaults to 1.

        Returns
        -------
        list of float
            A list of probabilities for each action in the action space, representing
            the MCTS policy.
        """
        self.total_depth = 0  # Reset total depth counter

        for _ in range(self.args.num_mcts_sims):
            self.search(canonical_board)

        self.avg_depth = self.total_depth / self.args.num_mcts_sims

        state = self.hasher.hash(canonical_board)
        counts = [self.Nsa[(state, action)] if (state, action) in self.Nsa else 0 for action in range(self.action_size)]

        if temperature == 0:
            # If temperature is 0, choose the action with the maximum visit count deterministically
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)  # Break ties randomly
            action_probs = [0] * len(counts)
            action_probs[best_action] = 1.0
            return action_probs

        # Apply temperature to visit counts and normalize to get probabilities
        tempered_counts = [count ** (1. / temperature) for count in counts]
        sum_tempered_counts = float(sum(tempered_counts)) + EPS  # Add EPS to avoid division by zero
        action_probs = [count / sum_tempered_counts for count in tempered_counts]
        return action_probs
    
    def exp_scale(self, x: int, start: float, approach: float, k: float = 0.001) -> float:
        """
        Exponential scaling function to adjust c_puct dynamically based on the number of visits.

        Parameters
        ----------
        x : int
            The number of visits to the state.
        start: float
            Initial c_puct value.
        approach: float
            The value that c_puct approaches as x increases.
        k : float, optional
            Decay rate constant (default: 0.001).

        Returns
        -------
        float
            The dynamically adjusted c_puct value.
        """
        return approach + (start - approach) * np.exp(-k * x)
    
    def get_dynamic_cpuct(self, state: str) -> float:
        """
        Computes a dynamically scaled exploration coefficient (c_puct) for the current state.
        
        This function adjusts c_puct based on the number of visits to the state, the scaling policy, 
        is defined by the cpuct_type attribute.

        cpuct_type can be:
        - 'static': 
            Returns the static c_puct value defined in args.cpuct.
        - 'decrement': 
            Returns a dynamically decreasing c_puct based on the number of visits.
            Starts at args.cpuct and approaches 0.5 as visits increase.
        - 'increment': 
            Returns a dynamically increasing c_puct based on the number of visits.
            Starts at 0.5 and approaches args.cpuct as visits increase.

        Note: The value `0.5` is chosen after experimentation to balance exploration and exploitation.

        Parameters
        ----------
        state : str
            The hashed representation of the current board state.

        Returns
        -------
        float
            The dynamically adjusted c_puct value based on the number of visits to the state and the set
            cpuct_type.   
        """
        if self.cpuct_type == 'static':
            return self.args.c_puct
        elif self.cpuct_type == 'decrement':
            return self.exp_scale(self.Ns.get(state, 0), start=self.args.c_puct, approach=0.5)
        elif self.cpuct_type == 'increment':
            return self.exp_scale(self.Ns.get(state, 0), start=0.5, approach=self.args.c_puct)
        else:
            self.log.error(f"Invalid cpuct_type: {self.cpuct_type}. Defaulting to static c_puct.")
            return self.args.c_puct
    
    def clear_memory(self):
        """
        Clears the MCTS memory structures to free up resources.
        This is useful when switching to a new game or resetting the MCTS state.
        """
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Es.clear()
        self.Vs.clear()
        self.total_depth = 0

    def search(self, canonical_board: np.ndarray, depth: int = 0) -> float:
        """
        Performs a single MCTS search iteration from the current canonical board state.
        This function recursively explores the game tree, uses the CNN for evaluation
        of new states, and backpropagates the resulting values.

        The value returned is from the perspective of the current player (who made the move
        to reach this state). If the game ends and the current player wins, it's +1.
        If the current player loses, it's -1.

        Parameters
        ----------
        canonical_board_state : numpy.ndarray
            The current canonical board state (from the perspective of the current player,
            whose pieces are represented as 1).

        Returns
        -------
        float
            The value (win/loss/draw outcome from the perspective of the current player)
            of the current board state. This value is backpropagated up the tree.
            NOTE: This function returns the negative of the value of the current
            canonical_board as per common MCTS implementations (value for parent).
        """
        state = self.hasher.hash(canonical_board)

        if state not in self.Es:
            self.Es[state] = BU.get_game_ended(canonical_board, 1) # 1 represents current player
        
        if self.Es[state] != 0:
            self.total_depth += depth  # Terminal node reached
            return -self.Es[state]

        if state not in self.Ps:
            predicted_policy, predicted_value = self.cnn.predict(canonical_board)
            self.Ps[state] = predicted_policy

            valid_moves = BU.get_valid_moves(canonical_board, 1)

            self.Ps[state] = self.Ps[state] * valid_moves
            sum_of_masked_policy = np.sum(self.Ps[state])

            if sum_of_masked_policy > 0:
                self.Ps[state] /= sum_of_masked_policy
            else:
                self.log.warning(f"All valid moves have been masked out. Setting uniform policy for state.")
                self.Ps[state] = self.Ps[state] + valid_moves 
                self.Ps[state] /= np.sum(self.Ps[state])

            self.Vs[state] = valid_moves
            self.Ns[state] = 0 

            self.total_depth += depth  # Leaf node reached
            return -predicted_value

        valid_moves = self.Vs[state]
        current_best_uct_score = -float('inf')
        action = -1

        for act in range(self.action_size):
            if valid_moves[act]: 
                c_puct = self.get_dynamic_cpuct(state)
                if (state, act) in self.Qsa:
                    uct_score = self.Qsa[(state, act)] + c_puct * self.Ps[state][act] * math.sqrt(self.Ns[state]) / (1 + self.Nsa[(state, act)])
                else:
                    uct_score = c_puct * self.Ps[state][act] * math.sqrt(self.Ns[state] + EPS)

                if uct_score > current_best_uct_score:
                    current_best_uct_score = uct_score
                    action = act

        # Simulate the game step for the chosen action
        next_state, next_player = BU.get_next_state(canonical_board, 1, action)
        next_state = BU.get_canonical_form(next_state, next_player)
        value = self.search(next_state, depth + 1)

        if (state, action) in self.Qsa:
            self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + value) / (self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1
        else:
            self.Qsa[(state, action)] = value
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1

        return -value