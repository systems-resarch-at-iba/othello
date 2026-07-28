import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import BOARD_SIZE
from utils import dotdict

# Model Architecture Parameters
args = dotdict({
    'dropout': 0.3,
    'cuda': torch.cuda.is_available(),
    'conv_channels_sequential': 16,
    'conv_channels_resnet': 64,
    'batch_size': 64,
    'groups': 8,  # Number of groups for GroupNorm
    'leaky_relu_slope': 0.1,  # Slope for LeakyReLU

    'num_mcts_sims' : 500,
    'c_puct': 1.5,
    "log_level": "INFO",
    'version': "alpha3S-v2.1",
    'model_name': "Alpha3S",
})

class CNN_vAlpha3(nn.Module):
    """
    Enhanced CNN architecture for Othello with residual connections and stacked convolutions.

    This model takes a 1-channel input representing the Othello board and outputs:
    - A policy head that predicts the log-probabilities of each possible move.
    - A value head that estimates the win probability (scalar in [-1, 1]).

    Architecture:
    - Initial Conv2d layer with a 5x5 kernel for broad context.
    - Three stacked 3x3 convolutional layers with GroupNorm and LeakyReLU.
    - A skip connection across the 3x3 conv block.
    - Two fully connected blocks with LayerNorm, LeakyReLU, and Dropout.
    - Final output heads: one for policy (logits), one for value (scalar).
    
    Parameters
    ----------
    args : dotdict
        Must contain:
            - conv_channels_sequential: int (number of channels in initial conv layer)
            - conv_channels_resnet : int (number of channels in residual block)
            - groups : int (groups for GroupNorm)
            - dropout : float (Dropout probability)
            - leaky_relu_slope : float (negative slope for LeakyReLU)
    """
    def __init__(self, args: dotdict) -> None:
        super().__init__()

        self.board_size = BOARD_SIZE
        self.action_size = BOARD_SIZE * BOARD_SIZE + 1
        self.args = args

        # Initial wide-field conv
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, args.conv_channels_sequential, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(args.groups, args.conv_channels_sequential),
            nn.LeakyReLU(negative_slope=args.leaky_relu_slope)
        )

        self.shortcut = nn.Conv2d(
            args.conv_channels_sequential,
            args.conv_channels_resnet,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Residual block: 3 stacked 3x3 convs
        self.residual_block = nn.Sequential(
            nn.Conv2d(args.conv_channels_sequential, args.conv_channels_resnet, kernel_size=3, padding=1),
            nn.GroupNorm(args.groups, args.conv_channels_resnet),
            nn.LeakyReLU(negative_slope=args.leaky_relu_slope),

            nn.Conv2d(args.conv_channels_resnet, args.conv_channels_resnet, kernel_size=3, padding=1),
            nn.GroupNorm(args.groups, args.conv_channels_resnet),
            nn.LeakyReLU(negative_slope=args.leaky_relu_slope),

            nn.Conv2d(args.conv_channels_resnet, args.conv_channels_resnet, kernel_size=3, padding=1),
            nn.GroupNorm(args.groups, args.conv_channels_resnet),
            nn.LeakyReLU(negative_slope=args.leaky_relu_slope)
        )

        # Conv output size (dynamically computed)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.board_size, self.board_size)
            dummy_out = self.initial_conv(dummy_input)
            residual = self.shortcut(dummy_out)
            dummy_out = self.residual_block(dummy_out) + residual
            flattened_size = dummy_out.view(1, -1).size(1)

        # Fully connected layers
        self.fc_block1 = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(negative_slope=args.leaky_relu_slope),
            nn.Dropout(p=args.dropout)
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(negative_slope=args.leaky_relu_slope),
            nn.Dropout(p=args.dropout)
        )

        # Output heads
        self.fc_policy = nn.Linear(512, self.action_size)
        self.fc_value = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch_size, BOARD_SIZE, BOARD_SIZE) or
            (batch_size, 1, BOARD_SIZE, BOARD_SIZE)

        Returns
        -------
        tuple
            - policy : torch.Tensor
                Log-softmax of action logits, shape (batch_size, action_size)
            - value : torch.Tensor
                Scalar win prediction in [-1, 1], shape (batch_size, 1)
        """
        x = x.view(-1, 1, self.board_size, self.board_size)  # ensure channel dim
        x = self.initial_conv(x)

        # Residual block with skip connection
        residual = self.shortcut(x)
        x = self.residual_block(x)
        x += residual

        # Flatten and pass through MLP
        x = x.view(x.size(0), -1)
        x = self.fc_block1(x)
        x = self.fc_block2(x)

        # Output heads
        policy = self.fc_policy(x)
        value = self.fc_value(x)

        return F.log_softmax(policy, dim=1), torch.tanh(value)
