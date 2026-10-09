"""
Packing of network parameters into the flat weight file read by the C++ network.
"""

from __future__ import annotations

import struct

import numpy as np
import torch

from arch import args as model_args

MAGIC = b"OTHELNET"
VERSION = 2
ALIGNMENT = 64
NORM_EPS = 1e-5
FC_TYPES = {"fp32": 0, "fp16": 1}

SQUARES = 64
SEQ_CHANNELS = 16
RES_CHANNELS = 64
FC1_OUT = 1024
FC2_OUT = 512
ACTIONS = 65
HEAD_OUT = 80
GROUP_WIDTH = 16

# Header layout: magic, version, fc_type, eight architecture sizes, leaky slope,
# norm eps, tensor count, reserved, then one file offset per tensor.
HEADER_FORMAT = "<8sII8IffII"
TENSOR_COUNT = 28


def conv_weights(w: torch.Tensor, in_channels: int, out_channels: int, kernel: int) -> np.ndarray:
    """
    Reorders convolution weights to [tap][input channel][output channel].

    Parameters
    ----------
    w : torch.Tensor
        Weights of shape (out_channels, in_channels, kernel, kernel).
    in_channels : int
        Expected input channels.
    out_channels : int
        Expected output channels.
    kernel : int
        Expected kernel side.

    Returns
    -------
    numpy.ndarray
        Contiguous float32 array of shape (kernel * kernel, in_channels, out_channels).
    """
    if tuple(w.shape) != (out_channels, in_channels, kernel, kernel):
        raise ValueError(f"unexpected convolution weight shape {tuple(w.shape)}")
    packed = w.detach().numpy().astype(np.float32).transpose(2, 3, 1, 0)
    return np.ascontiguousarray(packed.reshape(kernel * kernel, in_channels, out_channels))


def pack_linear(w: np.ndarray, out_padded: int) -> np.ndarray:
    """
    Packs fully connected weights in tiles of 16 outputs.

    Rows are zero padded to `out_padded`. Each tile is stored as
    [input][output within tile], and the tiles follow one another.

    Parameters
    ----------
    w : numpy.ndarray
        Weights of shape (out_features, in_features).
    out_padded : int
        Output count after padding.

    Returns
    -------
    numpy.ndarray
        Flat float32 array of out_padded * in_features elements.
    """
    out_features, in_features = w.shape
    padded = np.zeros((out_padded, in_features), dtype=np.float32)
    padded[:out_features] = w
    groups = [
        np.ascontiguousarray(padded[start : start + GROUP_WIDTH].T).reshape(-1)
        for start in range(0, out_padded, GROUP_WIDTH)
    ]
    return np.concatenate(groups)


def vector(t: torch.Tensor, length: int) -> np.ndarray:
    """
    Converts a one-dimensional parameter to float32 after checking its length.

    Parameters
    ----------
    t : torch.Tensor
        Parameter to convert.
    length : int
        Expected length.

    Returns
    -------
    numpy.ndarray
        Float32 array.
    """
    if tuple(t.shape) != (length,):
        raise ValueError(f"unexpected parameter shape {tuple(t.shape)}")
    return t.detach().numpy().astype(np.float32)


def linear_weights(w: torch.Tensor, out_features: int, in_features: int) -> np.ndarray:
    """
    Converts a fully connected weight matrix to float32 after checking its shape.

    Parameters
    ----------
    w : torch.Tensor
        Weights of shape (out_features, in_features).
    out_features : int
        Expected output features.
    in_features : int
        Expected input features.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape (out_features, in_features).
    """
    if tuple(w.shape) != (out_features, in_features):
        raise ValueError(f"unexpected linear weight shape {tuple(w.shape)}")
    return w.detach().numpy().astype(np.float32)


def tensors_in_file_order(state_dict: dict, fc_type: str) -> list[np.ndarray]:
    """
    Builds the tensors of the weight file in file order.

    Parameters
    ----------
    state_dict : dict
        Parameters of the trained network.
    fc_type : str
        "fp32" or "fp16", the storage type of the two large fully connected weight matrices.

    Returns
    -------
    list
        Flat arrays, one per tensor of the file.
    """
    sd = state_dict
    fc_dtype = np.float16 if fc_type == "fp16" else np.float32

    # The first fully connected layer reads the flattened residual output. The
    # trained layer expects channel-major order, and the C++ network keeps
    # square-major (channels-last) order.
    fc1 = linear_weights(sd["fc_block1.0.weight"], FC1_OUT, SQUARES * RES_CHANNELS)
    fc1 = fc1.reshape(FC1_OUT, RES_CHANNELS, SQUARES).transpose(0, 2, 1).reshape(FC1_OUT, -1)
    fc2 = linear_weights(sd["fc_block2.0.weight"], FC2_OUT, FC1_OUT)

    policy_w = linear_weights(sd["fc_policy.weight"], ACTIONS, FC2_OUT)
    value_w = linear_weights(sd["fc_value.weight"], 1, FC2_OUT)
    head_w = np.concatenate([policy_w, value_w])
    head_b = np.zeros(HEAD_OUT, dtype=np.float32)
    head_b[:ACTIONS] = vector(sd["fc_policy.bias"], ACTIONS)
    head_b[ACTIONS] = vector(sd["fc_value.bias"], 1)[0]

    tensors = [
        conv_weights(sd["initial_conv.0.weight"], 1, SEQ_CHANNELS, 5),
        vector(sd["initial_conv.0.bias"], SEQ_CHANNELS),
        vector(sd["initial_conv.1.weight"], SEQ_CHANNELS),
        vector(sd["initial_conv.1.bias"], SEQ_CHANNELS),
        conv_weights(sd["shortcut.weight"], SEQ_CHANNELS, RES_CHANNELS, 1),
        vector(sd["shortcut.bias"], RES_CHANNELS),
        conv_weights(sd["residual_block.0.weight"], SEQ_CHANNELS, RES_CHANNELS, 3),
        vector(sd["residual_block.0.bias"], RES_CHANNELS),
        vector(sd["residual_block.1.weight"], RES_CHANNELS),
        vector(sd["residual_block.1.bias"], RES_CHANNELS),
        conv_weights(sd["residual_block.3.weight"], RES_CHANNELS, RES_CHANNELS, 3),
        vector(sd["residual_block.3.bias"], RES_CHANNELS),
        vector(sd["residual_block.4.weight"], RES_CHANNELS),
        vector(sd["residual_block.4.bias"], RES_CHANNELS),
        conv_weights(sd["residual_block.6.weight"], RES_CHANNELS, RES_CHANNELS, 3),
        vector(sd["residual_block.6.bias"], RES_CHANNELS),
        vector(sd["residual_block.7.weight"], RES_CHANNELS),
        vector(sd["residual_block.7.bias"], RES_CHANNELS),
        pack_linear(fc1, FC1_OUT).astype(fc_dtype),
        vector(sd["fc_block1.0.bias"], FC1_OUT),
        vector(sd["fc_block1.1.weight"], FC1_OUT),
        vector(sd["fc_block1.1.bias"], FC1_OUT),
        pack_linear(fc2, FC2_OUT).astype(fc_dtype),
        vector(sd["fc_block2.0.bias"], FC2_OUT),
        vector(sd["fc_block2.1.weight"], FC2_OUT),
        vector(sd["fc_block2.1.bias"], FC2_OUT),
        pack_linear(head_w, HEAD_OUT),
        head_b,
    ]
    return [np.ascontiguousarray(t).reshape(-1) for t in tensors]


def pack_state_dict(state_dict: dict, fc_type: str = "fp32") -> bytes:
    """
    Serializes network parameters to the weight file format.

    Parameters
    ----------
    state_dict : dict
        Parameters of the trained network.
    fc_type : str, optional
        "fp32" or "fp16", the storage type of the two large fully connected weight matrices.

    Returns
    -------
    bytes
        Contents of the weight file.
    """
    if fc_type not in FC_TYPES:
        raise ValueError(f"fc_type must be one of {sorted(FC_TYPES)}")
    tensors = tensors_in_file_order(state_dict, fc_type)
    if len(tensors) != TENSOR_COUNT:
        raise ValueError("unexpected tensor count")

    header_size = struct.calcsize(HEADER_FORMAT) + 8 * TENSOR_COUNT
    offset = (header_size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    offsets = []
    for t in tensors:
        offsets.append(offset)
        offset = (offset + t.nbytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        VERSION,
        FC_TYPES[fc_type],
        SQUARES,
        SEQ_CHANNELS,
        RES_CHANNELS,
        model_args.groups,
        FC1_OUT,
        FC2_OUT,
        ACTIONS,
        HEAD_OUT,
        model_args.leaky_relu_slope,
        NORM_EPS,
        TENSOR_COUNT,
        0,
    ) + struct.pack(f"<{TENSOR_COUNT}Q", *offsets)

    out = bytearray(offset)
    out[: len(header)] = header
    for start, t in zip(offsets, tensors):
        out[start : start + t.nbytes] = t.tobytes()
    return bytes(out)
