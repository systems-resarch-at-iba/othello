"""
Exports a trained checkpoint to the flat weight file read by the C++ network.

Usage:
    python scripts/export_weights.py models/model-alpha3S-v2.1.pth.tar weights.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
# engine/ and arch.py are top-level modules under backend/. This script lives in
# scripts/, where sys.path[0] is scripts/, so backend/ is added to sys.path explicitly.
sys.path.insert(0, str(BACKEND_DIR))

from engine.weight_export import (  # noqa: E402,F401
    ACTIONS,
    FC1_OUT,
    FC2_OUT,
    FC_TYPES,
    HEAD_OUT,
    RES_CHANNELS,
    SEQ_CHANNELS,
    SQUARES,
    pack_state_dict,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--fc-type",
        choices=sorted(FC_TYPES),
        default="fp32",
        help="storage type of the two large fully connected weight matrices",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = parse_args(argv)
    checkpoint = torch.load(ns.checkpoint, map_location="cpu")
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_bytes(pack_state_dict(checkpoint["state_dict"], ns.fc_type))


if __name__ == "__main__":
    main()
