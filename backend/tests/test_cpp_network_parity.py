"""
Parity tests between the PyTorch network (arch.CNN_vAlpha3) and the C++
network (othello_cpp.Network). Skipped when the extension has not been built.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR / "cpp" / "build"))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
othello_cpp = pytest.importorskip("othello_cpp")

import export_weights  # noqa: E402
from arch import CNN_vAlpha3, args  # noqa: E402
from board import Utils as BU  # noqa: E402
from utils import dotdict  # noqa: E402

CHECKPOINT = BACKEND_DIR / "models" / f"model-{args.version}.pth.tar"
POSITION_COUNT = 150
LOGIT_TOLERANCE = 1e-4


def random_positions(count, seed):
    """
    Plays random games and returns canonical boards with their legal-move flags.

    Parameters
    ----------
    count : int
        Number of positions.
    seed : int
        Seed of the random games.

    Returns
    -------
    list
        Tuples of a float32 canonical board (own discs 1, opponent discs -1) and a uint8 array of 65 legal-move flags.
    """
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < count:
        board, player = BU.get_init_board(), -1
        while len(positions) < count and BU.get_game_ended(board, player) == 0:
            valid = BU.get_valid_moves(board, player)
            positions.append(((player * board).astype(np.float32), valid))
            board, player = BU.get_next_state(board, player, int(rng.choice(np.flatnonzero(valid))))
    return positions


def torch_forward(model, canonical):
    with torch.no_grad():
        log_policy, value = model(torch.from_numpy(canonical).view(1, 8, 8))
    return log_policy[0].numpy(), float(value[0, 0])


def perturbed_model(seed):
    """
    Builds a network whose parameters all differ from the defaults, so that a
    layout error in any parameter changes the output.
    """
    torch.manual_seed(seed)
    model = CNN_vAlpha3(dotdict({**args, "cuda": False})).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model


@pytest.fixture(scope="module")
def positions():
    return random_positions(POSITION_COUNT, seed=1)


@pytest.fixture(scope="module")
def model():
    return perturbed_model(seed=0)


@pytest.fixture(scope="module")
def weight_files(model, tmp_path_factory):
    directory = tmp_path_factory.mktemp("weights")
    paths = {}
    for fc_type in export_weights.FC_TYPES:
        paths[fc_type] = directory / f"{fc_type}.bin"
        paths[fc_type].write_bytes(export_weights.pack_state_dict(model.state_dict(), fc_type))
    return paths


@pytest.fixture(scope="module")
def network(weight_files):
    return othello_cpp.Network(str(weight_files["fp32"]))


def require_avx2(net):
    if net.isa != "avx2":
        pytest.skip("CPU without AVX2")


@pytest.mark.parametrize("reference", [True, False])
def test_fp32_matches_pytorch(model, network, positions, reference):
    if not reference:
        require_avx2(network)
    for canonical, _ in positions:
        expected_policy, expected_value = torch_forward(model, canonical)
        policy, value = network.forward(canonical, reference)
        assert np.allclose(policy, expected_policy, atol=LOGIT_TOLERANCE)
        assert abs(value - expected_value) < LOGIT_TOLERANCE


def test_avx2_matches_reference(network, positions):
    require_avx2(network)
    for canonical, _ in positions:
        policy, value = network.forward(canonical, False)
        reference_policy, reference_value = network.forward(canonical, True)
        assert np.allclose(policy, reference_policy, atol=LOGIT_TOLERANCE)
        assert abs(value - reference_value) < LOGIT_TOLERANCE


def test_fp16_avx2_matches_fp16_reference_and_stays_close_to_pytorch(model, weight_files, positions):
    net = othello_cpp.Network(str(weight_files["fp16"]))
    require_avx2(net)
    for canonical, _ in positions:
        policy, value = net.forward(canonical, False)
        reference_policy, reference_value = net.forward(canonical, True)
        assert np.allclose(policy, reference_policy, atol=LOGIT_TOLERANCE)
        assert abs(value - reference_value) < LOGIT_TOLERANCE
        expected_policy, expected_value = torch_forward(model, canonical)
        assert np.allclose(policy, expected_policy, atol=2e-2)
        assert abs(value - expected_value) < 2e-2


def test_masked_policy_matches_masked_exp_log_softmax(model, network, positions):
    for canonical, valid in positions:
        expected_log_policy, expected_value = torch_forward(model, canonical)
        expected = np.exp(expected_log_policy) * valid
        expected /= expected.sum()
        probs, value = network.masked_policy(canonical, valid)
        assert np.allclose(probs, expected, atol=LOGIT_TOLERANCE)
        assert abs(value - expected_value) < LOGIT_TOLERANCE


def test_missing_or_malformed_file_raises(weight_files, tmp_path):
    with pytest.raises(RuntimeError):
        othello_cpp.Network(str(tmp_path / "missing.bin"))
    data = weight_files["fp32"].read_bytes()
    bad_magic = tmp_path / "bad_magic.bin"
    bad_magic.write_bytes(b"XXXXXXXX" + data[8:])
    with pytest.raises(RuntimeError):
        othello_cpp.Network(str(bad_magic))
    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(RuntimeError):
        othello_cpp.Network(str(truncated))


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained checkpoint not available")
@pytest.mark.parametrize("fc_type,tolerance", [("fp32", LOGIT_TOLERANCE), ("fp16", 2e-2)])
def test_trained_checkpoint_matches_pytorch(fc_type, tolerance, tmp_path, positions):
    state_dict = torch.load(CHECKPOINT, map_location="cpu")["state_dict"]
    model = CNN_vAlpha3(dotdict({**args, "cuda": False})).eval()
    model.load_state_dict(state_dict)
    path = tmp_path / "weights.bin"
    path.write_bytes(export_weights.pack_state_dict(state_dict, fc_type))
    net = othello_cpp.Network(str(path))
    require_avx2(net)
    top1_matches = 0
    for canonical, valid in positions:
        expected_policy, expected_value = torch_forward(model, canonical)
        policy, value = net.forward(canonical, False)
        assert np.allclose(policy, expected_policy, atol=tolerance)
        assert abs(value - expected_value) < tolerance
        legal = np.flatnonzero(valid)
        top1_matches += legal[np.argmax(policy[legal])] == legal[np.argmax(expected_policy[legal])]
    assert top1_matches / len(positions) >= (1.0 if fc_type == "fp32" else 0.95)
