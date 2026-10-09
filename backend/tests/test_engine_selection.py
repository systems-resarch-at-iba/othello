"""
Tests for the startup choice between the C++ engine and the Python engine,
and for the log lines that report it.
"""
import logging

import pytest
from fastapi.testclient import TestClient

import main
from main import app


class _MissingExtension:
    def __init__(self, *args, **kwargs):
        raise ImportError("No module named 'othello_cpp'")


class _PythonEngine:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def restore_engine():
    saved = main.ai, main.CppAI, main.OthelloAI
    yield
    main.ai, main.CppAI, main.OthelloAI = saved


def test_missing_extension_falls_back_to_python_and_logs_it(restore_engine, caplog):
    main.CppAI = _MissingExtension
    main.OthelloAI = _PythonEngine
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with TestClient(app):
            assert isinstance(main.ai, _PythonEngine)
    messages = [record.getMessage() for record in caplog.records]
    assert any("C++ engine is unavailable" in m and "falling back to the Python engine" in m for m in messages)
    assert any(m.startswith("engine: Python") for m in messages)


def test_missing_checkpoint_logs_an_error(restore_engine, caplog):
    def no_checkpoint(*args, **kwargs):
        raise FileNotFoundError("No model found")

    main.CppAI = no_checkpoint
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        with TestClient(app):
            assert main.ai is None
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("model checkpoint was not found" in r.getMessage() for r in errors)


@pytest.mark.parametrize("isa,warns", [("avx2", False), ("scalar", True)])
def test_cpp_engine_log_names_the_instruction_set(isa, warns, caplog):
    class _Search:
        batch_size = 4

    class _Engine(main.CppAI):
        def __init__(self):
            self.isa = isa
            self.mcts = _Search()

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        main._log_engine(_Engine())
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("engine: C++") and f"instruction set {isa}" in m for m in messages)
    assert any("plain-loop network" in m for m in messages) == warns
    assert any(r.levelno == logging.WARNING for r in caplog.records) == warns
