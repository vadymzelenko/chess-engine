"""Tests for chess_engine.py: encodings, model, checkpoints and MCTS.

Run: python -m pytest tests/  (or python tests/test_chess_engine.py)
"""
import os
import sys
import tempfile

import chess
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chess_engine import (  # noqa: E402
    MCTS, POLICY_SIZE, DualHeadResNet, ModelConfig, encode_move,
    evaluate_single, fen_to_planes, load_model, save_checkpoint,
)

SMALL = ModelConfig(channels=16, num_blocks=2, se_hidden=4, policy_channels=4, value_channels=2, value_hidden=16)


def _random_positions(n=300, seed=0):
    import random
    rnd = random.Random(seed)
    boards = []
    for _ in range(n // 30):
        b = chess.Board()
        for _ in range(60):
            moves = list(b.legal_moves)
            if not moves:
                break
            boards.append(b.copy())
            b.push(rnd.choice(moves))
    return boards


def test_move_encoding_is_unique_and_in_range():
    for board in _random_positions():
        idx = [encode_move(m) for m in board.legal_moves]
        assert len(idx) == len(set(idx))
        assert all(0 <= i < POLICY_SIZE for i in idx)


def test_promotions_encode_distinctly():
    board = chess.Board("8/P6k/8/8/8/8/6Kp/8 w - - 0 1")
    idx = {encode_move(m) for m in board.legal_moves if m.promotion}
    assert len(idx) == 4
    board.turn = chess.BLACK
    idx = {encode_move(m) for m in board.legal_moves if m.promotion}
    assert len(idx) == 4


def test_planes_shape_and_content():
    board = chess.Board()
    hist = [chess.STARTING_FEN]
    board.push_san("e4")
    planes = fen_to_planes(board.fen(), hist, 3)
    assert planes.shape == (54, 8, 8)
    assert planes[:12].sum() == 32           # all pieces present
    assert planes[12].sum() == 0             # black to move
    assert planes[17].sum() == 0             # no capturable en passant pawn
    assert planes[18:30].sum() == 32         # previous position
    assert planes[30:].sum() == 0            # no older history


def test_en_passant_plane():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3")
    assert fen_to_planes(board.fen())[17, 2, 4] == 1


def test_forward_and_evaluate():
    model = DualHeadResNet(SMALL).eval()
    priors, value = evaluate_single(chess.Board(), [], model, "cpu", SMALL.history_len)
    assert len(priors) == 20
    assert abs(sum(priors.values()) - 1) < 1e-5
    assert -1 <= value <= 1


def test_checkpoint_roundtrip_and_positional_load():
    model = DualHeadResNet(SMALL).eval()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pth")
        save_checkpoint(path, model)
        loaded, hist = load_model(path, "cpu")
        assert hist == 3
        x = torch.randn(2, 54, 8, 8)
        assert torch.allclose(model(x)[0], loaded(x)[0])

        # Same tensors under different names (older training script): loaded by position.
        renamed = {f"p{i}": t for i, t in enumerate(model.state_dict().values())}
        path2 = os.path.join(d, "old.pth")
        torch.save({"model_state_dict": renamed}, path2)
        loaded2, _ = load_model(path2, "cpu")
        assert loaded2.config.channels == 16 and loaded2.config.num_blocks == 2
        assert torch.allclose(model(x)[1], loaded2(x)[1])


def test_mcts_returns_legal_move_and_finds_mate():
    model = DualHeadResNet(SMALL).eval()
    mcts = MCTS(model, "cpu", history_len=3, n_simulations=64, batch_size=8)
    move, info = mcts.select_move(chess.Board(), [])
    assert move in chess.Board().legal_moves
    assert sum(v for _, v, _ in info) >= 1

    # Mate in one (Qh5xf7#): terminal values must steer the search even with an untrained net.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    mcts = MCTS(model, "cpu", history_len=3, n_simulations=400, batch_size=8)
    move, _ = mcts.select_move(board, [])
    assert move == chess.Move.from_uci("h5f7")


def test_game_over_returns_none():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    mcts = MCTS(DualHeadResNet(SMALL).eval(), "cpu", n_simulations=8)
    assert mcts.select_move(board, []) == (None, [])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
