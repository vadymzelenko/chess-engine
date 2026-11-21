"""
chess_engine.py — neural chess engine: DualHeadResNet + batched MCTS (PUCT).

Shared by uci_engine.py (UCI / lichess-bot) and train_dual.py (training).

Input encoding (in_planes = 18 + history_len * 12, 54 for history_len = 3):
    0-5    white pawn, knight, bishop, rook, queen, king
    6-11   black pawn, knight, bishop, rook, queen, king
    12     side to move (all ones when White is to move)
    13-16  castling rights: White O-O, White O-O-O, Black O-O, Black O-O-O
    17     en passant target square
    18+    12 piece planes for each previous position, most recent first
           (zeros when the game history is shorter)
The board is not flipped: plane[r, f] is square chess.square(f, r).

Policy (4672 = 73 * 64, AlphaZero layout, index = plane * 64 + from_square):
    planes 0-55   queen-like moves: 8 directions * distances 1..7
                  (also covers queen promotions)
    planes 56-63  knight moves
    planes 64-72  underpromotions to knight, bishop, rook * file delta -1, 0, +1

Value head: tanh output in [-1, 1] from White's point of view.

History convention (same as uci_engine.py): a list of FENs of the positions
BEFORE the current one, most recent first.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_PLANES = 18
PIECE_PLANES = 12
POLICY_SIZE = 73 * 64
DEFAULT_HISTORY_LEN = 3

# ----------------------------------------------------------------------------
# Move encoding
# ----------------------------------------------------------------------------

# Queen-move directions as (file delta, rank delta): N, NE, E, SE, S, SW, W, NW.
_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def encode_move(move: chess.Move) -> int:
    """Index of a move in the 4672-dimensional policy vector."""
    from_sq, to_sq = move.from_square, move.to_square
    df = chess.square_file(to_sq) - chess.square_file(from_sq)
    dr = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = 64 + _UNDERPROMOTIONS.index(move.promotion) * 3 + (df + 1)
        return plane * 64 + from_sq

    if (df, dr) in _KNIGHT_DELTAS:
        plane = 56 + _KNIGHT_DELTAS.index((df, dr))
        return plane * 64 + from_sq

    distance = max(abs(df), abs(dr))
    direction = _DIRECTIONS.index((int(np.sign(df)), int(np.sign(dr))))
    plane = direction * 7 + (distance - 1)
    return plane * 64 + from_sq


# ----------------------------------------------------------------------------
# Board encoding
# ----------------------------------------------------------------------------

def _piece_planes(board: chess.Board, out: np.ndarray, offset: int) -> None:
    for square, piece in board.piece_map().items():
        idx = offset + (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        out[idx, chess.square_rank(square), chess.square_file(square)] = 1.0


def fen_to_planes(fen: str, history: Sequence[str] = (), history_len: int = DEFAULT_HISTORY_LEN) -> np.ndarray:
    """Encodes a position and its history into a (18 + 12 * history_len, 8, 8) array."""
    board = chess.Board(fen)
    planes = np.zeros((BASE_PLANES + PIECE_PLANES * history_len, 8, 8), dtype=np.float32)

    _piece_planes(board, planes, 0)
    if board.turn == chess.WHITE:
        planes[12].fill(1.0)
    planes[13].fill(float(board.has_kingside_castling_rights(chess.WHITE)))
    planes[14].fill(float(board.has_queenside_castling_rights(chess.WHITE)))
    planes[15].fill(float(board.has_kingside_castling_rights(chess.BLACK)))
    planes[16].fill(float(board.has_queenside_castling_rights(chess.BLACK)))
    if board.ep_square is not None:
        planes[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0

    for i, past_fen in enumerate(list(history)[:history_len]):
        _piece_planes(chess.Board(past_fen), planes, BASE_PLANES + i * PIECE_PLANES)
    return planes


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-excitation: re-weights channels using global context."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels: int, se_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.se(self.bn2(self.conv2(y)))
        return F.relu(x + y)


@dataclass
class ModelConfig:
    in_planes: int = BASE_PLANES + PIECE_PLANES * DEFAULT_HISTORY_LEN
    channels: int = 256
    num_blocks: int = 16
    history_len: int = DEFAULT_HISTORY_LEN
    se_hidden: int = 32
    policy_channels: int = 32
    value_channels: int = 32
    value_hidden: int = 256

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class DualHeadResNet(nn.Module):
    """Residual tower with a policy head (4672 logits) and a value head (tanh)."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.conv_input = nn.Conv2d(c.in_planes, c.channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(c.channels)
        self.blocks = nn.ModuleList(ResBlock(c.channels, c.se_hidden) for _ in range(c.num_blocks))

        self.policy_conv = nn.Conv2d(c.channels, c.policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(c.policy_channels)
        self.policy_fc = nn.Linear(c.policy_channels * 64, POLICY_SIZE)

        self.value_conv = nn.Conv2d(c.channels, c.value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(c.value_channels)
        self.value_fc1 = nn.Linear(c.value_channels * 64, c.value_hidden)
        self.value_fc2 = nn.Linear(c.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.blocks:
            x = block(x)
        p = F.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        policy = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        return policy, value.squeeze(1)


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------

def save_checkpoint(path: str, model: DualHeadResNet, **extra) -> None:
    torch.save({"model_state_dict": model.state_dict(), "config": model.config.to_dict(), **extra}, path)


def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _infer_config(state: Dict[str, torch.Tensor], saved: Optional[dict]) -> ModelConfig:
    """Builds a config from the checkpoint: saved values first, missing ones from tensor shapes."""
    cfg = ModelConfig()
    tensors = list(state.values())
    conv4d = [t for t in tensors if t.dim() == 4]
    linear = [t for t in tensors if t.dim() == 2]
    if conv4d:
        cfg.channels, cfg.in_planes = conv4d[0].shape[0], conv4d[0].shape[1]
        cfg.num_blocks = max(0, (len(conv4d) - 3) // 2)
    if len(linear) >= 3:
        cfg.se_hidden = linear[0].shape[0] if cfg.num_blocks else cfg.se_hidden
        policy_fc = next((t for t in linear if t.shape[0] == POLICY_SIZE), None)
        if policy_fc is not None:
            cfg.policy_channels = policy_fc.shape[1] // 64
        cfg.value_hidden, value_in = linear[-2].shape
        cfg.value_channels = value_in // 64
    cfg.history_len = max(0, (cfg.in_planes - BASE_PLANES) // PIECE_PLANES)
    for key, value in (saved or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, int(value))
    return cfg


def load_model(path: str, device="cpu") -> Tuple[DualHeadResNet, int]:
    """Loads a checkpoint and returns (model in eval mode, history_len).

    Checkpoints saved by save_checkpoint load by parameter name. Checkpoints from an
    earlier version of the training script, with different parameter names, are
    loaded by position when every tensor shape matches.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(ckpt)
    state = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v for k, v in state.items()}
    saved_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    model = DualHeadResNet(_infer_config(state, saved_cfg))

    own = model.state_dict()
    if set(own) == set(state):
        model.load_state_dict(state)
    else:
        own_items, ckpt_items = list(own.items()), list(state.items())
        mismatch = [
            (a[0], b[0], tuple(a[1].shape), tuple(b[1].shape))
            for a, b in zip(own_items, ckpt_items)
            if a[1].shape != b[1].shape
        ]
        if len(own_items) != len(ckpt_items) or mismatch:
            detail = mismatch[0] if mismatch else (len(own_items), len(ckpt_items))
            raise RuntimeError(
                f"Checkpoint does not match DualHeadResNet "
                f"(model tensors: {len(own_items)}, checkpoint: {len(ckpt_items)}; first mismatch: {detail})"
            )
        model.load_state_dict({name: tensor for (name, _), (_, tensor) in zip(own_items, ckpt_items)})

    model.to(device).eval()
    return model, model.config.history_len


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def _to_move_value(value_white: float, board: chess.Board) -> float:
    return value_white if board.turn == chess.WHITE else -value_white


@torch.no_grad()
def evaluate_batch(
    boards: Sequence[chess.Board],
    histories: Sequence[Sequence[str]],
    model: DualHeadResNet,
    device,
    history_len: int,
) -> List[Tuple[Dict[chess.Move, float], float]]:
    """Runs the network on several positions. Returns (legal move priors, value for White) per position."""
    x = np.stack([fen_to_planes(b.fen(), h, history_len) for b, h in zip(boards, histories)])
    logits, values = model(torch.from_numpy(x).to(device))
    logits = logits.float().cpu()
    values = values.float().cpu().tolist()

    results = []
    for i, board in enumerate(boards):
        moves = list(board.legal_moves)
        if not moves:
            results.append(({}, values[i]))
            continue
        idx = torch.tensor([encode_move(m) for m in moves], dtype=torch.long)
        probs = torch.softmax(logits[i, idx], dim=0).tolist()
        results.append((dict(zip(moves, probs)), values[i]))
    return results


def evaluate_single(board: chess.Board, history: Sequence[str], model: DualHeadResNet, device, history_len: int):
    """Priors over legal moves and the value from White's point of view for one position."""
    return evaluate_batch([board], [history], model, device, history_len)[0]


# ----------------------------------------------------------------------------
# MCTS
# ----------------------------------------------------------------------------

@dataclass
class Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0  # from the point of view of the player who made the move into this node
    virtual_loss: int = 0
    children: Dict[chess.Move, "Node"] = field(default_factory=dict)
    expanded: bool = False

    def q(self) -> float:
        n = self.visits + self.virtual_loss
        return (self.value_sum - self.virtual_loss) / n if n else 0.0


class MCTS:
    """Monte Carlo tree search with PUCT selection.

    Leaves are collected in batches of `batch_size` with virtual loss, so the network
    evaluates several positions per forward pass.
    """

    def __init__(
        self,
        model: DualHeadResNet,
        device,
        history_len: int = DEFAULT_HISTORY_LEN,
        c_puct: float = 1.5,
        n_simulations: int = 400,
        batch_size: int = 16,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        fpu_reduction: float = 0.2,
    ):
        self.model = model
        self.device = device
        self.history_len = history_len
        self.c_puct = c_puct
        self.n_simulations = max(1, int(n_simulations))
        self.batch_size = max(1, int(batch_size))
        self.add_root_noise = add_root_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.fpu_reduction = fpu_reduction

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _terminal_value(board: chess.Board) -> Optional[float]:
        """Value for the side to move if the game is over, otherwise None."""
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    def _select_child(self, node: Node) -> Tuple[chess.Move, Node]:
        total = node.visits + node.virtual_loss
        sqrt_total = math.sqrt(max(total, 1))
        # First-play urgency: unvisited moves start slightly below the parent's value.
        visited_prior = sum(c.prior for c in node.children.values() if c.visits)
        fpu = -node.q() - self.fpu_reduction * math.sqrt(visited_prior)
        best, best_score = None, -float("inf")
        for move, child in node.children.items():
            n = child.visits + child.virtual_loss
            q = child.q() if n else fpu
            score = q + self.c_puct * child.prior * sqrt_total / (1 + n)
            if score > best_score:
                best, best_score = (move, child), score
        return best

    def _expand(self, node: Node, priors: Dict[chess.Move, float]) -> None:
        node.children = {m: Node(prior=p) for m, p in priors.items()}
        node.expanded = True

    def _add_noise(self, node: Node) -> None:
        moves = list(node.children)
        if not moves:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for move, n in zip(moves, noise):
            child = node.children[move]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(n)

    @staticmethod
    def _backup(path: List[Node], value_to_move: float, virtual: bool) -> None:
        """value_to_move is the leaf value for the side to move at the leaf."""
        value = -value_to_move
        for node in reversed(path[1:]):
            if virtual:
                node.virtual_loss -= 1
            node.visits += 1
            node.value_sum += value
            value = -value
        if virtual:
            path[0].virtual_loss -= 1
        path[0].visits += 1

    # -- search ----------------------------------------------------------------

    def search(self, board: chess.Board, history: Sequence[str]) -> Node:
        root = Node(prior=1.0)
        history = list(history)
        if self._terminal_value(board) is not None:
            return root

        priors, _ = evaluate_single(board, history, self.model, self.device, self.history_len)
        self._expand(root, priors)
        if self.add_root_noise:
            self._add_noise(root)

        done = 0
        while done < self.n_simulations:
            pending = []  # (path, board, history)
            terminal = []  # (path, value)
            for _ in range(min(self.batch_size, self.n_simulations - done)):
                node, sim, hist, path = root, board.copy(), history, [root]
                root.virtual_loss += 1
                while node.expanded and node.children:
                    move, child = self._select_child(node)
                    hist = [sim.fen()] + hist[: self.history_len - 1]
                    sim.push(move)
                    child.virtual_loss += 1
                    path.append(child)
                    node = child
                done += 1
                value = self._terminal_value(sim)
                if value is not None:
                    terminal.append((path, value))
                elif any(p[-1] is node for p, _, _ in pending):
                    # The same leaf was picked twice in this batch: undo and stop collecting.
                    for n in path:
                        n.virtual_loss -= 1
                    done -= 1
                    break
                else:
                    pending.append((path, sim, hist))

            for path, value in terminal:
                self._backup(path, value, virtual=True)
            if pending:
                results = evaluate_batch(
                    [b for _, b, _ in pending], [h for _, _, h in pending],
                    self.model, self.device, self.history_len,
                )
                for (path, sim, _), (priors, value_white) in zip(pending, results):
                    self._expand(path[-1], priors)
                    self._backup(path, _to_move_value(value_white, sim), virtual=True)
            if not pending and not terminal:
                break
        return root

    def select_move(
        self, board: chess.Board, history: Sequence[str], temperature: float = 0.0
    ) -> Tuple[Optional[chess.Move], List[Tuple[chess.Move, int, float]]]:
        """Runs the search and returns (move, [(move, visits, q), ...] sorted by visits)."""
        root = self.search(board, history)
        if not root.children:
            return None, []
        info = sorted(
            ((m, c.visits, c.q()) for m, c in root.children.items()),
            key=lambda item: -item[1],
        )
        if temperature <= 1e-6:
            return info[0][0], info
        moves = [m for m, _, _ in info]
        weights = [max(v, 1e-9) ** (1.0 / temperature) for _, v, _ in info]
        return random.choices(moves, weights=weights, k=1)[0], info


__all__ = [
    "DualHeadResNet",
    "ModelConfig",
    "MCTS",
    "encode_move",
    "evaluate_batch",
    "evaluate_single",
    "fen_to_planes",
    "load_model",
    "save_checkpoint",
    "POLICY_SIZE",
]
