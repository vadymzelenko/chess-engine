"""
Признаки позиции и вычисление eval по весам (tapered eval, PeSTO-style).

Идея: для каждого (тип фигуры, клетка) есть midgame-вес и endgame-вес.
Итоговая оценка = interpolate(mg_score, eg_score, phase).
Веса — это как раз то, что подбирает Texel tuning.

768 параметров = 6 типов фигур * 64 клетки * 2 фазы (mg/eg).
"""
import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
N_PIECES = 6
N_SQUARES = 64
N_PARAMS = N_PIECES * N_SQUARES * 2  # mg + eg -> 768

# Веса фаз для подсчёта game phase (стандартные значения, не тюнятся)
PHASE_WEIGHTS = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
MAX_PHASE = 24  # 4*1(N) + 4*1(B) + 4*2(R) + 2*4(Q) = 4+4+8+8=24


def mg_index(piece_type_idx: int, square: int) -> int:
    return piece_type_idx * N_SQUARES + square


def eg_index(piece_type_idx: int, square: int) -> int:
    return N_PIECES * N_SQUARES + piece_type_idx * N_SQUARES + square


def mirror_square(sq: int) -> int:
    # чёрные PST читаются как зеркало белых по горизонтали (rank flip)
    return chess.square_mirror(sq)


# pt.piece_type (1..6) -> наш индекс 0..5 в PIECE_TYPES (порядок совпадает с chess: P,N,B,R,Q,K)
_PT_TO_IDX = {pt: i for i, pt in enumerate(PIECE_TYPES)}
_MIRROR = [chess.square_mirror(sq) for sq in range(64)]  # предвычисленная таблица зеркала


def extract_features(board: chess.Board):
    """
    Возвращает (mg_idx, eg_idx, phase) где mg_idx/eg_idx — списки индексов параметров
    (белые +1, чёрные -1 закодированы отдельными списками add/sub для скорости).
    Быстрый путь через board.piece_map() (один вызов вместо 12).
    """
    add_mg, sub_mg, add_eg, sub_eg = [], [], [], []
    phase = 0

    for sq, piece in board.piece_map().items():
        pt_idx = _PT_TO_IDX[piece.piece_type]
        if piece.color:  # White
            use_sq = sq
            add_mg.append(pt_idx * 64 + use_sq)
            add_eg.append(pt_idx * 64 + use_sq)
        else:
            use_sq = _MIRROR[sq]
            sub_mg.append(pt_idx * 64 + use_sq)
            sub_eg.append(pt_idx * 64 + use_sq)
        w = PHASE_WEIGHTS.get(piece.piece_type)
        if w:
            phase += w

    phase = min(phase, MAX_PHASE)
    return (add_mg, sub_mg, add_eg, sub_eg), phase


def eval_from_features(feats, phase, weights: np.ndarray) -> float:
    add_mg, sub_mg, add_eg, sub_eg = feats
    eg_off = N_PIECES * N_SQUARES
    mg_score = sum(weights[i] for i in add_mg) - sum(weights[i] for i in sub_mg)
    eg_score = sum(weights[i + eg_off] for i in add_eg) - sum(weights[i + eg_off] for i in sub_eg)
    return (mg_score * phase + eg_score * (MAX_PHASE - phase)) / MAX_PHASE


def evaluate(board: chess.Board, weights: np.ndarray) -> int:
    """Оценка с точки зрения белых (centipawn-like), напрямую по доске."""
    feats, phase = extract_features(board)
    score = eval_from_features(feats, phase, weights)
    return int(round(score))


def default_weights() -> np.ndarray:
    """
    Стартовые веса = классические значения фигур (без позиционных бонусов).
    Texel tuning дальше научит PST-часть с нуля.
    Использовать как safe fallback, если ещё нет tuned weights.json.
    """
    w = np.zeros(N_PARAMS, dtype=np.float64)
    mg_vals = {chess.PAWN: 82, chess.KNIGHT: 337, chess.BISHOP: 365, chess.ROOK: 477, chess.QUEEN: 1025, chess.KING: 0}
    eg_vals = {chess.PAWN: 94, chess.KNIGHT: 281, chess.BISHOP: 297, chess.ROOK: 512, chess.QUEEN: 936, chess.KING: 0}
    for pt_idx, pt in enumerate(PIECE_TYPES):
        for sq in range(64):
            w[mg_index(pt_idx, sq)] = mg_vals[pt]
            w[eg_index(pt_idx, sq)] = eg_vals[pt]
    return w
