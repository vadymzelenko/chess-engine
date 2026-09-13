"""
Движок: alpha-beta + quiescence search + transposition table + iterative deepening.
Eval подключается снаружи (features.evaluate с текущими весами).
"""
import time
import chess
import numpy as np
from features import evaluate, default_weights

MATE_SCORE = 1_000_000
INF = 10_000_000
MATE_THRESHOLD = MATE_SCORE - 1000  # для определения "это mate score"

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# MVV-LVA таблица для сортировки взятий (Most Valuable Victim - Least Valuable Attacker)
PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000, None: 0
}


class Engine:
    def __init__(self, weights: np.ndarray = None, tt_size_mb: int = 64):
        self.weights = weights if weights is not None else default_weights()
        self.tt = {}
        self.tt_max_entries = (tt_size_mb * 1024 * 1024) // 64  # грубая оценка размера записи
        self.nodes = 0
        self.start_time = 0
        self.time_limit = 5.0
        self.stop_search = False
        self.killers = {}  # {depth: [move1, move2]} — killer move heuristic

    # ---------- вспомогательное ----------

    def _time_up(self) -> bool:
        return (time.time() - self.start_time) > self.time_limit

    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            if victim is None and board.is_en_passant(move):
                victim = chess.PAWN
            attacker = board.piece_type_at(move.from_square)
            return 10_000 + PIECE_VALUE.get(victim, 0) * 10 - PIECE_VALUE.get(attacker, 0)
        return 0

    def _order_moves(self, board: chess.Board, moves, tt_move, depth):
        killer_list = self.killers.get(depth, [])

        def key(m):
            if tt_move is not None and m == tt_move:
                return 1_000_000
            if m in killer_list:
                return 5_000
            return self._mvv_lva_score(board, m)

        return sorted(moves, key=key, reverse=True)

    # ---------- mate-score <-> TT score conversion ----------
    # Mate score включает в себя ply ("расстояние до мата от текущего узла").
    # Если хранить его в TT как есть, при попадании в ту же позицию на ДРУГОЙ
    # глубине дерева (той же board-позиции соответствует разный ply) число будет
    # означать неверную дистанцию до мата. Стандартное решение: при записи в TT
    # переводим score в "дистанцию от корня хранения", при чтении — обратно
    # пересчитываем под текущий ply.
    @staticmethod
    def _score_to_tt(score: int, ply: int) -> int:
        if score >= MATE_THRESHOLD:
            return score + ply
        if score <= -MATE_THRESHOLD:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score: int, ply: int) -> int:
        if score >= MATE_THRESHOLD:
            return score - ply
        if score <= -MATE_THRESHOLD:
            return score + ply
        return score

    def _tt_store(self, key, depth, score, flag, move):
        entry = {"depth": depth, "score": score, "flag": flag, "move": move}
        if key in self.tt:
            self.tt[key] = entry
            return
        if len(self.tt) < self.tt_max_entries:
            self.tt[key] = entry
            return
        # Таблица заполнена: раньше запись просто отбрасывалась и TT "замерзала"
        # навсегда на глубоких итерациях iterative deepening. Вместо этого
        # вытесняем произвольную существующую запись (dict в Python 3.7+
        # сохраняет порядок вставки, так что это грубый FIFO), чтобы TT
        # продолжала обновляться на больших глубинах/долгих партиях.
        evict_key = next(iter(self.tt))
        del self.tt[evict_key]
        self.tt[key] = entry

    # ---------- quiescence search ----------

    def quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        self.nodes += 1
        stand_pat = evaluate(board, self.weights)
        if board.turn == chess.BLACK:
            stand_pat = -stand_pat

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        captures = self._order_moves(board, captures, None, -1)

        for move in captures:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

            if self.nodes % 2048 == 0 and self._time_up():
                self.stop_search = True
                return alpha
        return alpha

    # ---------- alpha-beta с TT ----------

    def alphabeta(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        alpha_orig = alpha

        key = board._transposition_key() if hasattr(board, "_transposition_key") else board.fen()
        tt_entry = self.tt.get(key)
        tt_move = None
        if tt_entry and tt_entry["depth"] >= depth:
            tt_move = tt_entry.get("move")
            tt_score = self._score_from_tt(tt_entry["score"], ply)
            if tt_entry["flag"] == TT_EXACT:
                return tt_score
            elif tt_entry["flag"] == TT_LOWER:
                alpha = max(alpha, tt_score)
            elif tt_entry["flag"] == TT_UPPER:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score
        elif tt_entry:
            tt_move = tt_entry.get("move")

        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return 0
        if ply > 0 and board.halfmove_clock >= 4 and board.is_repetition(3):
            return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return -MATE_SCORE + ply if board.is_check() else 0

        legal_moves = self._order_moves(board, legal_moves, tt_move, ply)

        best_score = -INF
        best_move = None

        for move in legal_moves:
            board.push(move)
            score = -self.alphabeta(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if self.stop_search:
                return best_score if best_move else 0

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

            if alpha >= beta:
                # killer move (не взятие, но вызвало отсечение)
                if not board.is_capture(move):
                    kl = self.killers.setdefault(ply, [])
                    if move not in kl:
                        kl.insert(0, move)
                        del kl[2:]
                break

            if self.nodes % 2048 == 0 and self._time_up():
                self.stop_search = True
                return best_score

        flag = TT_EXACT
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER

        self._tt_store(key, depth, self._score_to_tt(best_score, ply), flag, best_move)

        return best_score

    # ---------- iterative deepening (точка входа) ----------

    def search(self, board: chess.Board, time_limit: float = 5.0, max_depth: int = 64):
        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        self.stop_search = False
        self.killers = {}

        best_move = None
        best_score = 0

        for depth in range(1, max_depth + 1):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            legal_moves = self._order_moves(board, legal_moves, best_move, 0)

            alpha, beta = -INF, INF
            depth_best_move = None
            depth_best_score = -INF

            for move in legal_moves:
                board.push(move)
                score = -self.alphabeta(board, depth - 1, -beta, -alpha, 1)
                board.pop()

                if self.stop_search:
                    break

                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move
                alpha = max(alpha, score)

            if self.stop_search and depth_best_move is None:
                break

            if depth_best_move is not None:
                best_move = depth_best_move
                best_score = depth_best_score

            elapsed = time.time() - self.start_time
            nps = int(self.nodes / elapsed) if elapsed > 0 else 0
            print(f"info depth {depth} score cp {best_score} nodes {self.nodes} nps {nps} "
                  f"time {elapsed:.2f} pv {best_move}")

            if self.stop_search or self._time_up():
                break
            if abs(best_score) >= MATE_SCORE - 1000:
                break

        return best_move, best_score
