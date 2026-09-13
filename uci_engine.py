"""
uci_engine.py — UCI-движок для lichess-bot поверх DualHeadResNet + MCTS
(см. chess_engine.py / train_dual.py / play_gradio.py).

ПОЧЕМУ ПЕРЕПИСАНО:
  Старая версия этого файла содержала СВОЮ архитектуру ChessResNet
  (value-only, 18 входных плоскостей, PVS/quiescence/alpha-beta поиск).
  Чекпоинт chess_dualhead_best.pth, который реально лежит на диске,
  обучался в train_dual.py под ДРУГУЮ архитектуру — DualHeadResNet из
  chess_engine.py: policy+value головы, вход 54 плоскости (18 текущих +
  3 истории * 12), поиск MCTS/PUCT, а не alpha-beta. Отсюда и падение:
  "Ключей в чекпоинте: 280, в модели: 272" + size mismatch на
  conv_input.weight (54 канала в чекпоинте против 18 в ChessResNet).

  Вместо попытки "подогнать" веса под чужую архитектуру — движок теперь
  напрямую использует chess_engine.py (общий модуль модели + MCTS),
  как это уже делает play_gradio.py. Так гарантированно нет рассинхрона
  архитектур между обучением/Gradio-игрой/lichess-ботом — все три
  используют один и тот же код модели.

ЧТО СОХРАНЕНО из предыдущей версии uci_engine.py (устойчивость):
  • Обёртка тела UCI-цикла в try/except — единичная битая команда от GUI
    не убивает процесс движка посреди партии/турнира.
  • board.push_uci(m) в "position" оборачивается в try/except — рассинхрон
    позиции с GUI не должен ронять процесс.
  • Device выбирается автоматически (mps > cuda > cpu), с возможностью
    явного переопределения через переменную окружения
    CHESS_ENGINE_DEVICE=cpu|mps|cuda — для диагностики device-специфичных
    багов без правки кода.
  • Путь к весам берётся из CHESS_MODEL_PATH (env), несовпадение ключей
    state_dict ловится отдельно с понятной диагностикой (какие ключи
    лишние/каких не хватает) вместо сырого traceback.
  • sys.stdout.reconfigure(line_buffering=True) — иначе lichess-bot
    зависает, не дождавшись вывода.
  • CLI-режимы для диагностики без lichess-bot:
        python uci_engine.py bench            — замер скорости (sims/sec)
        python uci_engine.py eval "<FEN>"      — быстрая оценка позиции
        python uci_engine.py                   — обычный UCI-цикл

ЧТО НОВОЕ / ОТЛИЧАЕТСЯ ОТ ChessResNet-версии:
  • Поиск — MCTS (PUCT + батч через virtual loss) вместо PVS/alpha-beta.
    Понятие "depth" в классическом смысле здесь не применимо; "go depth N"
    интерпретируется как число симуляций, кратное N (см. compute_n_simulations).
  • Модели для оценки позиции нужна ИСТОРИЯ последних ходов (history_len,
    берётся из конфига чекпоинта). Движок ведёт список FEN-ов партии
    (до каждого хода) с самого starpos/fen из команды "position", чтобы
    на каждый вызов MCTS передавать корректную историю — как это делает
    play_gradio.py (history_for_model).
  • "info" в стандартный поток мы не шлём (кроме bestmove) — все логи и
    диагностика уходят в stderr, как и раньше, чтобы не портить UCI-канал.
"""

import os
import sys
import time
from typing import List, Optional

import chess
import torch

from chess_engine import MCTS, evaluate_single, load_model

# ============================== НАСТРОЙКИ ДВИЖКА ==============================
MODEL_PATH = os.environ.get(
    "CHESS_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_dualhead_best.pth"),
)

# CHESS_ENGINE_DEVICE=cpu|mps|cuda|auto — для диагностики device-специфичных
# багов (например, если качество игры отличается на MPS и CPU) без правки кода.
_DEVICE_OVERRIDE = os.environ.get("CHESS_ENGINE_DEVICE", "auto").strip().lower()

if _DEVICE_OVERRIDE in ("cpu", "mps", "cuda"):
    DEVICE = _DEVICE_OVERRIDE
elif torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# --- MCTS / сила движка --------------------------------------------------
N_SIMULATIONS_DEFAULT = int(os.environ.get("CHESS_N_SIMULATIONS", "400"))
MCTS_BATCH_SIZE = int(os.environ.get("CHESS_MCTS_BATCH_SIZE", "16"))
C_PUCT = float(os.environ.get("CHESS_C_PUCT", "1.5"))
TEMPERATURE = float(os.environ.get("CHESS_TEMPERATURE", "0.0"))  # 0 = всегда самый посещённый ход

# --- управление временем ---------------------------------------------------
ENGINE_TIME_LIMIT_SEC = 3.0     # дефолт, если GUI не прислал ни movetime, ни wtime/btime
MOVE_OVERHEAD_MS = 300          # запас на сетевую задержку lichess + накладные расходы движка
MIN_TIME_LIMIT_SEC = 0.1

# Грубая оценка скорости для перевода "время -> число симуляций", пока не
# откалибровано под конкретное железо через bench (см. run_bench). Обновится
# автоматически по факту первых ходов партии (см. _EngineState.sims_per_sec).
DEFAULT_SIMS_PER_SEC = 60.0
MIN_SIMULATIONS = 32
MAX_SIMULATIONS = 3000


# ==============================================================================
# СОСТОЯНИЕ ДВИЖКА: модель + история партии для построения входа с историей
# ==============================================================================

class EngineState:
    """Держит загруженную модель и FEN-историю текущей партии — она нужна
    модели (in_planes = 18 + history_len*12), а не только последней позиции."""

    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device)
        print(f"ℹ️ Загрузка модели из {model_path} (device={self.device}) ...", file=sys.stderr)
        try:
            self.model, self.history_len = load_model(model_path, self.device)
        except RuntimeError as e:
            # Несовпадение архитектуры/размеров — печатаем понятную
            # диагностику вместо сырого traceback внутри lichess-bot.
            print("❌ Веса не подходят под архитектуру DualHeadResNet.", file=sys.stderr)
            print(f"   Ошибка загрузки: {e}", file=sys.stderr)
            print("   Проверьте, что CHESS_MODEL_PATH указывает на чекпоинт, "
                  "обученный train_dual.py (DualHeadResNet), а не на веса "
                  "от другой архитектуры.", file=sys.stderr)
            raise
        except FileNotFoundError:
            print(f"❌ Файл весов не найден: {model_path}. "
                  f"Задайте CHESS_MODEL_PATH.", file=sys.stderr)
            raise

        self.model.eval()
        print(f"✅ Модель загружена: history_len={self.history_len}, device={self.device}",
              file=sys.stderr)

        # FEN-и всех позиций С НАЧАЛА ПАРТИИ, в хронологическом порядке
        # (FEN ДО каждого хода) — та же конвенция, что в play_gradio.py.
        self.all_fens: List[str] = []
        self.board = chess.Board()

        # Онлайн-калибровка скорости MCTS (симуляций/сек), чтобы точнее
        # переводить бюджет времени UCI в число симуляций.
        self.sims_per_sec = DEFAULT_SIMS_PER_SEC

    def reset(self):
        self.all_fens.clear()
        self.board = chess.Board()

    def set_position(self, board: chess.Board, played_fens_in_order: List[str]):
        """played_fens_in_order — FEN-и ДО каждого сыгранного хода, в порядке
        партии (то, что мы сами накопили при разборе команды 'position')."""
        self.board = board
        self.all_fens = played_fens_in_order

    def history_for_model(self) -> List[str]:
        """Последние history_len FEN-ов, самая свежая позиция первой —
        см. history_for_model в play_gradio.py."""
        if not self.all_fens:
            return []
        return list(reversed(self.all_fens[-self.history_len:]))


# ==============================================================================
# УПРАВЛЕНИЕ ВРЕМЕНЕМ -> ЧИСЛО СИМУЛЯЦИЙ MCTS
# ==============================================================================

def compute_time_budget_sec(tokens: List[str], board: chess.Board) -> Optional[float]:
    """Возвращает бюджет времени в секундах, либо None если бюджет не
    времени-ориентированный (например, 'go depth N' без времени — тогда
    используем compute_n_simulations_from_depth напрямую)."""
    overhead_s = MOVE_OVERHEAD_MS / 1000.0

    if "movetime" in tokens:
        idx = tokens.index("movetime")
        requested = int(tokens[idx + 1]) / 1000.0
        return max(MIN_TIME_LIMIT_SEC, requested - overhead_s)

    if "infinite" in tokens:
        # У MCTS нет честного "прервать на лету между симуляциями батча"
        # в этой реализации — ограничиваемся щедрым, но конечным бюджетом,
        # чтобы процесс не завис насмерть в турнире.
        return 60.0

    if "wtime" in tokens and "btime" in tokens:
        my_time_ms = int(tokens[tokens.index("wtime") + 1]) if board.turn == chess.WHITE \
            else int(tokens[tokens.index("btime") + 1])
        inc_ms = 0
        if board.turn == chess.WHITE and "winc" in tokens:
            inc_ms = int(tokens[tokens.index("winc") + 1])
        elif board.turn == chess.BLACK and "binc" in tokens:
            inc_ms = int(tokens[tokens.index("binc") + 1])

        my_time_s = my_time_ms / 1000.0
        raw = my_time_s * 0.05 + (inc_ms / 1000.0) * 0.8
        capped = min(raw, max(0.0, my_time_s - overhead_s))
        return max(MIN_TIME_LIMIT_SEC, capped)

    return None


def compute_n_simulations(tokens: List[str], board: chess.Board, state: "EngineState") -> int:
    """Переводит параметры команды 'go' в число MCTS-симуляций.

    Приоритет:
      1) явный 'go nodes N' -> N симуляций напрямую (nodes ~ simulations для MCTS)
      2) явный 'go depth N' -> N трактуется как грубый множитель силы
         (depth * 200 симуляций) — в MCTS нет классической "глубины",
         но lichess-bot/операторы иногда шлют 'go depth N' для тестов.
      3) время (movetime/wtime+btime/infinite) -> время * sims_per_sec
      4) фоллбек: дефолтное число симуляций (N_SIMULATIONS_DEFAULT)
    """
    if "nodes" in tokens:
        try:
            return max(MIN_SIMULATIONS, min(MAX_SIMULATIONS, int(tokens[tokens.index("nodes") + 1])))
        except (ValueError, IndexError):
            pass

    if "depth" in tokens:
        try:
            d = int(tokens[tokens.index("depth") + 1])
            return max(MIN_SIMULATIONS, min(MAX_SIMULATIONS, d * 200))
        except (ValueError, IndexError):
            pass

    time_budget = compute_time_budget_sec(tokens, board)
    if time_budget is not None:
        est = int(time_budget * state.sims_per_sec)
        return max(MIN_SIMULATIONS, min(MAX_SIMULATIONS, est))

    return N_SIMULATIONS_DEFAULT


# ==============================================================================
# ОДИН ХОД ДВИЖКА
# ==============================================================================

def engine_search_move(state: EngineState, tokens: List[str]) -> Optional[chess.Move]:
    board = state.board
    if board.is_game_over(claim_draw=True) or not list(board.legal_moves):
        return None

    n_sim = compute_n_simulations(tokens, board, state)
    history = state.history_for_model()

    mcts = MCTS(
        model=state.model,
        device=state.device,
        history_len=state.history_len,
        c_puct=C_PUCT,
        n_simulations=n_sim,
        batch_size=MCTS_BATCH_SIZE,
        add_root_noise=False,
    )

    t0 = time.time()
    move, info = mcts.select_move(board, history, temperature=TEMPERATURE)
    elapsed = time.time() - t0

    if elapsed > 0:
        # Скользящее среднее — калибруем sims_per_sec по факту игры, чтобы
        # следующий ход точнее укладывался в бюджет времени.
        observed = n_sim / elapsed
        state.sims_per_sec = 0.7 * state.sims_per_sec + 0.3 * observed

    top_str = " | ".join(f"{m.uci()}:{v}" for m, v, _ in info[:5]) if info else "—"
    print(f"info depth 1 nodes {n_sim} time {int(elapsed * 1000)} "
          f"bestmove {move.uci() if move else None} pv-candidates [{top_str}] "
          f"sims_per_sec {state.sims_per_sec:.1f}", file=sys.stderr)

    return move


# ==============================================================================
# UCI LOOP
# ==============================================================================

def _rebuild_all_fens(board_after_moves: chess.Board, base_fen: str, moves_uci: List[str]) -> List[str]:
    """Проигрывает партию с нуля из base_fen, накапливая FEN ДО каждого хода —
    та же конвенция, что и all_fens в play_gradio.py."""
    all_fens: List[str] = []
    b = chess.Board(base_fen)
    for m in moves_uci:
        try:
            move = chess.Move.from_uci(m)
            if move not in b.legal_moves:
                print(f"info string bad move '{m}' ignored (illegal)", file=sys.stderr)
                continue
            all_fens.append(b.fen())
            b.push(move)
        except ValueError as e:
            print(f"info string bad move '{m}' ignored: {e}", file=sys.stderr)
    board_after_moves.set_fen(b.fen())
    return all_fens


def uci_loop():
    # Критически важно: отключаем буферизацию вывода, иначе lichess-bot зависнет.
    sys.stdout.reconfigure(line_buffering=True)

    state = EngineState(MODEL_PATH, DEVICE)

    print(f"info string device={DEVICE} history_len={state.history_len} "
          f"mcts_batch_size={MCTS_BATCH_SIZE} c_puct={C_PUCT} model={MODEL_PATH}",
          file=sys.stderr)

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break

        if not line:
            break

        line = line.strip()
        if not line:
            continue

        # KEY FIX (сохранено из предыдущей версии): любая ошибка внутри
        # обработки ОДНОЙ команды не должна убивать процесс движка — иначе
        # он выпадает из турнира навсегда.
        try:
            tokens = line.split()
            command = tokens[0]

            if command == "uci":
                print("id name DualHeadResNet Chess Bot (MCTS/PUCT)")
                print("id author Vadym")
                print("uciok")

            elif command == "isready":
                print("readyok")

            elif command == "ucinewgame":
                state.reset()

            elif command == "position":
                board = chess.Board()
                if "startpos" in tokens:
                    base_fen = chess.STARTING_FEN
                    if "moves" in tokens:
                        moves_idx = tokens.index("moves")
                        moves_uci = tokens[moves_idx + 1:]
                    else:
                        moves_uci = []
                elif "fen" in tokens:
                    fen_idx = tokens.index("fen")
                    if "moves" in tokens:
                        moves_idx = tokens.index("moves")
                        base_fen = " ".join(tokens[fen_idx + 1:moves_idx])
                        moves_uci = tokens[moves_idx + 1:]
                    else:
                        base_fen = " ".join(tokens[fen_idx + 1:])
                        moves_uci = []
                else:
                    base_fen = chess.STARTING_FEN
                    moves_uci = []

                try:
                    all_fens = _rebuild_all_fens(board, base_fen, moves_uci)
                    state.set_position(board, all_fens)
                except ValueError as e:
                    print(f"info string bad fen '{base_fen}' ignored: {e}", file=sys.stderr)

            elif command == "go":
                best_move = engine_search_move(state, tokens)

                if best_move:
                    print(f"bestmove {best_move.uci()}")
                else:
                    legal = list(state.board.legal_moves)
                    if legal:
                        print(f"bestmove {legal[0].uci()}")
                    else:
                        print("bestmove 0000")

            elif command == "stop":
                # MCTS-поиск синхронный (успевает завершиться до чтения
                # следующей команды), поэтому к моменту получения "stop"
                # bestmove уже отправлен — реального прерывания на лету
                # не реализовано (как и в предыдущей версии).
                pass

            elif command == "quit":
                break

            # Неизвестные команды (setoption, ponderhit, register и т.п.)
            # осознанно игнорируются — так безопаснее, чем падать.

        except Exception as e:
            print(f"info string error handling '{line}': {e}", file=sys.stderr)


# ==============================================================================
# CLI-РЕЖИМЫ ДЛЯ ДИАГНОСТИКИ (без lichess-bot)
# ==============================================================================

def run_bench(n_simulations: int = 400, batch_size: int = MCTS_BATCH_SIZE):
    """python uci_engine.py bench — замер sims/sec на стартовой позиции,
    чтобы откалибровать DEFAULT_SIMS_PER_SEC под конкретное железо."""
    state = EngineState(MODEL_PATH, DEVICE)
    board = chess.Board()
    mcts = MCTS(
        model=state.model, device=state.device, history_len=state.history_len,
        c_puct=C_PUCT, n_simulations=n_simulations, batch_size=batch_size,
        add_root_noise=False,
    )
    t0 = time.time()
    move, info = mcts.select_move(board, [], temperature=0.0)
    dt = time.time() - t0
    sims_per_sec = n_simulations / dt if dt > 0 else 0.0
    print(f"device={DEVICE} n_simulations={n_simulations} batch_size={batch_size} "
          f"elapsed={dt:.2f}s sims_per_sec={sims_per_sec:.1f} bestmove={move.uci() if move else None}")
    print(f"Подсказка: установите DEFAULT_SIMS_PER_SEC ~= {sims_per_sec:.0f} "
          f"(или переменную окружения) для точного расчёта времени на ход.")


def run_eval(fen: str):
    """python uci_engine.py eval "<FEN>" — быстро посмотреть value-оценку
    одной позиции (без истории — для истории используйте полноценную игру)."""
    state = EngineState(MODEL_PATH, DEVICE)
    board = chess.Board(fen)
    priors, v_white = evaluate_single(board, [], state.model, state.device, state.history_len)
    top = sorted(priors.items(), key=lambda kv: -kv[1])[:5]
    top_str = ", ".join(f"{m.uci()}:{p:.3f}" for m, p in top)
    print(f"device={DEVICE} fen='{fen}' value_white={v_white:+.3f} top_policy=[{top_str}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        n_sim = int(sys.argv[2]) if len(sys.argv) > 2 else 400
        run_bench(n_simulations=n_sim)
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        if len(sys.argv) < 3:
            print("Использование: python uci_engine.py eval \"<FEN>\"", file=sys.stderr)
            sys.exit(1)
        run_eval(sys.argv[2])
    else:
        uci_loop()