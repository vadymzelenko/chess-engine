import os
import io
import json
import time
import hashlib
import collections
from concurrent.futures import ProcessPoolExecutor

import zstandard as zstd

import chess.pgn

# --- НАСТРОЙКИ ---
INPUT_FILE = "texel_tuning_2-0/lichess-2400-eval.pgn.zst"
OUTPUT_FILE = "clean_positions.csv"
CHECKPOINT_FILE = "checkpoint.json"

# Формат CSV изменился: теперь 3 колонки FEN,Eval,BestMove вместо 2.
# BestMove — ход, реально сыгранный ИЗ данной позиции следующим полуходом
# (то есть тот же board state, что описан в FEN). Это weak label для
# policy-головы: шумный (игрок мог ошибиться), но осмысленный — в отличие
# от прежнего варианта, где policy-таргет вообще отсутствовал.
CSV_HEADER = "FEN,Eval,BestMove\n"

TARGET_POSITIONS = 1_000_000  # macbook air m4 +- 1min

# --- ФИЛЬТРЫ ---
MIN_FULLMOVE = 1
MIN_EVAL = -1000
MAX_EVAL = 1000
SKIP_CHECKS = False
SKIP_HALFMOVE_ZERO = False

# --- СТРАТИФИКАЦИЯ ---
BUCKET_WEIGHTS = {
    "opening_close":       0.10,
    "opening_edge":        0.05,
    "opening_decisive":    0.03,
    "middlegame_close":    0.20,
    "middlegame_edge":     0.15,
    "middlegame_decisive": 0.10,
    "endgame_close":       0.15,
    "endgame_edge":        0.12,
    "endgame_decisive":    0.10,
}
assert abs(sum(BUCKET_WEIGHTS.values()) - 1.0) < 1e-9, "Веса бакетов должны суммироваться в 1.0"

OPENING_MAX_FULLMOVE = 10
MIDDLEGAME_MAX_FULLMOVE = 30
CLOSE_MAX_ABS_EVAL = 150
EDGE_MAX_ABS_EVAL = 400

GAMES_PER_CHUNK = 200

NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
IN_FLIGHT_MULTIPLIER = 2

SAVE_INTERVAL_CHUNKS = 25
CSV_BUFFER_SIZE = 4 * 1024 * 1024


def base_fen(fen: str) -> str:
    """Первые 4 поля FEN (расстановка, ход, рокировки, en passant) — уникальность позиции."""
    return " ".join(fen.split(" ", 4)[:4])


def fen_hash(base: str) -> bytes:
    return hashlib.blake2b(base.encode("utf-8"), digest_size=8).digest()


def get_bucket(fullmove: int, score: int) -> str:
    if fullmove <= OPENING_MAX_FULLMOVE:
        phase = "opening"
    elif fullmove <= MIDDLEGAME_MAX_FULLMOVE:
        phase = "middlegame"
    else:
        phase = "endgame"

    a = abs(score)
    if a <= CLOSE_MAX_ABS_EVAL:
        magnitude = "close"
    elif a <= EDGE_MAX_ABS_EVAL:
        magnitude = "edge"
    else:
        magnitude = "decisive"

    return f"{phase}_{magnitude}"


def fullmove_from_fen(fen: str) -> int:
    parts = fen.split(" ")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 1


def split_games(text_stream):
    buf = []
    game_lines = []
    count = 0
    for line in text_stream:
        if line.startswith("[Event "):
            if game_lines:
                buf.append("".join(game_lines))
                game_lines = []
                count += 1
                if count >= GAMES_PER_CHUNK:
                    yield buf
                    buf = []
                    count = 0
        game_lines.append(line)
    if game_lines:
        buf.append("".join(game_lines))
    if buf:
        yield buf


def process_chunk(games_text_list):
    """Выполняется в отдельном процессе. Для каждой позиции, прошедшей
    фильтры, дополнительно ищем ход, сыгранный ИЗ этой позиции следующим
    полуходом (look-ahead по mainline) — он и станет BestMove в CSV."""
    results = []
    for text in games_text_list:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue

        nodes = list(game.mainline())
        if not nodes:
            continue

        board = game.board()
        for i, node in enumerate(nodes):
            board.push(node.move)

            if board.fullmove_number < MIN_FULLMOVE:
                continue
            if SKIP_HALFMOVE_ZERO and board.halfmove_clock == 0:
                continue

            eval_obj = node.eval()
            if eval_obj is None or eval_obj.is_mate():
                continue

            score = eval_obj.white().score()
            if score is None or not (MIN_EVAL <= score <= MAX_EVAL):
                continue

            if SKIP_CHECKS and board.is_check():
                continue

            # Ход, сыгранный из ТЕКУЩЕЙ (уже пропушенной) позиции —
            # это следующий узел в mainline. Если текущая позиция —
            # последняя в партии, bestmove отсутствует (пустая строка).
            if i + 1 < len(nodes):
                next_move = nodes[i + 1].move
                bestmove_uci = next_move.uci()
            else:
                bestmove_uci = ""

            results.append((board.fen(), score, bestmove_uci))

    return results, len(games_text_list)


def _check_existing_header(path):
    """Проверяет, что уже существующий CSV в новом формате (3 колонки).
    Если формат старый (FEN,Eval) — возвращает False, чтобы не смешивать
    строки разных форматов в одном файле."""
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
    return first_line == CSV_HEADER


def process_lichess_pgn():
    seen_hashes = set()
    bucket_counts = collections.Counter()
    games_processed = 0

    bucket_targets = {b: int(TARGET_POSITIONS * w) for b, w in BUCKET_WEIGHTS.items()}

    if os.path.exists(OUTPUT_FILE):
        if not _check_existing_header(OUTPUT_FILE):
            print(
                f"ОШИБКА: {OUTPUT_FILE} уже существует, но в старом формате "
                f"(без колонки BestMove). Продолжать нельзя — это смешает "
                f"строки с 2 и 3 колонками в одном CSV.\n"
                f"Переименуйте/удалите старый {OUTPUT_FILE} и {CHECKPOINT_FILE} "
                f"и запустите сбор заново (чекпоинт по номеру партии всё равно "
                f"не поможет добавить BestMove в уже собранные старые строки)."
            )
            return

        print(f"Загрузка существующих позиций из {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "r", encoding="utf-8", newline="") as f:
            reader = f.readlines()
        for line in reader[1:]:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            fen = parts[0]
            score_str = parts[1]
            seen_hashes.add(fen_hash(base_fen(fen)))
            try:
                score = int(score_str)
                bucket = get_bucket(fullmove_from_fen(fen), score)
                bucket_counts[bucket] += 1
            except ValueError:
                pass
        print(f"В памяти уникальных позиций: {len(seen_hashes)}")
        print("Заполненность бакетов на старте:")
        for b, target in bucket_targets.items():
            print(f"  {b:22s} {bucket_counts[b]:>8d} / {target}")

    if len(seen_hashes) >= TARGET_POSITIONS:
        print("Целевое количество позиций уже собрано. Работа завершена.")
        return

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
            games_processed = data.get("games_processed", 0)
        print(f"Восстановление с партии №{games_processed}...")

    start_time = time.time()
    positions_at_start = len(seen_hashes)
    n_with_bestmove = 0

    with open(INPUT_FILE, "rb") as zst_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(zst_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            if games_processed > 0:
                print("Пропуск старых партий (это может занять пару минут)...")
                for _ in range(games_processed):
                    chess.pgn.skip_game(text_stream)
                print("Пропуск завершён, начинаю обработку новых партий.")

            mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
            with open(OUTPUT_FILE, mode, encoding="utf-8", newline="", buffering=CSV_BUFFER_SIZE) as out_file:
                if mode == "w":
                    out_file.write(CSV_HEADER)

                chunk_iter = split_games(text_stream)
                pending = collections.deque()
                max_in_flight = NUM_WORKERS * IN_FLIGHT_MULTIPLIER
                chunks_since_checkpoint = 0
                target_reached = False

                with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

                    def try_submit():
                        try:
                            chunk = next(chunk_iter)
                        except StopIteration:
                            return False
                        pending.append(executor.submit(process_chunk, chunk))
                        return True

                    for _ in range(max_in_flight):
                        if not try_submit():
                            break

                    try:
                        while pending:
                            fut = pending.popleft()
                            results, n_games = fut.result()
                            games_processed += n_games

                            for fen, score, bestmove_uci in results:
                                if len(seen_hashes) >= TARGET_POSITIONS:
                                    target_reached = True
                                    break

                                h = fen_hash(base_fen(fen))
                                if h in seen_hashes:
                                    continue

                                bucket = get_bucket(fullmove_from_fen(fen), score)
                                if bucket_counts[bucket] >= bucket_targets[bucket]:
                                    continue

                                seen_hashes.add(h)
                                bucket_counts[bucket] += 1
                                if bestmove_uci:
                                    n_with_bestmove += 1
                                out_file.write(f"{fen},{score},{bestmove_uci}\n")

                            if not target_reached:
                                try_submit()

                            chunks_since_checkpoint += 1
                            if target_reached or chunks_since_checkpoint >= SAVE_INTERVAL_CHUNKS:
                                out_file.flush()
                                with open(CHECKPOINT_FILE, "w") as f:
                                    json.dump({"games_processed": games_processed}, f)

                                elapsed = time.time() - start_time
                                new_positions = len(seen_hashes) - positions_at_start
                                speed_pos = new_positions / elapsed if elapsed > 0 else 0
                                print(
                                    f"Партий: {games_processed} | "
                                    f"Собрано: {len(seen_hashes)}/{TARGET_POSITIONS} | "
                                    f"С BestMove: {n_with_bestmove} | "
                                    f"Скорость: {speed_pos:.0f} позиций/сек"
                                )
                                start_time = time.time()
                                positions_at_start = len(seen_hashes)
                                chunks_since_checkpoint = 0

                            if target_reached:
                                executor.shutdown(wait=False, cancel_futures=True)
                                break
                    except KeyboardInterrupt:
                        print("\nПрервано пользователем — сохраняю чекпоинт...")
                        out_file.flush()
                        with open(CHECKPOINT_FILE, "w") as f:
                            json.dump({"games_processed": games_processed}, f)
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

                if not target_reached:
                    print("Достигнут конец файла!")

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"games_processed": games_processed}, f)

    print(f"\nГотово! Собрано уникальных позиций: {len(seen_hashes)}")
    print(f"Из них с BestMove: {n_with_bestmove}")
    print("Итоговое распределение по бакетам:")
    for b, target in bucket_targets.items():
        print(f"  {b:22s} {bucket_counts[b]:>8d} / {target}")


if __name__ == "__main__":
    process_lichess_pgn()