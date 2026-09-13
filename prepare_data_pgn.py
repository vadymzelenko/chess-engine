"""
prepare_data_pgn.py — конвертация PGN-датасетов с [%eval]-аннотациями
(например, chessmont lichess-2400-eval.pgn.zst, lichess-2500-eval.pgn.zst
и т.п. с https://database.chessmont.com/) в {"fen": ..., "cp": ...} для
tuner.py.

ОТЛИЧИЕ ОТ prepare_data.py (сырой lichess_db_eval.jsonl.zst):
Здесь вход — PGN-партии, комментарий после хода вида "{ [%eval 0.34] }"
(иногда с [%clk ...] рядом). ВАЖНО: этот eval всегда дан С ТОЧКИ ЗРЕНИЯ
БЕЛЫХ (не "со стороны хода", как в сыром дампе lichess_db_eval). Чтобы
tuner.py не пришлось трогать — он уже написан под конвенцию "cp дан со
стороны хода, кто ходит следующим" (см. _parse_line: `if not board.turn:
cp = -cp`) — здесь мы сами пересчитываем white-POV eval в эту конвенцию
перед записью:
    cp_side_to_move = cp_white  if позиция после хода: ходят белые
                     = -cp_white if ходят чёрные
Это ровно компенсирует инверсию внутри tuner.py, так что итоговый target
(sigmoid от эвала с точки зрения белых) получится корректным без правок
tuner.py/features.py.

Мат-аннотации ("[%eval #3]", "[%eval #-2]") пропускаются — как и в
prepare_data.py, они нелинейны относительно cp и портят регрессию.

Использование:
    zstd -dc lichess-2400-eval.pgn.zst | python prepare_data_pgn.py - \
        --out train.jsonl --limit 3000000

Зависимости: python-chess (уже есть в проекте).
"""
import argparse
import io
import json
import random
import re
import sys

import chess
import chess.pgn

EVAL_RE = re.compile(r"\[%eval\s+([^\]]+)\]")


def open_input(path):
    if path == "-":
        return sys.stdin
    if path.endswith(".zst"):
        try:
            import zstandard as zstd
        except ImportError:
            sys.exit(
                "Нужен пакет zstandard (pip install zstandard), либо распакуйте "
                "заранее: zstd -dc file.pgn.zst | python prepare_data_pgn.py -"
            )
        f = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(f), encoding="utf-8", errors="replace")
    return open(path, "r")


def parse_eval_comment(comment):
    """Возвращает cp_white (int) или None (нет eval / это мат)."""
    m = EVAL_RE.search(comment or "")
    if not m:
        return None
    raw = m.group(1).strip()
    if raw.startswith("#"):  # мат — пропускаем
        return None
    try:
        pawns = float(raw)
    except ValueError:
        return None
    return int(round(pawns * 100))


def is_quiet(board):
    if board.is_check():
        return False
    for move in board.legal_moves:
        if board.is_capture(move):
            return False
    return True


def iter_positions(game, cp_cap, skip_plies, require_quiet):
    board = game.board()
    ply = 0
    node = game
    while node.variations:
        next_node = node.variations[0]  # только mainline, без побочных вариантов
        move = next_node.move
        board.push(move)
        ply += 1

        cp_white = parse_eval_comment(next_node.comment)
        node = next_node

        if cp_white is None:
            continue
        if ply <= skip_plies:
            continue
        if abs(cp_white) > cp_cap:
            continue
        if require_quiet and not is_quiet(board):
            continue

        cp_side_to_move = cp_white if board.turn else -cp_white
        yield board.fen(), cp_side_to_move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="путь к .pgn(.zst) файлу с [%%eval], или '-' для stdin")
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                     help="итоговый размер датасета (reservoir sampling)")
    ap.add_argument("--cp-cap", type=int, default=1000)
    ap.add_argument("--skip-plies", type=int, default=10,
                     help="пропускать первые N полуходов партии (дебютная теория)")
    ap.add_argument("--no-quiet-filter", action="store_true")
    ap.add_argument("--max-games", type=int, default=None,
                     help="ограничить число разобранных партий (для быстрого теста)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reservoir = []
    n_candidates = 0
    n_games = 0

    f = open_input(args.data)
    try:
        while True:
            if args.max_games is not None and n_games >= args.max_games:
                break
            try:
                game = chess.pgn.read_game(f)
            except Exception as e:
                print(f"пропущена битая партия: {e}", file=sys.stderr)
                continue
            if game is None:
                break
            n_games += 1
            if n_games % 200_000 == 0:
                print(f"...партий: {n_games}, кандидатов: {n_candidates}", file=sys.stderr)

            for fen, cp in iter_positions(
                game,
                cp_cap=args.cp_cap,
                skip_plies=args.skip_plies,
                require_quiet=not args.no_quiet_filter,
            ):
                n_candidates += 1
                rec = {"fen": fen, "cp": cp}
                if args.limit is None:
                    reservoir.append(rec)
                else:
                    if len(reservoir) < args.limit:
                        reservoir.append(rec)
                    else:
                        j = rng.randint(0, n_candidates - 1)
                        if j < args.limit:
                            reservoir[j] = rec
    finally:
        if f is not sys.stdin:
            f.close()

    with open(args.out, "w") as out_f:
        for rec in reservoir:
            out_f.write(json.dumps(rec) + "\n")

    print(f"Готово. Партий: {n_games}, кандидатов-позиций: {n_candidates}, "
          f"записано в {args.out}: {len(reservoir)}")


if __name__ == "__main__":
    main()
