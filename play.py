import sys
import os
import json
import chess
import numpy as np
from engine import Engine


def load_engine():
    # Загружаем обученные веса, если файл существует, иначе используем дефолтные
    weights_path = "../chess_engine_20260907_191818/weights.json"
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            data = json.load(f)
            weights = np.array(data["weights"], dtype=np.float64)
    else:
        weights = None

    return Engine(weights=weights)


def main():
    engine = load_engine()
    board = chess.Board()

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        parts = line.split()
        command = parts[0]

        if command == "uci":
            print("id name MyEngine")
            print("id author Vadym Zelenko")
            print("uciok")

        elif command == "isready":
            print("readyok")

        elif command == "ucinewgame":
            engine.tt = {}  # Очищаем таблицу транспозиций перед новой партией

        elif command == "position":
            if "startpos" in parts:
                board.set_fen(chess.STARTING_FEN)
                moves_idx = parts.index("moves") if "moves" in parts else -1
            elif "fen" in parts:
                fen_idx = parts.index("fen")
                moves_idx = parts.index("moves") if "moves" in parts else -1
                fen_str = " ".join(parts[fen_idx + 1: moves_idx if moves_idx != -1 else len(parts)])
                board.set_fen(fen_str)
            else:
                moves_idx = -1

            # Применяем ходы, если они переданы после FEN или startpos
            if moves_idx != -1:
                for move_str in parts[moves_idx + 1:]:
                    board.push_uci(move_str)

        elif command == "go":
            time_limit = 5.0
            max_depth = 64

            # Парсинг базовых настроек времени
            if "movetime" in parts:
                idx = parts.index("movetime")
                time_limit = int(parts[idx + 1]) / 1000.0
            elif "wtime" in parts and "btime" in parts:
                wtime_idx = parts.index("wtime")
                btime_idx = parts.index("btime")
                wtime = int(parts[wtime_idx + 1])
                btime = int(parts[btime_idx + 1])

                # Выделяем ~5% от оставшегося времени на текущий ход
                time_left = wtime if board.turn == chess.WHITE else btime
                time_limit = (time_left / 1000.0) / 20.0

            best_move, _ = engine.search(board, time_limit=time_limit, max_depth=max_depth)

            if best_move:
                print(f"bestmove {best_move.uci()}")
            else:
                print("bestmove 0000")

        elif command == "quit":
            break

        # Обязательный сброс буфера для корректного общения по протоколу UCI
        sys.stdout.flush()


if __name__ == "__main__":
    main()