import chess
import chess.engine
import numpy as np
import json
from engine import Engine

# Путь к Stockfish. Для мака с Apple Silicon (Homebrew) обычно такой:
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


# Если у вас Intel Mac, путь может быть: "/usr/local/bin/stockfish"

def load_my_engine():
    with open("../chess_engine_20260907_191818/weights.json", "r") as f:
        weights = np.array(json.load(f)["weights"], dtype=np.float64)
    return Engine(weights=weights)


def main():
    my_engine = load_my_engine()
    board = chess.Board()

    print("Запуск матча: Ваша Модель (Белые) vs Stockfish (Черные)")

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as sf:
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                print(f"\n--- Ход {board.fullmove_number}. Ваша модель думает... ---")

                # Объективная оценка ДО хода модели
                info_before = sf.analyse(board, chess.engine.Limit(time=0.1))
                score_before = info_before["score"].white().score(mate_score=10000) / 100.0

                # Модель выбирает ход (даем ей 1 секунду)
                best_move, my_cp = my_engine.search(board, time_limit=1.0, max_depth=64)

                if not best_move:
                    print("Модель не нашла ход и сдается.")
                    break

                print(f"Модель сходила: {best_move} (её иллюзия оценки: {my_cp / 100.0:+.2f})")
                board.push(best_move)

                # Объективная оценка ПОСЛЕ хода модели
                info_after = sf.analyse(board, chess.engine.Limit(time=0.1))
                score_after = info_after["score"].white().score(mate_score=10000) / 100.0

                error = score_after - score_before

                # Вывод вердикта
                verdict = "✅ Отличный ход" if error > -0.2 else ("⚠️ Неточность" if error > -1.0 else "❌ Грубый зевок")
                print(f"Оценка Stockfish: {score_after:+.2f} | Потеря: {error:.2f} пешек -> {verdict}")
                print(board)

            else:
                print("\n--- Stockfish наносит ответный удар ---")
                # Стокфишу даем всего 0.1 сек, чтобы не затягивать вывод
                result = sf.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
                print(f"Stockfish сходил: {result.move}")
                print(board)

        print(f"\nИгра окончена! Результат: {board.result()}")


if __name__ == "__main__":
    main()