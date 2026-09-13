<div align="center">

# ♞ Chess Engine
**Полный цикл построения шахматного движка: от оценочной функции до игры на Lichess**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-C9A961?style=flat-square)](LICENSE)
[![python-chess](https://img.shields.io/badge/python--chess-1.11+-4B8BBE?style=flat-square)](https://github.com/niklasf/python-chess)

*Классический alpha-beta движок с Texel-тюнингом оценочной функции &nbsp;·&nbsp; Нейросетевой AlphaZero-style движок с MCTS*

</div>

---

## Содержание

- [О проекте](#о-проекте)
- [Архитектура](#архитектура)
- [Структура репозитория](#структура-репозитория)
- [Описание файлов](#описание-файлов)
  - [Ядро: признаки и оценка](#ядро-признаки-и-оценка)
  - [Обучение: пайплайн данных](#обучение-пайплайн-данных)
  - [Классический движок](#классический-движок)
  - [Нейросетевой движок](#нейросетевой-движок)
  - [Веб-интерфейсы](#веб-интерфейсы)
  - [Тестирование и игра](#тестирование-и-игра)
  - [Конфигурация и данные](#конфигурация-и-данные)
  - [Ноутбуки](#ноутбуки)
- [Быстрый старт](#быстрый-старт)
- [Обученные веса и модель](#обученные-веса-и-модель)
- [Требования](#требования)
- [Лицензия](#лицензия)

---

## О проекте

Этот репозиторий — **технический отчёт в виде кода**. Он показывает два принципиально разных подхода к построению шахматного движка, реализованных полностью с нуля:

### ⚙️ Классический подход
**Оценочная функция** задаётся вручную: 768 параметров Piece-Square Tables (6 типов фигур × 64 клетки × 2 фазы игры). Параметры оптимизируются методом **Texel Tuning** на 3 миллионах партий Lichess с рейтингом 2400+ ELO. **Поиск** — alpha-beta отсечение с четырьмя классическими оптимизациями: транспозиционная таблица, quiescence search, сортировка ходов и итеративное углубление.

### 🧠 Нейросетевой подход
**DualHeadResNet** — свёрточная сеть с 16 residual блоками, SE-вниманием и двумя головами: policy (4672 вероятности ходов в стиле AlphaZero) и value (оценка позиции). **Поиск** — MCTS с формулой PUCT и батчированием симуляций через технику virtual loss. Обучение — self-play.

---

## Архитектура

```text
┌─────────────────────────────────────┐
│         Lichess PGN (.zst)          │
│     партии 2400+ ELO с [%eval]      │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────────────┐
       │                       │
       ▼                       ▼
┌────────────────────┐  ┌────────────────────┐
│ prepare_data_pgn   │  │    data_compile    │
│   PGN → jsonl      │  │     PGN → CSV      │
└─────────┬──────────┘  └─────────┬──────────┘
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │     features.py      │  Извлечение признаков
           │  768 параметров PST  │  (tapered eval)
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │       tuner.py       │  Adam поверх sparse
           │     weights.json     │  design-матрицы
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │      engine.py       │  Alpha-beta + TT
           │    (классический)    │  + quiescence
           └──────────────────────┘
──────────────────────────────────────────────────────────────────────
           ┌──────────────────────┐
           │   chess_engine.py    │  DualHeadResNet
           │    (нейросетевой)    │  + MCTS с PUCT
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │    uci_engine.py     │  UCI-протокол
           │       play.py        │  для lichess-bot
           └──────────────────────┘
```

---

## Структура репозитория

```text
chess-engine/
│
├── 📄 README.md              Документация
├── 📄 LICENSE                MIT License
├── 📄 config.yml             Конфигурация lichess-bot
├── 📄 index.html             Веб-лендинг проекта
├── 📄 weights.json           Обученные Texel-веса (768 параметров)
│
├── 🧠 Ядро: признаки и оценка
│   ├── features.py           Признаки позиции и tapered eval
│   └── engine.py             Alpha-beta движок
│
├── 📊 Обучение: пайплайн данных
│   ├── prepare_data_pgn.py   PGN (.zst) → jsonl
│   ├── data_compile.py       PGN → CSV + стратификация
│   ├── data_compile.cpp      C++ версия (экспериментальная)
│   └── tuner.py              Adam-оптимизация весов
│
├── ⚡ Нейросетевой движок
│   ├── chess_engine.py       DualHeadResNet + MCTS
│   ├── claude_engine.py      Улучшенный alpha-beta (экспериментальный)
│   └── uci_engine.py         UCI-обёртка для lichess-bot
│
├── 🌐 Веб-интерфейсы
│   ├── web_app.py            FastAPI + React UI
│   └── server.py             Flask + React UI
│
├── 🎮 Игра и тесты
│   ├── play.py               UCI-цикл для игры с движком
│   └── vs_stockfish.py       Тест силы против Stockfish
│
└── 📓 Ноутбуки
    ├── texel.ipynb           Обучение Texel-eval в Colab
    └── chessAI.ipynb         Self-play обучение нейросети
```

---

## Описание файлов

### Ядро: признаки и оценка

#### 📄 `features.py`
**Назначение:** извлечение признаков позиции и вычисление tapered eval.  
Файл определяет 768 параметров оценочной функции. Для каждого типа фигуры и каждой клетки доски хранится два веса: `midgame` и `endgame`. Итоговая оценка — линейная интерполяция между ними по «фазе игры» (от 24 в начальной позиции до 0 в чистом эндшпиле).

**Ключевые функции:**
| Функция | Назначение |
|---------|-----------|
| `extract_features(board)` | Извлекает списки индексов активных параметров |
| `eval_from_features(feats, phase, weights)` | Считает оценку по весам |
| `evaluate(board, weights)` | Полный forward: board → cp |
| `default_weights()` | Стартовые веса (классические значения фигур) |

**Особенности реализации:**
- Чёрные фигуры читаются через **зеркало по горизонтали** (`chess.square_mirror`) — это уменьшает число параметров вдвое и делает признаки инвариантными к цвету.
- Используется `board.piece_map()` — один вызов вместо 12 обходов доски.
- Признаки разделены на четыре списка `add_mg / sub_mg / add_eg / sub_eg` — чтобы знак (+1 для белых, −1 для чёрных) кодировался неявно, без хранения отдельного массива знаков.

**Пример:**
```python
import chess
from features import evaluate, default_weights

board = chess.Board()
weights = default_weights()
score = evaluate(board, weights)  # 0 в начальной позиции
```

#### 📄 `engine.py`
**Назначение:** классический alpha-beta движок с четырьмя оптимизациями.

**Компоненты:**
| Компонент | Что делает |
|-----------|-----------|
| `Engine.alphabeta()` | Negamax с alpha-beta отсечением и TT |
| `Engine.quiescence()` | Продление поиска по взятиям |
| `Engine._order_moves()` | Сортировка ходов (TT → MVV-LVA → killers) |
| `Engine.search()` | Итеративное углубление с контролем времени |
| `Engine._tt_store()` | Запись в транспозиционную таблицу с FIFO-eviction |

**Особенности:**
- Transposition Table на базе Zobrist-хеша с тремя типами записей: EXACT, LOWER, UPPER.
- Коррекция mate-score при записи и чтении из TT — без неё движок не видит мат при транспозиции.
- MVV-LVA сортировка взятий: жертва ценная, атакующий дешёвый.
- Killer moves хранятся по ply, а не по глубине — это стандартный приём.
- TT eviction (FIFO): при переполнении таблицы вытесняется самая старая запись, а не отбрасывается новая. Это критично для длинных партий.

**Пример использования:**
```python
import json
import chess
import numpy as np
from engine import Engine

weights = np.array(json.load(open("weights.json"))["weights"])
engine = Engine(weights=weights)
board = chess.Board()
best_move, score = engine.search(board, time_limit=2.0, max_depth=64)
print(f"Лучший ход: {best_move} (оценка: {score} cp)")
```

### Обучение: пайплайн данных

#### 📄 `prepare_data_pgn.py`
**Назначение:** конвертация PGN-датасетов с аннотациями `[%eval ...]` в формат jsonl для тюнера.

**Что делает:**
- Читает `.pgn` или `.pgn.zst` (с автоматической распаковкой через `zstandard`).
- Парсит комментарии вида `{ [%eval 0.34] }` после каждого хода.
- Применяет фильтры:
  - пропуск первых 10 полуходов (дебютная теория),
  - только «тихие» позиции (без шаха, без взятия на следующем ходу),
  - отсечение по `|eval| > 1000 cp`,
  - пропуск мат-аннотаций (`#3`, `#-2`).
- Пересчитывает white-POV eval в сторону хода — компенсирует внутреннюю конвенцию тюнера.
- Делает reservoir sampling до заданного `--limit`.

**Команда:**
```bash
zstd -dc lichess-2400-eval.pgn.zst | python prepare_data_pgn.py -     --out train.jsonl     --limit 3000000     --cp-cap 1000     --skip-plies 10
```

#### 📄 `data_compile.py`
**Назначение:** параллельный парсинг PGN со стратифицированной выборкой и записью в CSV.

**Отличия от `prepare_data_pgn.py`:**
| Особенность | `prepare_data_pgn` | `data_compile` |
|-------------|--------------------|----------------|
| Формат выхода | jsonl | CSV с 3 колонками |
| BestMove label | ✗ | ✓ (для policy-головы) |
| Стратификация | ✗ | ✓ (9 бакетов) |
| Параллелизм | ✗ | ✓ (ProcessPoolExecutor) |
| Checkpoint по номеру партии | ✗ | ✓ |

Стратификация — ключевой приём. Без неё датасет перекошен в сторону дебютных равных позиций (около 60%), и endgame-параметры недообучаются. Используется 9 бакетов: `{opening, middlegame, endgame} × {close, edge, decisive}`.

**Команда:**
```bash
python data_compile.py
# Настроить INPUT_FILE, OUTPUT_FILE и TARGET_POSITIONS в начале файла
```

#### 📄 `data_compile.cpp`
**Назначение:** экспериментальная C++ версия `data_compile.py`.  
Заготовка для ускорения парсинга PGN в 10–20 раз. На данный момент содержит только скелет — полноценная реализация требует интеграции с `libchess` или написания собственного парсера.  
**Статус:** ⚠️ экспериментальный, не используется в production-пайплайне.

#### 📄 `tuner.py`
**Назначение:** обучение 768 весов оценочной функции методом Texel Tuning.

**Как работает:**
- Читает jsonl с `{fen, cp}`.
- Строит sparse design-матрицу $X$ формы $(N, 768)$: для каждой позиции заполняются только те колонки, которые соответствуют фигурам на доске (≈ 30 nonzeros на строку).
- Запускает mini-batch Adam на MSE между $\sigma(	ext{eval})$ и результатом партии.
- Сохраняет чекпоинты каждую эпоху (веса + состояние Adam + номер эпохи).
- При `--resume` — продолжает с последнего чекпоинта (критично для Colab).

**Производительность:**
196 000 позиций → эпоха за 0.9 секунды на CPU.  
Причина: два sparse matmul на батч через BLAS. Никаких Python-циклов по фичам.

**Команда:**
```bash
python tuner.py train.jsonl     --epochs 30     --lr 1.0     --batch-size 8192     --out weights.json     --checkpoint-dir ckpt/     --resume     --init-material
```

**Результат:** `weights.json` с 768 числами. Train MSE = val MSE = 0.0104 (без переобучения — модель имеет всего 768 параметров на 196 000 примеров).

### Классический движок

#### 📄 `play.py`
**Назначение:** UCI-цикл для игры с движком в консоли или через GUI (Arena, CuteChess).

**Что делает:**
- Принимает UCI-команды: `uci`, `isready`, `ucinewgame`, `position`, `go`, `quit`.
- Парсит бюджет времени из go-команды: `movetime`, `wtime`/`btime` + `winc`/`binc`.
- Выделяет 5% от оставшегося времени на ход (типичная эвристика).
- Возвращает `bestmove <uci>`.

**Использование:**
```bash
python play.py
# > uci
# < id name MyEngine
# < uciok
# > position startpos moves e2e4 e7e5
# > go movetime 1000
# < bestmove g1f3
```

#### 📄 `claude_engine.py`
**Назначение:** экспериментальная, более полная реализация alpha-beta движка с PVS (Principal Variation Search), history heuristic и quiescence depth limits.

**Отличия от `engine.py`:**
| Компонент | `engine.py` | `claude_engine.py` |
|-----------|-------------|---------------------|
| Поиск | Negamax | PVS (Principal Variation Search) |
| Сортировка | TT + MVV-LVA + killers | + history heuristic |
| Quiescence | без лимита | с лимитом `max_qdepth=8` |
| Aspiration windows | ✗ | заготовка |
| Web UI | ✗ | ✓ (встроенный HTML) |

**Статус:** ⚠️ экспериментальный, но полностью функциональный. Содержит встроенный веб-интерфейс на FastAPI.

**Запуск:**
```bash
python claude_engine.py
# Откроется http://127.0.0.1:8000
```

### Нейросетевой движок

#### 📄 `chess_engine.py`
**Назначение:** ядро нейросетевого движка — модель + MCTS.

**Что содержит:**
| Класс / функция | Назначение |
|-----------------|-----------|
| `DualHeadResNet` | Свёрточная сеть с двумя головами (policy + value) |
| `SEBlock` | Squeeze-and-Excitation внимание |
| `ResBlock` | Residual блок с SE |
| `fen_to_planes()` | Кодирование позиции в 54 плоскости |
| `encode_move()` | AlphaZero-кодирование хода (73 × 64) |
| `MCTS` | Дерево поиска с PUCT и virtual loss |
| `Node` | Узел дерева (prior, N, W, children) |
| `load_model()` | Загрузка чекпоинта с автоматическим чтением конфига |
| `evaluate_single()` | Один forward pass сети |

Ключевая оптимизация — **virtual loss**. Вместо одной симуляции за раз, MCTS собирает батч из 16 листьев, временно «наказывая» посещённые узлы, чтобы следующие выборки шли по другим веткам. Затем — один forward pass на весь батч. На GPU это даёт ускорение в 8–12 раз.

**Вход сети — 54 плоскости:**
- 0–11: фигуры текущей позиции (6 белых + 6 чёрных)
- 12: чей ход
- 13–16: права рокировки (K, Q, k, q)
- 17: en passant
- 18–53: 3 предыдущие позиции × 12 плоскостей каждая

**Выход:**
- Policy: 4672 логита (73 × 64)
- Value: скаляр ∈ [−1, +1] через tanh

#### 📄 `uci_engine.py`
**Назначение:** UCI-обёртка для нейросетевого движка — позволяет играть на Lichess через `lichess-bot`.

**Ключевые особенности:**
- **Онлайн-калибровка скорости MCTS.** Движок измеряет фактическую скорость (sims/sec) после каждого хода и использует скользящее среднее для следующего — это даёт точное попадание в бюджет времени.
- **История FEN-ов.** Модель требует 3 предыдущие позиции на входе. Движок ведёт список FEN-ов с начала партии и корректно восстанавливает историю при каждой команде `position`.
- **Устойчивость.** Обработка каждой команды в `try/except` — единичная ошибка GUI не роняет процесс.
- **Автоматический выбор устройства:** MPS (Apple Silicon) → CUDA → CPU. Можно переопределить через `CHESS_ENGINE_DEVICE=cpu`.

**Переменные окружения:**
```bash
CHESS_MODEL_PATH=chess_dualhead_best.pth   # путь к чекпоинту
CHESS_ENGINE_DEVICE=cpu                    # cpu / mps / cuda / auto
CHESS_N_SIMULATIONS=400                    # сила MCTS
CHESS_MCTS_BATCH_SIZE=16                   # батч virtual loss
CHESS_C_PUCT=1.5                           # константа PUCT
CHESS_TEMPERATURE=0.0                      # 0 = всегда лучший ход
```

**CLI-режимы для диагностики:**
```bash
# Замер скорости MCTS на текущем железе
python uci_engine.py bench

# Быстрая оценка одной позиции
python uci_engine.py eval "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Обычный UCI-цикл (для lichess-bot или GUI)
python uci_engine.py
```

### Веб-интерфейсы

#### 📄 `web_app.py`
**Назначение:** веб-интерфейс на FastAPI + React для игры против классического движка.

**Что делает:**
- Отдаёт HTML-страницу с React-доской (drag-and-drop).
- API:
  - `POST /api/move` — ход игрока
  - `POST /api/engine_move` — ход движка
  - `POST /api/reset` — сброс доски
- Автоматически открывает браузер при запуске.
- Работает на `http://127.0.0.1:8000`

**Запуск:**
```bash
python web_app.py
```

#### 📄 `server.py`
**Назначение:** веб-интерфейс на Flask + React — альтернатива `web_app.py` с более современным дизайном.

**Отличия от `web_app.py`:**
| Особенность | `web_app.py` | `server.py` |
|-------------|--------------|-------------|
| Фреймворк | FastAPI | Flask |
| Стиль доски | нейтральный | Chess.com-стиль |
| Drag-and-drop | ✓ | ✓ |
| Undo (отмена хода) | ✗ | ✓ |
| Индикатор «думает» | ✓ | ✓ |
| Порт | 8000 | 5050 |

**Запуск:**
```bash
python server.py
# Откроется http://127.0.0.1:5050
```

### Тестирование и игра

#### 📄 `vs_stockfish.py`
**Назначение:** тест силы движка против Stockfish.

**Что делает:**
- Загружает `weights.json` и создаёт экземпляр `Engine`.
- Запускает партию: наша модель (белые) против Stockfish (чёрные).
- Перед каждым ходом модели:
  - Запрашивает у Stockfish оценку позиции до (0.1 сек).
  - Делает ход модели (1 сек на ход).
  - Запрашивает у Stockfish оценку после.
  - Считает потерю в пешках и выводит вердикт:
    - ✅ Отличный ход (потеря < 0.2 пешки)
    - ⚠️ Неточность (потеря < 1.0)
    - ❌ Грубый зевок (потеря ≥ 1.0).
- Stockfish отвечает за 0.1 сек (быстро, чтобы не затягивать вывод).

**Требования:**
- Установленный Stockfish: `brew install stockfish` (macOS) или `apt install stockfish` (Linux).
- Путь к бинарнику настраивается в переменной `STOCKFISH_PATH`.

**Запуск:**
```bash
python vs_stockfish.py
```

### Конфигурация и данные

#### 📄 `config.yml`
**Назначение:** конфигурационный файл для `lichess-bot` — фреймворка для игры на Lichess.

**Что настраивает:**
| Секция | Значение |
|--------|----------|
| `token` | OAuth2-токен бота (получается на Lichess) |
| `engine` | Настройки UCI-движка (путь, интерпретатор, протокол) |
| `challenge` | Параметры входящих вызовов (контроль времени, варианты) |
| `matchmaking` | Параметры автоматических вызовов других ботов |
| `greeting` | Сообщения в чате |

**Ключевые параметры:**
```yaml
engine:
  name: "uci_engine.py"       # Наш движок
  interpreter: "python"        # Через интерпретатор
  protocol: "uci"
challenge:
  variants:                    # Только стандартные шахматы
    - standard
  time_controls:               # Какие контроли принимать
    - bullet
    - blitz
    - rapid
```

⚠️ **Важно:** `token` в файле — это placeholder. Замените на свой реальный токен, полученный через [lichess.org/account/oauth/token](https://lichess.org/account/oauth/token).

#### 📄 `weights.json`
**Назначение:** обученные веса оценочной функции (768 чисел с плавающей точкой).

**Структура:**
```json
{
  "weights": [
    82.0, 82.0, ...,    // индексы 0-63:       пешка, midgame
    337.0, ...,          // индексы 64-127:     конь, midgame
    ...
    140.44, ...,         // индексы 384-447:    пешка, endgame
    ...
  ]
}
```

**Схема индексов:**
| Диапазон | Что содержит |
|----------|-------------|
| 0 – 383 | Midgame-веса (6 фигур × 64 клетки) |
| 384 – 767 | Endgame-веса (6 фигур × 64 клетки) |

**Как получить:** обучение через `tuner.py` (см. [Быстрый старт](#быстрый-старт)).

**Загрузка в код:**
```python
import json
import numpy as np

with open("weights.json") as f:
    weights = np.array(json.load(f)["weights"], dtype=np.float64)
```

#### 📄 `index.html`
**Назначение:** веб-лендинг проекта — технический отчёт в стиле научной статьи.

**Что содержит:**
- Аннотацию и метаданные документа
- Боковое содержание (sticky TOC) с подсветкой активного раздела
- 8 разделов: введение, оценка, поиск, нейросети, результаты, источники, глоссарий, приложение
- Формулы с нумерацией (2.1), (3.1), (4.1)
- Листинги кода с подсветкой синтаксиса
- Библиографию из 12 источников
- Глоссарий из 17 терминов

**Стек:**
- Source Serif 4 для заголовков и body (академический шрифт)
- IBM Plex Sans для UI
- IBM Plex Mono для кода
- GSAP + ScrollTrigger для плавного появления секций

**Просмотр:**
- Локально: откройте `index.html` в браузере.
- Через GitHub Pages: `Settings → Pages → Branch: main`.

### Ноутбуки

#### 📓 `texel.ipynb`
**Назначение:** Google Colab ноутбук для обучения Texel-весов в облаке.

**Содержание:**
1. Монтирование Google Drive.
2. Установка зависимостей (`chess`, `zstandard`, `scipy`).
3. Копирование исходников в `/content/work`.
4. **Стадия 1 — сбор данных из PGN:**
   - стратифицированная выборка до 200 000 позиций,
   - сохранение design-матрицы на Drive (переживает перезапуск Colab).
5. **Стадия 2 — обучение с чекпоинтами:**
   - `--checkpoint-dir` на Drive,
   - `--resume` продолжает после обрыва сессии.
6. Финальное сохранение `weights.json`.
7. Упаковка кода + весов в zip на Drive.

**Особенности:**
- Двухстадийный пайплайн — критичен для Colab, где сессия обрывается через 12 часов. Сбор данных (часы) отделён от обучения (минуты).
- Design-матрица сохраняется один раз — повторное обучение не требует парсить PGN заново.

**Запуск:** загрузите `texel.ipynb` в Colab, положите PGN-датасет и `.py`-файлы в папку `chess_engine/` на Drive, запустите ячейки сверху вниз.

#### 📓 `chessAI.ipynb`
**Назначение:** Colab-ноутбук для обучения нейросетевого движка (self-play + supervised).

**Содержание:**
1. Монтирование Drive, установка `torch`, `python-chess`.
2. Скачивание лицензированных датасетов с Lichess (через `database.lichess.org`).
3. Проверка целостности `.pgn.zst` — читает первые партии, проверяет ELO и заголовки.
4. **Препроцессинг** — конвертация PGN в бинарные тензоры (`.pt` чанки):
   - фильтр по ELO ≥ 2400,
   - минимум 15 ходов,
   - формат: boards (half-precision для экономии места), policies (long), values (float).
5. **Датасет** — `IterableDataset` для потоковой загрузки без загрузки всего датасета в RAM.
6. **Модель** — `DualHeadChessNet` (упрощённая версия `DualHeadResNet`):
   - 12 входных плоскостей (без истории),
   - 128 каналов,
   - 10 residual блоков,
   - policy-голова на 4168 выходов.
7. **Обучение** — mixed precision (`torch.amp`), AdamW с cosine annealing, gradient clipping.
8. **MCTS-движок** — `MCTSNode`, `MCTS.run()`, UCI-цикл и консольный режим.
9. Сохранение чекпоинтов на Drive каждые $N$ шагов.

**Отличия от `chess_engine.py`:**
| Параметр | Colab-версия | Production |
|----------|--------------|------------|
| Входных плоскостей | 12 (без истории) | 54 (3 истории) |
| Каналов | 128 | 256 |
| Residual блоков | 10 | 16 |
| Policy size | 4168 | 4672 |

**Запуск:** загрузите `chessAI.ipynb` в Colab, убедитесь, что GPU включён (`Runtime → Change runtime type → T4 GPU`), запустите ячейки сверху вниз.

---

## Быстрый старт

### 1. Клонирование и установка
```bash
git clone https://github.com/USERNAME/chess-engine.git
cd chess-engine
pip install -r requirements.txt
```

### 2. Обучение Texel-весов (опционально)
Если у вас есть PGN-датасет с `[%eval]`:
```bash
# Конвертация PGN в jsonl
python prepare_data_pgn.py lichess-2400-eval.pgn.zst --out train.jsonl --limit 1000000

# Обучение
python tuner.py train.jsonl --out weights.json --epochs 30 --init-material
```
Или используйте готовый `weights.json` из раздела [Обученные веса и модель](#обученные-веса-и-модель).

### 3. Игра против классического движка
```bash
# Веб-интерфейс
python server.py
# Откроется http://127.0.0.1:5050
```

### 4. Игра на Lichess (нейросетевой движок)
```bash
# Получите токен на https://lichess.org/account/oauth/token
# Замените его в config.yml

# Запуск UCI-движка как бота
python uci_engine.py
```

### 5. Тест силы против Stockfish
```bash
python vs_stockfish.py
```

---

## Обученные веса и модель

Готовые артефакты (веса Texel-eval и чекпоинт нейросети) доступны по ссылкам ниже:

| Артефакт | Размер | Ссылка |
|----------|--------|--------|
| `weights.json` — Texel-eval (768 параметров) | ~15 KB | 📥 Download |
| `chess_dualhead_best.pth` — DualHeadResNet чекпоинт | ~40 MB | 📥 Download |

### Как использовать:
```bash
# Скачайте weights.json в корень проекта
wget ССЫЛКА_НА_ВЕСА -O weights.json

# Скачайте чекпоинт модели
wget ССЫЛКА_НА_МОДЕЛЬ -O chess_dualhead_best.pth

# Укажите путь к модели для UCI-движка
export CHESS_MODEL_PATH=$(pwd)/chess_dualhead_best.pth
python uci_engine.py
```

**Что внутри `weights.json`:**
- 768 чисел с плавающей точкой.
- Индексы 0–383: midgame-веса (6 фигур × 64 клетки).
- Индексы 384–767: endgame-веса.
- Обучено на 3M позициях Lichess 2400+ ELO.
- Финальный MSE: 0.0104.

**Что внутри `chess_dualhead_best.pth`:**
- Полный `state_dict` модели DualHeadResNet (54 входные плоскости, 256 каналов, 16 residual блоков).
- Конфигурация архитектуры (`config` в чекпоинте): `in_planes`, `channels`, `num_blocks`, `history_len`.
- Оптимизатор (не нужен для inference).
- Обучено методом self-play + supervised на датасете Lichess.

---

## Требования

### Python
```text
chess>=1.11
numpy>=1.24
scipy>=1.10
torch>=2.0
gradio>=5.0          # для play_gradio.py
fastapi>=0.100       # для web_app.py, claude_engine.py
uvicorn>=0.23        # для web_app.py, claude_engine.py
flask>=3.0           # для server.py
zstandard>=0.21      # для распаковки .pgn.zst
```

### Внешние утилиты (опционально)
- **Stockfish** — для теста силы:
  - macOS: `brew install stockfish`
  - Linux: `apt install stockfish`
  - Windows: [stockfishchess.org/download](https://stockfishchess.org/download/)
- **zstd** — для распаковки PGN-архивов: `apt install zstd` / `brew install zstd`

### Аппаратное обеспечение
| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| CPU (Texel training) | 4 ядра | 8+ ядер |
| RAM (Texel training) | 4 GB | 8 GB |
| GPU (Neural training) | CUDA-capable | NVIDIA T4+ |
| RAM (Neural inference) | 8 GB | 16 GB |
| Disk | 10 GB (dataset) | 50+ GB |

---

## Лицензия

Проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE).

Данные Lichess распространяются под CC0. Данные ChessMont — по условиям их сайта.

<div align="center">

♟️ **Chess Engine Project**  
*Сделано с ❤️ для сообщества шахматного программирования*

[Chess Programming Wiki](https://www.chessprogramming.org/) &nbsp;·&nbsp;
[Lichess Database](https://database.lichess.org/) &nbsp;·&nbsp;
[AlphaZero Paper](https://arxiv.org/abs/1712.01815) &nbsp;·&nbsp;
[Leela Chess Zero](https://lczero.org/)
[Interesting data and files](https://drive.google.com/drive/folders/1ax7fn876b3cY4Jqk0K_pxnmulKn9acFK?usp=sharing)

</div>
