# Chess Engine

Two approaches to building a chess engine, implemented end to end: from the evaluation function and its training data to playing on Lichess.

- **Classical:** alpha-beta search with a hand-structured evaluation (768 piece-square parameters) tuned on Lichess games with Texel tuning.
- **Neural:** an AlphaZero-style network (policy and value heads) with Monte Carlo tree search.

Technical report: https://vadymzelenko.github.io/chess-engine/
Play in the browser: https://vadymzelenko.github.io/chess-engine/game_site/

## Classical engine

**Evaluation.** Piece-square tables for 6 piece types, 64 squares and 2 game phases: 768 parameters. The score is interpolated between middlegame and endgame weights by material phase (tapered eval). Black pieces are read through a mirrored board, which halves the parameter count and keeps the evaluation symmetric.

**Training (Texel tuning).** Positions from Lichess games of 2400+ rated players with engine evaluations are converted to sparse feature vectors. The weights are optimised with Adam on a sparse design matrix to minimise the mean squared error between the sigmoid of our evaluation and the sigmoid of the Stockfish evaluation stored in the games. The published weights were trained on 3 million positions (final MSE 0.0104).

**Search.**

| Component | Description |
|---|---|
| Negamax with alpha-beta | Main search with iterative deepening and time control |
| Transposition table | Zobrist hashing, EXACT, LOWER and UPPER bounds, mate score correction, FIFO eviction |
| Quiescence search | Extends captures to avoid the horizon effect |
| Move ordering | TT move, then captures by MVV-LVA, then killer moves stored per ply |

```python
import json
import chess
import numpy as np
from engine import Engine

weights = np.array(json.load(open("weights.json"))["weights"])
engine = Engine(weights=weights)
best_move, score = engine.search(chess.Board(), time_limit=2.0, max_depth=64)
```

## Neural engine

**DualHeadResNet:** a convolutional network with 16 residual blocks, 256 channels and squeeze-and-excitation attention.

- Input: 54 planes (12 for the current pieces, side to move, 4 castling rights, en passant, and the 3 previous positions)
- Policy head: 4672 logits (73 move types for each of 64 squares, AlphaZero encoding)
- Value head: a scalar in [-1, 1]

**Search:** MCTS with PUCT. Leaves are evaluated in batches of 16 using virtual loss, which keeps the GPU busy instead of evaluating one position at a time.

**UCI wrapper (`uci_engine.py`):** plays through any UCI GUI or `lichess-bot`. It calibrates the number of simulations to the measured speed after every move, keeps the position history the network needs, picks the device automatically (Apple MPS, CUDA, CPU) and never lets a single malformed command crash the process.

| Variable | Default | Meaning |
|---|---|---|
| `CHESS_MODEL_PATH` | `chess_dualhead_best.pth` | Model checkpoint |
| `CHESS_ENGINE_DEVICE` | `auto` | `cpu`, `mps`, `cuda` or `auto` |
| `CHESS_N_SIMULATIONS` | `400` | MCTS simulations per move |
| `CHESS_MCTS_BATCH_SIZE` | `16` | Virtual loss batch |
| `CHESS_C_PUCT` | `1.5` | Exploration constant |
| `CHESS_TEMPERATURE` | `0.0` | 0 always plays the best move |

```bash
python uci_engine.py bench                 # MCTS speed on this machine
python uci_engine.py eval "<FEN>"          # evaluate one position
python uci_engine.py                       # UCI loop
```

**Training (`train_dual.py`):** supervised learning on Lichess games. The policy target is the move played; the value target is `tanh(cp / 400)` from the `[%eval]` annotations, or the game result when a game has none. Mixed precision on CUDA, AdamW with a one-cycle schedule, validation move accuracy and value error after each epoch, resumable checkpoints.

```bash
python train_dual.py prepare lichess-2400.pgn.zst --out data/ --max-positions 5000000 --min-elo 2400
python train_dual.py train data/ --epochs 3 --out chess_dualhead_best.pth
CHESS_MODEL_PATH=chess_dualhead_best.pth python uci_engine.py
```

The input and policy encodings are documented at the top of `chess_engine.py`. `load_model` also accepts checkpoints from the earlier training script when every tensor shape matches; a checkpoint with a different layout is reported with the first mismatching tensor.

## Repository structure

| File | Purpose |
|---|---|
| `features.py` | Feature extraction and tapered evaluation |
| `engine.py` | Classical alpha-beta engine |
| `weights.json` | Trained Texel weights (indices 0 to 383 middlegame, 384 to 767 endgame) |
| `prepare_data_pgn.py` | Lichess PGN (`.pgn.zst`) with evaluations to JSON Lines |
| `data_compile.py`, `data_compile.cpp` | PGN to CSV with phase stratification (Python and an experimental C++ version) |
| `tuner.py` | Texel tuning with Adam |
| `chess_engine.py` | Neural engine: input and move encoding, DualHeadResNet, batched MCTS, checkpoint loading |
| `train_dual.py` | Data preparation and training for the neural engine |
| `uci_engine.py` | UCI wrapper for the neural engine |
| `tests/` | Tests for the neural engine (`python -m pytest tests/`) |
| `claude_engine.py` | Experimental improved alpha-beta engine with a FastAPI interface |
| `play.py` | UCI loop for the classical engine |
| `web_app.py` | FastAPI web interface, http://127.0.0.1:8000 |
| `server.py` | Flask web interface with undo, http://127.0.0.1:5050 |
| `vs_stockfish.py` | Strength test against Stockfish |
| `config.yml` | `lichess-bot` configuration |
| `texel.ipynb`, `chessAI.ipynb` | Training notebooks (Texel tuning in Colab, neural network training) |
| `index.html`, `game_site/` | Technical report page and the browser version of the engine |

## Getting started

```bash
pip install -r requirements.txt

# Play against the classical engine in the browser
python server.py

# Test strength against Stockfish (requires the stockfish binary)
python vs_stockfish.py
```

Train your own weights from a Lichess database export with evaluations:

```bash
python prepare_data_pgn.py lichess-2400-eval.pgn.zst --out train.jsonl --limit 1000000
python tuner.py train.jsonl --out weights.json --epochs 30 --init-material
```

### Playing on Lichess

1. Create a bot account and a token at https://lichess.org/account/oauth/token.
2. Put the token into `config.yml` (`token:`) and set the engine directory. Do not commit the real token.
3. Run the engine through [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot).

## Requirements

- Python 3.10+
- Optional: Stockfish for strength tests, `zstd` to unpack Lichess archives
- Neural training needs a CUDA GPU (NVIDIA T4 or better recommended); inference runs on CPU with 8 GB RAM or more

## License

[MIT](LICENSE) © Vadym Zelenko. Lichess data is released under CC0.

References: [Chess Programming Wiki](https://www.chessprogramming.org/), [Lichess Database](https://database.lichess.org/), [AlphaZero paper](https://arxiv.org/abs/1712.01815), [Leela Chess Zero](https://lczero.org/).
