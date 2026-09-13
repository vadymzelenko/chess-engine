import os
import math
import time
import webbrowser
from threading import Timer
from dataclasses import dataclass
from typing import Optional, List, Tuple

import uvicorn
import torch
import torch.nn as nn
import numpy as np
import chess
import chess.polyglot
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()
board = chess.Board()

# ============================== НАСТРОЙКИ ДВИЖКА ==============================
MODEL_PATH = "lichess-bot-master/chess_resnet_best.pth"
DEVICE = "cpu"
SCORE_SCALE_DEFAULT = 400.0
CHANNELS_DEFAULT = 128
BLOCKS_DEFAULT = 8
PLANES = 18

# Глубина/время — то, что раньше было depth=2 в minimax. Тут это мягкий лимит:
# движок идёт итеративным углублением и отдаёт лучший ход из последней
# ПОЛНОСТЬЮ досчитанной глубины, если время истекло на середине следующей.
ENGINE_MAX_DEPTH = 4
ENGINE_TIME_LIMIT_SEC = 3.0  # None -> считать строго до ENGINE_MAX_DEPTH без лимита времени

PIECE_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}


# ============================== МОДЕЛЬ ==============================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, channels, se_reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + residual)


class ChessResNet(nn.Module):
    def __init__(self, in_planes=PLANES, channels=CHANNELS_DEFAULT, num_blocks=BLOCKS_DEFAULT):
        super().__init__()
        self.conv_input = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)
        return self.value_head(x)


def board_to_planes(b: chess.Board) -> np.ndarray:
    t = np.zeros((PLANES, 8, 8), dtype=np.float32)
    for square, piece in b.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        t[PIECE_MAP[piece.symbol()], row, col] = 1.0
    if b.turn == chess.WHITE:
        t[12, :, :] = 1.0
    if b.has_kingside_castling_rights(chess.WHITE):
        t[13, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE):
        t[14, :, :] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):
        t[15, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK):
        t[16, :, :] = 1.0
    if b.ep_square is not None:
        row = 7 - chess.square_rank(b.ep_square)
        col = chess.square_file(b.ep_square)
        t[17, row, col] = 1.0
    return t


class NeuralEvaluator:
    def __init__(self, model_path, device="cpu", score_scale=SCORE_SCALE_DEFAULT,
                 channels=CHANNELS_DEFAULT, num_blocks=BLOCKS_DEFAULT):
        self.device = torch.device(device)
        self.score_scale = score_scale
        self.model = ChessResNet(PLANES, channels, num_blocks).to(self.device)
        self.cache = {}

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                config = checkpoint.get("config", {})
                self.score_scale = config.get("score_scale", self.score_scale)
                # архитектура фиксирована при конструировании self.model выше;
                # если конфиг отличается по channels/num_blocks — пересоберём модель
                if config.get("channels", channels) != channels or config.get("num_blocks", num_blocks) != num_blocks:
                    channels = config.get("channels", channels)
                    num_blocks = config.get("num_blocks", num_blocks)
                    self.model = ChessResNet(PLANES, channels, num_blocks).to(self.device)
            else:
                state_dict = checkpoint
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
            print(f"Веса успешно загружены из {model_path} (score_scale={self.score_scale})")
        else:
            print(f"⚠️ Файл весов {model_path} не найден! Бот делает случайные оценки.")

        self.model.eval()

    @torch.no_grad()
    def eval_white_cp(self, b: chess.Board, zkey: int) -> float:
        cached = self.cache.get(zkey)
        if cached is not None:
            return cached
        planes = board_to_planes(b)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        out = self.model(x).item()
        out = max(-0.999, min(0.999, out))
        cp = math.atanh(out) * self.score_scale
        self.cache[zkey] = cp
        return cp

    def eval_side_to_move_cp(self, b: chess.Board, zkey: int) -> float:
        white_cp = self.eval_white_cp(b, zkey)
        return white_cp if b.turn == chess.WHITE else -white_cp


# ============================== ПОИСК (PVS + quiescence + TT) ==============================

MATE_VALUE = 32000
INF = 10 ** 9
PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
}
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2


@dataclass
class TTEntry:
    depth: int
    value: float
    flag: int
    best_move: Optional[chess.Move]


class SearchEngine:
    """PVS поверх alpha-beta: TT-move -> MVV-LVA захваты -> killer -> history
    для сортировки, quiescence на взятиях в листьях, итеративное углубление
    с мягким лимитом по времени. То же, что в chess_engine_search.py, здесь
    встроено прямо в веб-сервер."""

    def __init__(self, evaluator: NeuralEvaluator, tt_size_mb: int = 256):
        self.evaluator = evaluator
        self.tt = {}
        self.max_tt_entries = (tt_size_mb * 1024 * 1024) // 64
        self.killers: List[List[Optional[chess.Move]]] = []
        self.history = {}
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = None
        self.stop_search = False

    def _zkey(self, b):
        return chess.polyglot.zobrist_hash(b)

    def _time_up(self):
        if self.time_limit is None:
            return False
        if self.nodes % 2048 == 0 and (time.time() - self.start_time) > self.time_limit:
            self.stop_search = True
        return self.stop_search

    def _mvv_lva(self, b, move):
        victim = b.piece_type_at(move.to_square)
        if victim is None:
            victim = chess.PAWN if b.is_en_passant(move) else 0
        attacker = b.piece_type_at(move.from_square)
        return PIECE_VALUE.get(victim, 0) * 100 - PIECE_VALUE.get(attacker, 0)

    def _order_moves(self, b, ply, tt_move):
        moves = list(b.legal_moves)
        killers = self.killers[ply] if ply < len(self.killers) else [None, None]

        def score(m):
            if tt_move is not None and m == tt_move:
                return 10_000_000
            if b.is_capture(m):
                return 1_000_000 + self._mvv_lva(b, m)
            if m.promotion == chess.QUEEN:
                return 900_000
            if killers[0] == m:
                return 800_000
            if killers[1] == m:
                return 700_000
            return self.history.get((m.from_square, m.to_square), 0)

        moves.sort(key=score, reverse=True)
        return moves

    def _record_killer(self, ply, move):
        while len(self.killers) <= ply:
            self.killers.append([None, None])
        if self.killers[ply][0] != move:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def quiescence(self, b, alpha, beta, ply, qdepth=0, max_qdepth=8):
        self.nodes += 1
        if self._time_up():
            return alpha
        zkey = self._zkey(b)
        stand_pat = self.evaluator.eval_side_to_move_cp(b, zkey)
        if qdepth >= max_qdepth:
            return stand_pat
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        moves = [m for m in b.legal_moves if b.is_capture(m) or m.promotion]
        moves.sort(key=lambda m: self._mvv_lva(b, m), reverse=True)
        for move in moves:
            victim = b.piece_type_at(move.to_square)
            victim_val = PIECE_VALUE.get(victim, 0) if victim else 0
            if stand_pat + victim_val + 200 < alpha and not move.promotion:
                continue
            b.push(move)
            value = -self.quiescence(b, -beta, -alpha, ply + 1, qdepth + 1, max_qdepth)
            b.pop()
            if self.stop_search:
                return alpha
            if value >= beta:
                return value
            if value > alpha:
                alpha = value
        return alpha

    def negamax(self, b, depth, alpha, beta, ply):
        self.nodes += 1
        if self._time_up():
            return alpha
        if b.is_checkmate():
            return -(MATE_VALUE - ply)
        if b.is_stalemate() or b.is_insufficient_material() or \
                b.can_claim_threefold_repetition() or b.can_claim_fifty_moves():
            return 0

        zkey = self._zkey(b)
        entry = self.tt.get(zkey)
        tt_move = None
        if entry is not None:
            tt_move = entry.best_move
            if entry.depth >= depth:
                if entry.flag == EXACT:
                    return entry.value
                elif entry.flag == LOWERBOUND:
                    alpha = max(alpha, entry.value)
                elif entry.flag == UPPERBOUND:
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value

        if depth <= 0:
            return self.quiescence(b, alpha, beta, ply)

        orig_alpha = alpha
        moves = self._order_moves(b, ply, tt_move)
        best_value = -INF
        best_move = None

        for i, move in enumerate(moves):
            b.push(move)
            if i == 0:
                value = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1)
            else:
                value = -self.negamax(b, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < value < beta:
                    value = -self.negamax(b, depth - 1, -beta, -alpha, ply + 1)
            b.pop()

            if self.stop_search:
                break
            if value > best_value:
                best_value = value
                best_move = move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                if not b.is_capture(move):
                    self._record_killer(ply, move)
                self.history[(move.from_square, move.to_square)] = \
                    self.history.get((move.from_square, move.to_square), 0) + depth * depth
                break

        if not self.stop_search:
            flag = UPPERBOUND if best_value <= orig_alpha else (LOWERBOUND if best_value >= beta else EXACT)
            if len(self.tt) > self.max_tt_entries:
                self.tt.clear()
            self.tt[zkey] = TTEntry(depth, best_value, flag, best_move)

        return best_value

    def search(self, b, max_depth=8, time_limit=None):
        self.nodes = 0
        self.stop_search = False
        self.time_limit = time_limit
        self.start_time = time.time()
        self.killers = [[None, None] for _ in range(max_depth + 32)]
        self.history.clear()

        best_move, best_value = None, 0.0
        for depth in range(1, max_depth + 1):
            value = self.negamax(b, depth, -INF, INF, 0)
            if self.stop_search and depth > 1:
                break
            entry = self.tt.get(self._zkey(b))
            if entry and entry.best_move is not None:
                best_move, best_value = entry.best_move, value
            if self.stop_search:
                break
            if abs(best_value) >= MATE_VALUE - 1000:
                break
        return best_move, best_value


# Инициализация движка
evaluator = NeuralEvaluator(MODEL_PATH, device=DEVICE)
search_engine = SearchEngine(evaluator)


# ============================== API ENDPOINTS ==============================

class MoveReq(BaseModel):
    move: str


@app.post("/api/move")
def player_move(req: MoveReq):
    try:
        move = chess.Move.from_uci(req.move)
        if move not in board.legal_moves:
            move = chess.Move.from_uci(req.move + "q")
        if move in board.legal_moves:
            board.push(move)
            return {"fen": board.fen(), "legal": True, "game_over": board.is_game_over()}
    except ValueError:
        pass
    return {"fen": board.fen(), "legal": False, "game_over": board.is_game_over()}


@app.post("/api/engine_move")
def engine_move():
    if not board.is_game_over():
        best_move, score = search_engine.search(
            board, max_depth=ENGINE_MAX_DEPTH, time_limit=ENGINE_TIME_LIMIT_SEC
        )
        if best_move:
            board.push(best_move)
            print(f"Ход бота: {best_move} | Оценка: {score:.1f} cp | узлов: {search_engine.nodes}")
    return {"fen": board.fen(), "game_over": board.is_game_over()}


@app.post("/api/reset")
def reset():
    board.reset()
    search_engine.tt.clear()
    evaluator.cache.clear()
    return {"fen": board.fen()}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


# ============================== ФРОНТЕНД (без изменений) ==============================
# Полная замена интерфейса на чистый JS + SVG из GitHub lichess/lila без проблемных внешних библиотек
HTML_CONTENT = """<!DOCTYPE html>
<html lang="ru" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ResNet Neural Chess</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .board-grid {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      grid-template-rows: repeat(8, minmax(0, 1fr));
    }
    .square-light { background-color: #dee3e6; }
    .square-dark { background-color: #8ca2ad; }
    .square-selected { background-color: #bbc958 !important; }
    .square-last-move { background-color: #cedb72 !important; }
    .square-hint::after {
      content: '';
      position: absolute;
      width: 28%;
      height: 28%;
      background: rgba(0, 0, 0, 0.22);
      border-radius: 9999px;
      pointer-events: none;
    }
    .square-capture-hint::after {
      content: '';
      position: absolute;
      inset: 0;
      border: 5px solid rgba(0, 0, 0, 0.22);
      border-radius: 9999px;
      pointer-events: none;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-4 selection:bg-blue-500/30">
  <div class="w-full max-w-5xl bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl shadow-black/60">

    <!-- Header -->
    <header class="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 mb-6 border-b border-slate-800 gap-4">
      <div>
        <div class="flex items-center gap-2">
          <h1 class="text-xl font-bold tracking-tight text-white">ResNet Chess</h1>
          <span class="text-xs px-2 py-0.5 rounded-full font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">MPS/CPU Engine</span>
        </div>
        <p class="text-xs text-slate-400 mt-1">PVS + Quiescence + ResNet Value Head</p>
      </div>

      <button id="reset-btn" class="w-full sm:w-auto px-4 py-2 text-sm font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 active:scale-95 transition rounded-lg border border-slate-700">
        Новая партия
      </button>
    </header>

    <!-- Main Content -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

      <!-- Chess Board Container -->
      <div class="lg:col-span-8 flex flex-col items-center">
        <div class="w-full max-w-[540px] aspect-square relative border border-slate-700 rounded-xl overflow-hidden shadow-2xl bg-slate-800 select-none">
          <div id="board" class="w-full h-full board-grid relative"></div>
        </div>
      </div>

      <!-- Control Panel -->
      <div class="lg:col-span-4 flex flex-col gap-4">

        <!-- Status -->
        <div class="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
          <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Статус</h2>
          <div id="status-text" class="text-base font-medium text-slate-100 flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            Ваш ход (Белые)
          </div>
        </div>

        <!-- Evaluation Bar -->
        <div class="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
          <div class="flex justify-between items-center mb-2">
            <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Оценка позиции</span>
            <span id="eval-value" class="text-sm font-bold font-mono text-slate-200">0.00</span>
          </div>
          <div class="w-full bg-slate-700 rounded-full h-2.5 overflow-hidden flex">
            <div id="eval-bar" class="bg-blue-500 h-full transition-all duration-300" style="width: 50%"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
            <span>Черные (-10)</span>
            <span>0.0</span>
            <span>Белые (+10)</span>
          </div>
        </div>

        <!-- Last Engine Move -->
        <div class="bg-slate-800/60 border border-slate-700/50 rounded-xl p-4">
          <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Последний ход движка</span>
          <p id="engine-move-text" class="text-sm font-mono text-slate-300 mt-1">—</p>
        </div>

        <div class="text-[11px] text-slate-500 text-center mt-2">
          Игра против обученной сверточной сети ResNet с SE-блоками.
        </div>
      </div>

    </div>
  </div>

  <script>
    const PIECE_IMAGES = {
      'P': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/wP.svg',
      'N': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/wN.svg',
      'B': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/wB.svg',
      'R': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/wR.svg',
      'Q': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/wQ.svg',
      'K': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/wK.svg',
      'p': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/bP.svg',
      'n': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/bN.svg',
      'b': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/bB.svg',
      'r': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/bR.svg',
      'q': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/bQ.svg',
      'k': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/bK.svg'
    };

    const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const RANKS = ['8', '7', '6', '5', '4', '3', '2', '1'];

    let currentFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let selectedSquare = null;
    let isThinking = false;
    let lastMoveSquares = [];

    const boardEl = document.getElementById('board');
    const statusText = document.getElementById('status-text');
    const evalValue = document.getElementById('eval-value');
    const evalBar = document.getElementById('eval-bar');
    const engineMoveText = document.getElementById('engine-move-text');
    const resetBtn = document.getElementById('reset-btn');

    function fenToPieceMap(fen) {
      const map = {};
      const placement = fen.split(' ')[0];
      const rows = placement.split('/');
      for (let r = 0; r < 8; r++) {
        let f = 0;
        for (const ch of rows[r]) {
          if (!isNaN(ch)) {
            f += parseInt(ch, 10);
          } else {
            const sq = FILES[f] + RANKS[r];
            map[sq] = ch;
            f++;
          }
        }
      }
      return map;
    }

    function updateEval(cp) {
      const clamped = Math.max(-1000, Math.min(1000, cp));
      const percentage = ((clamped + 1000) / 2000) * 100;
      evalBar.style.width = percentage + '%';
      const pawns = (cp / 100).toFixed(2);
      evalValue.textContent = (cp > 0 ? '+' : '') + pawns;
    }

    function renderBoard() {
      boardEl.innerHTML = '';
      const pieceMap = fenToPieceMap(currentFen);

      for (let r = 0; r < 8; r++) {
        for (let f = 0; f < 8; f++) {
          const sq = FILES[f] + RANKS[r];
          const isLight = (r + f) % 2 === 0;
          const squareDiv = document.createElement('div');

          squareDiv.dataset.square = sq;
          squareDiv.className = `relative flex items-center justify-center cursor-pointer transition-colors ${
            isLight ? 'square-light' : 'square-dark'
          }`;

          if (lastMoveSquares.includes(sq)) {
            squareDiv.classList.add('square-last-move');
          }

          if (selectedSquare === sq) {
            squareDiv.classList.add('square-selected');
          }

          const piece = pieceMap[sq];
          if (piece) {
            const img = document.createElement('img');
            img.src = PIECE_IMAGES[piece];
            img.alt = piece;
            img.className = 'w-[85%] h-[85%] pointer-events-none drop-shadow-md select-none';
            squareDiv.appendChild(img);
          }

          // Coordinate labels
          if (f === 0) {
            const rankLabel = document.createElement('span');
            rankLabel.className = `absolute top-1 left-1 text-[10px] font-bold ${isLight ? 'text-[#8ca2ad]' : 'text-[#dee3e6]'}`;
            rankLabel.textContent = RANKS[r];
            squareDiv.appendChild(rankLabel);
          }
          if (r === 7) {
            const fileLabel = document.createElement('span');
            fileLabel.className = `absolute bottom-1 right-1 text-[10px] font-bold ${isLight ? 'text-[#8ca2ad]' : 'text-[#dee3e6]'}`;
            fileLabel.textContent = FILES[f];
            squareDiv.appendChild(fileLabel);
          }

          squareDiv.addEventListener('click', () => handleSquareClick(sq, piece));
          boardEl.appendChild(squareDiv);
        }
      }
    }

    async function handleSquareClick(sq, piece) {
      if (isThinking) return;

      const pieceMap = fenToPieceMap(currentFen);

      if (!selectedSquare) {
        // Выбор своей белой фигуры
        if (piece && piece === piece.toUpperCase()) {
          selectedSquare = sq;
          renderBoard();
        }
        return;
      }

      if (selectedSquare === sq) {
        selectedSquare = null;
        renderBoard();
        return;
      }

      // Если кликнули по другой своей фигуре — переключаем выбор
      if (piece && piece === piece.toUpperCase()) {
        selectedSquare = sq;
        renderBoard();
        return;
      }

      // Попытка хода
      const fromSq = selectedSquare;
      const toSq = sq;
      const movingPiece = pieceMap[fromSq];
      const isPromotion = (movingPiece === 'P') && toSq[1] === '8';
      const uciMove = fromSq + toSq + (isPromotion ? 'q' : '');

      selectedSquare = null;
      isThinking = true;
      statusText.innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse"></span>Проверка хода...';

      try {
        const res = await fetch('/api/move', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ move: uciMove })
        });
        const data = await res.json();

        if (!data.legal) {
          statusText.innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-rose-500"></span>Недопустимый ход';
          isThinking = false;
          renderBoard();
          return;
        }

        currentFen = data.fen;
        lastMoveSquares = [fromSq, toSq];
        updateEval(data.eval_cp);
        renderBoard();

        if (data.game_over) {
          statusText.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-red-500"></span>Игра окончена: ${data.result}`;
          isThinking = false;
          return;
        }

        // Запрос хода нейросети
        statusText.innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-blue-500 animate-spin"></span>Движок думает...';
        const engRes = await fetch('/api/engine_move', { method: 'POST' });
        const engData = await engRes.json();

        currentFen = engData.fen;
        if (engData.last_engine_move) {
          engineMoveText.textContent = engData.last_engine_move;
          const engFrom = engData.last_engine_move.substring(0, 2);
          const engTo = engData.last_engine_move.substring(2, 4);
          lastMoveSquares = [engFrom, engTo];
        }

        updateEval(engData.eval_cp);
        renderBoard();

        if (engData.game_over) {
          statusText.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-red-500"></span>Игра окончена: ${engData.result}`;
        } else {
          statusText.innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>Ваш ход (Белые)';
        }
      } catch (err) {
        console.error(err);
        statusText.innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-red-500"></span>Ошибка соединения';
      } finally {
        isThinking = false;
      }
    }

    resetBtn.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/reset', { method: 'POST' });
        const data = await res.json();
        currentFen = data.fen;
        selectedSquare = null;
        lastMoveSquares = [];
        updateEval(0);
        engineMoveText.textContent = '—';
        statusText.innerHTML = '<span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>Ваш ход (Белые)';
        renderBoard();
      } catch (err) {
        console.error(err);
      }
    });

    // Стартовый рендер
    renderBoard();
  </script>
</body>
</html>
"""


def open_browser():
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)