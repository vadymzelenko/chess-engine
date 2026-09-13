import os
import json
import webbrowser
from threading import Timer

import uvicorn
import numpy as np
import chess
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from engine import Engine

app = FastAPI()
board = chess.Board()

# Загрузка обученных весов, если они есть в текущей директории
weights_path = "../chess_engine_20260907_191818/weights.json"
if os.path.exists(weights_path):
    with open(weights_path, "r") as f:
        weights = np.array(json.load(f)["weights"], dtype=np.float64)
else:
    weights = None

chess_engine = Engine(weights=weights)


class MoveReq(BaseModel):
    move: str


@app.post("/api/move")
def player_move(req: MoveReq):
    try:
        move = chess.Move.from_uci(req.move)
        # Автоматическое превращение в ферзя (q), если это ход пешкой на последнюю горизонталь
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
        best_move, _ = chess_engine.search(board, time_limit=1.5, max_depth=64)
        if best_move:
            board.push(best_move)
    return {"fen": board.fen(), "game_over": board.is_game_over()}


@app.post("/api/reset")
def reset():
    board.reset()
    chess_engine.tt = {}
    return {"fen": board.fen()}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chess Engine UI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: { sans: ['Inter', 'sans-serif'] }
                }
            }
        }
    </script>
    <style>
        body { background-color: #0a0a0a; color: #ededed; }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect } = React;

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
    'k': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/bK.svg',
};

        function App() {
            const [fen, setFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
            const [selected, setSelected] = useState(null);
            const [isThinking, setIsThinking] = useState(false);
            const [gameOver, setGameOver] = useState(false);

            const parseFen = (fenStr) => {
                const rows = fenStr.split(' ')[0].split('/');
                return rows.map(row => {
                    const boardRow = [];
                    for (let char of row) {
                        if (!isNaN(char)) {
                            for (let i = 0; i < parseInt(char); i++) boardRow.push(null);
                        } else {
                            boardRow.push(char);
                        }
                    }
                    return boardRow;
                });
            };

            const files = ['a','b','c','d','e','f','g','h'];
            const getSquare = (r, c) => `${files[c]}${8 - r}`;

            const handleSquareClick = async (r, c) => {
                if (isThinking || gameOver) return;

                const sq = getSquare(r, c);

                if (!selected) {
                    const board = parseFen(fen);
                    if (board[r][c]) setSelected(sq);
                } else {
                    if (selected === sq) {
                        setSelected(null);
                        return;
                    }

                    const moveStr = selected + sq;
                    setSelected(null);

                    const res = await fetch('/api/move', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ move: moveStr })
                    });
                    const data = await res.json();
                    setFen(data.fen);
                    setGameOver(data.game_over);

                    if (data.legal && !data.game_over) {
                        setIsThinking(true);
                        const engRes = await fetch('/api/engine_move', { method: 'POST' });
                        const engData = await engRes.json();
                        setFen(engData.fen);
                        setGameOver(engData.game_over);
                        setIsThinking(false);
                    }
                }
            };

            const handleReset = async () => {
                const res = await fetch('/api/reset', { method: 'POST' });
                const data = await res.json();
                setFen(data.fen);
                setGameOver(false);
                setSelected(null);
                setIsThinking(false);
            };

            const board2D = parseFen(fen);

            return (
                <div className="min-h-screen flex items-center justify-center p-4">
                    <div className="flex flex-col md:flex-row gap-10 items-center md:items-start max-w-4xl w-full">

                        <div className="grid grid-cols-8 grid-rows-8 w-full max-w-[440px] aspect-square rounded-xl overflow-hidden shadow-[0_8px_30px_rgb(0,0,0,0.4)] border border-neutral-800">
                            {board2D.map((row, r) => row.map((piece, c) => {
                                const isDark = (r + c) % 2 === 1;
                                const sq = getSquare(r, c);
                                const isSelected = selected === sq;

                                return (
                                    <div 
                                        key={sq}
                                        onClick={() => handleSquareClick(r, c)}
                                        className={`
                                            relative flex justify-center items-center cursor-pointer transition-colors
                                            ${isDark ? 'bg-neutral-700' : 'bg-neutral-300'}
                                            ${isSelected ? 'ring-4 ring-inset ring-yellow-500 bg-yellow-500/40' : ''}
                                            hover:opacity-90
                                        `}
                                    >
                                        {c === 0 && <span className="absolute top-1 left-1 text-[10px] font-semibold opacity-40 text-neutral-900 pointer-events-none">{8 - r}</span>}
                                        {r === 7 && <span className="absolute bottom-1 right-1 text-[10px] font-semibold opacity-40 text-neutral-900 pointer-events-none">{files[c]}</span>}

                                        {piece && (
                                            <img 
                                                src={PIECE_IMAGES[piece]} 
                                                alt={piece} 
                                                className="w-[85%] h-[85%] object-contain drop-shadow-md pointer-events-none"
                                            />
                                        )}
                                    </div>
                                )
                            }))}
                        </div>

                        <div className="flex flex-col gap-6 flex-1 min-w-[260px] w-full">
                            <div>
                                <h1 className="text-2xl font-semibold tracking-tight text-white mb-1">My Chess Engine</h1>
                                <p className="text-neutral-400 text-sm">Texel-tuned evaluation</p>
                            </div>

                            <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/40 flex flex-col gap-4 shadow-sm">
                                <div className="flex items-center gap-3">
                                    <div className={`w-2.5 h-2.5 rounded-full ${gameOver ? 'bg-red-500' : (isThinking ? 'bg-yellow-500 animate-pulse' : 'bg-green-500')}`} />
                                    <span className="font-medium text-sm text-neutral-200">
                                        {gameOver ? 'Мат / Ничья' : (isThinking ? 'Движок анализирует...' : 'Ваш ход (белые)')}
                                    </span>
                                </div>

                                <button 
                                    onClick={handleReset}
                                    disabled={isThinking}
                                    className="mt-2 flex items-center justify-center gap-2 w-full py-2.5 px-4 rounded-xl bg-neutral-100 hover:bg-white text-neutral-900 font-medium text-sm transition-all disabled:opacity-50"
                                >
                                    <svg className="w-4 h-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
                                    Перезапустить
                                </button>
                            </div>
                        </div>

                    </div>
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>
"""


def open_browser():
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)