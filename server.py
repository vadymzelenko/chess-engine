# server.py
# Запускает Flask-сервер и отдает современный React/Tailwind интерфейс с поддержкой drag-and-drop и отмены ходов.
# Обновлен дизайн доски: классический зелено-белый стиль (Chess.com).
import os
import json
import webbrowser
from threading import Timer

import numpy as np
import chess
from flask import Flask, request, jsonify, render_template_string
from engine import Engine

app = Flask(__name__)
board = chess.Board()

# Инициализация движка и загрузка весов
WEIGHTS_PATH = "weights.json"
weights = None
if os.path.exists(WEIGHTS_PATH):
    with open(WEIGHTS_PATH, "r") as f:
        weights = np.array(json.load(f)["weights"], dtype=np.float64)

chess_engine = Engine(weights=weights)


@app.route("/")
def index():
    return render_template_string(HTML_CONTENT)


@app.route("/api/move", methods=["POST"])
def player_move():
    data = request.json
    move_str = data.get("move")

    try:
        move = chess.Move.from_uci(move_str)
        # Автоматическое превращение в ферзя
        if move not in board.legal_moves:
            move = chess.Move.from_uci(move_str + "q")

        if move in board.legal_moves:
            board.push(move)
            return jsonify({"fen": board.fen(), "legal": True, "game_over": board.is_game_over()})
    except ValueError:
        pass

    return jsonify({"fen": board.fen(), "legal": False, "game_over": board.is_game_over()})


@app.route("/api/engine_move", methods=["POST"])
def engine_move():
    if not board.is_game_over():
        best_move, _ = chess_engine.search(board, time_limit=1.0, max_depth=64)
        if best_move:
            board.push(best_move)
    return jsonify({"fen": board.fen(), "game_over": board.is_game_over()})


@app.route("/api/undo", methods=["POST"])
def undo_move():
    # Отменяем 2 хода (движка и игрока), если возможно
    if len(board.move_stack) >= 2:
        board.pop()
        board.pop()
    elif len(board.move_stack) == 1:
        board.pop()
    return jsonify({"fen": board.fen(), "game_over": board.is_game_over()})


@app.route("/api/reset", methods=["POST"])
def reset():
    board.reset()
    chess_engine.tt = {}
    return jsonify({"fen": board.fen()})


HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Chess vs Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: { sans: ['Inter', 'sans-serif'] },
                    colors: {
                        neutral: { 750: '#2d2d2d', 850: '#1f1f1f', 950: '#0a0a0a' },
                        chess: { light: '#ebecd0', dark: '#739552', highlight: '#f4f680' }
                    }
                }
            }
        }
    </script>
    <style>
        body { background-color: #0a0a0a; color: #ededed; overflow-x: hidden; touch-action: manipulation; }
        .piece { filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3)); }
        .dragging { opacity: 0.5; }
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

        const UndoIcon = () => (
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/></svg>
        );

        const ResetIcon = () => (
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
        );

        function App() {
            const [fen, setFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
            const [selected, setSelected] = useState(null);
            const [isThinking, setIsThinking] = useState(false);
            const [gameOver, setGameOver] = useState(false);
            const [draggedSq, setDraggedSq] = useState(null);

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

            const attemptMove = async (moveStr) => {
                if (isThinking || gameOver) return;

                const res = await fetch('/api/move', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ move: moveStr })
                });
                const data = await res.json();

                if (data.legal) {
                    setFen(data.fen);
                    setGameOver(data.game_over);
                    setSelected(null);

                    if (!data.game_over) {
                        setIsThinking(true);
                        const engRes = await fetch('/api/engine_move', { method: 'POST' });
                        const engData = await engRes.json();
                        setFen(engData.fen);
                        setGameOver(engData.game_over);
                        setIsThinking(false);
                    }
                } else {
                    setSelected(null);
                }
            };

            const handleSquareClick = (r, c) => {
                const sq = getSquare(r, c);
                const board = parseFen(fen);

                if (!selected) {
                    if (board[r][c]) setSelected(sq);
                } else {
                    if (selected === sq) {
                        setSelected(null);
                    } else {
                        attemptMove(selected + sq);
                    }
                }
            };

            const handleDragStart = (e, r, c) => {
                const sq = getSquare(r, c);
                setDraggedSq(sq);
                setSelected(sq);
                e.dataTransfer.effectAllowed = "move";
                const img = new Image();
                img.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
                e.dataTransfer.setDragImage(img, 0, 0);
            };

            const handleDragOver = (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = "move";
            };

            const handleDrop = (e, r, c) => {
                e.preventDefault();
                const targetSq = getSquare(r, c);
                setDraggedSq(null);

                if (draggedSq && draggedSq !== targetSq) {
                    attemptMove(draggedSq + targetSq);
                }
            };

            const handleUndo = async () => {
                if (isThinking) return;
                const res = await fetch('/api/undo', { method: 'POST' });
                const data = await res.json();
                setFen(data.fen);
                setGameOver(data.game_over);
                setSelected(null);
            };

            const handleReset = async () => {
                if (isThinking) return;
                const res = await fetch('/api/reset', { method: 'POST' });
                const data = await res.json();
                setFen(data.fen);
                setGameOver(false);
                setSelected(null);
            };

            const board2D = parseFen(fen);

            return (
                <div className="min-h-screen flex items-center justify-center p-4 sm:p-8 selection:bg-neutral-800">
                    <div className="flex flex-col lg:flex-row gap-8 lg:gap-12 items-center lg:items-start max-w-5xl w-full">

                        {/* Доска */}
                        <div className="grid grid-cols-8 grid-rows-8 w-full max-w-[500px] aspect-square rounded-sm overflow-hidden shadow-2xl border-4 border-neutral-800">
                            {board2D.map((row, r) => row.map((piece, c) => {
                                const isDark = (r + c) % 2 === 1;
                                const sq = getSquare(r, c);
                                const isSelected = selected === sq;
                                const isDragged = draggedSq === sq;

                                // Цвета полей в стиле Chess.com
                                const bgClass = isSelected ? 'bg-chess-highlight' : (isDark ? 'bg-chess-dark' : 'bg-chess-light');
                                const textClass = isDark ? 'text-chess-light/80' : 'text-chess-dark/80';

                                return (
                                    <div 
                                        key={sq}
                                        onClick={() => handleSquareClick(r, c)}
                                        onDragOver={handleDragOver}
                                        onDrop={(e) => handleDrop(e, r, c)}
                                        className={`
                                            relative flex justify-center items-center cursor-pointer transition-colors duration-75
                                            ${bgClass} ${textClass}
                                        `}
                                    >
                                        {c === 0 && <span className="absolute top-1 left-1 text-[11px] font-bold pointer-events-none">{8 - r}</span>}
                                        {r === 7 && <span className="absolute bottom-0.5 right-1 text-[11px] font-bold pointer-events-none">{files[c]}</span>}

                                        {piece && (
                                            <img 
                                                src={PIECE_IMAGES[piece]} 
                                                alt={piece} 
                                                draggable
                                                onDragStart={(e) => handleDragStart(e, r, c)}
                                                onDragEnd={() => setDraggedSq(null)}
                                                className={`w-[85%] h-[85%] object-contain piece ${isDragged ? 'dragging' : ''}`}
                                            />
                                        )}
                                    </div>
                                )
                            }))}
                        </div>

                        {/* Панель управления */}
                        <div className="flex flex-col gap-6 w-full lg:max-w-[320px]">
                            <div>
                                <h1 className="text-3xl font-semibold tracking-tight text-white">Engine UI</h1>
                            </div>

                            <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 backdrop-blur-xl flex flex-col gap-5 shadow-lg">
                                <div className="flex items-center gap-3 bg-neutral-950 p-3 rounded-xl border border-neutral-800">
                                    <div className={`w-2.5 h-2.5 rounded-full shadow-sm ${gameOver ? 'bg-red-500' : (isThinking ? 'bg-orange-500 animate-pulse' : 'bg-green-500')}`} />
                                    <span className="font-medium text-sm text-neutral-200">
                                        {gameOver ? 'Мат / Ничья' : (isThinking ? 'Анализ позиции...' : 'Ваш ход (белые)')}
                                    </span>
                                </div>

                                <div className="flex gap-3">
                                    <button 
                                        onClick={handleUndo}
                                        disabled={isThinking || fen === 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'}
                                        className="flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl border border-neutral-700 bg-neutral-800 hover:bg-neutral-700 text-white font-medium text-sm transition-all active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none"
                                    >
                                        <UndoIcon />
                                        Назад
                                    </button>

                                    <button 
                                        onClick={handleReset}
                                        disabled={isThinking}
                                        className="flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-white hover:bg-neutral-200 text-neutral-950 font-medium text-sm transition-all active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none shadow-sm"
                                    >
                                        <ResetIcon />
                                        Сброс
                                    </button>
                                </div>
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
    webbrowser.open("http://127.0.0.1:5050")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    app.run(host="0.0.0.0", port=5050, debug=False)