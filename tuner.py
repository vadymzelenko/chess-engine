"""
Texel tuning: подбираем 768 весов (mg/eg PST) так, чтобы sigmoid(eval)
приближала sigmoid(stockfish_cp) на датасете позиций.

Вход: jsonl-файл со строками {"fen": "...", "cp": 123} (после prepare_data.py).
Выход: weights.json с обученными весами.

ВАЖНО про формат данных (частая причина "тюнер учится, а движок играет хуже"):
код ниже предполагает, что cp в датасете дано С ТОЧКИ ЗРЕНИЯ СТОРОНЫ, ЧЕЙ ХОД
(это конвенция, например, у lichess evaluation dumps). Если в вашем датасете cp
всегда с точки зрения белых — уберите инверсию в load_dataset(), иначе веса
обучатся на инвертированном сигнале и результат будет тихо сломан (без ошибок,
просто движок будет слабым/странным).

Производительность: раньше здесь был чистый Python-цикл по каждой позиции и
каждому активному признаку на КАЖДОЙ эпохе — это на несколько порядков медленнее,
чем нужно, и почти не выигрывает от мощного сервера (упирается в GIL/интерпретатор,
а не в CPU/RAM). Датасет один раз переводится в разреженную (sparse) матрицу
design-фичей, дальше вся эпоха — это пара sparse matmul через BLAS/scipy:
и на ноутбуке, и на сервере с большим числом ядер это использует их эффективно
(scipy/numpy сами параллелят BLAS-часть), и allows датасеты на десятки миллионов
позиций.

CHECKPOINTING (для Colab): обучение периодически сохраняет чекпоинт
(веса + состояние Adam m/v/t_step + номер эпохи) в --checkpoint-dir. При
--resume, если в этой директории уже есть чекпоинты, обучение продолжается
с последнего, а не с нуля — это переживает дисконнект/перезапуск среды
выполнения Colab. Дополнительно --out сохраняется на диск после каждого
чекпоинта (а не только в конце), чтобы веса были доступны даже если
обучение прервётся между чекпоинтами.
"""
import argparse
import datetime
import json
import os
import time

import numpy as np
from scipy import sparse

import chess
from features import extract_features, N_PARAMS, N_PIECES, N_SQUARES, MAX_PHASE, default_weights

K = 1.0 / 400.0  # масштаб для sigmoid (стандартный texel-scale)
EG_OFFSET = N_PIECES * N_SQUARES  # 384


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-K * x))


def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Загрузка данных и построение design-матрицы
# ---------------------------------------------------------------------------

def load_dataset(path: str, limit: int = None, n_workers: int = 1):
    """
    Читает jsonl {fen, cp}. Возвращает (X, targets):
      X       — scipy.sparse.csr_matrix формы (N, N_PARAMS), уже с фазовым
                масштабированием "внутри" (см. _row_from_features), так что
                predicted_raw = X @ weights напрямую воспроизводит tapered eval.
      targets — np.ndarray формы (N,), sigmoid(K * cp) с учётом стороны хода.

    n_workers > 1 включает параллельный разбор FEN через multiprocessing —
    полезно на сервере для датасетов в миллионы строк (разбор FEN — это
    единственная по-настоящему CPU-per-line часть пайплайна).
    """
    lines = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                lines.append(line)

    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            rows = pool.map(_parse_line, lines, chunksize=2048)
    else:
        rows = [_parse_line(l) for l in lines]

    rows = [r for r in rows if r is not None]

    n = len(rows)
    row_idx, col_idx, vals = [], [], []
    targets = np.empty(n, dtype=np.float64)

    for r, (feats, phase, target) in enumerate(rows):
        targets[r] = target
        add_mg, sub_mg, add_eg, sub_eg = feats
        mg_scale = phase / MAX_PHASE
        eg_scale = (MAX_PHASE - phase) / MAX_PHASE
        for i in add_mg:
            row_idx.append(r); col_idx.append(i); vals.append(mg_scale)
        for i in sub_mg:
            row_idx.append(r); col_idx.append(i); vals.append(-mg_scale)
        for i in add_eg:
            row_idx.append(r); col_idx.append(EG_OFFSET + i); vals.append(eg_scale)
        for i in sub_eg:
            row_idx.append(r); col_idx.append(EG_OFFSET + i); vals.append(-eg_scale)

    X = sparse.csr_matrix((vals, (row_idx, col_idx)), shape=(n, N_PARAMS))
    return X, targets


def _parse_line(line: str):
    rec = json.loads(line)
    fen, cp = rec["fen"], rec["cp"]
    board = chess.Board(fen)
    feats, phase = extract_features(board)
    # cp в датасете — с точки зрения стороны, чей ход (см. предупреждение в шапке файла).
    # Приводим к "с точки зрения белых", как и наш eval.
    if not board.turn:  # чёрные должны ходить -> инвертируем
        cp = -cp
    target = 1.0 / (1.0 + np.exp(-K * cp))
    return feats, phase, target


# ---------------------------------------------------------------------------
# Чекпоинты
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, epoch: int, w: np.ndarray, m: np.ndarray,
                     v: np.ndarray, t_step: int):
    # np.savez сам дописывает ".npz" к имени, если его нет — избегаем сюрпризов,
    # сохраняя во временный файл С таким же расширением и потом атомарно
    # переименовывая (os.replace) в целевой путь.
    assert path.endswith(".npz"), "checkpoint path must end with .npz"
    tmp_path = path[:-len(".npz")] + ".tmp.npz"
    np.savez(tmp_path, epoch=epoch, w=w, m=m, v=v, t_step=t_step)
    os.replace(tmp_path, path)


def load_checkpoint(path: str):
    d = np.load(path)
    return int(d["epoch"]), d["w"], d["m"], d["v"], int(d["t_step"])


def find_latest_checkpoint(checkpoint_dir: str):
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    candidates = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("ckpt_epoch_") and f.endswith(".npz")
    ]
    if not candidates:
        return None

    def epoch_of(fname):
        # ckpt_epoch_0007.npz -> 7
        return int(fname[len("ckpt_epoch_"):-len(".npz")])

    candidates.sort(key=epoch_of)
    return os.path.join(checkpoint_dir, candidates[-1])


# ---------------------------------------------------------------------------
# Обучение (векторизованный Adam поверх sparse-матрицы)
# ---------------------------------------------------------------------------

def train(X: sparse.csr_matrix, targets: np.ndarray, epochs=15, lr=1.0,
          batch_size=4096, init_weights=None, seed=0,
          checkpoint_dir: str = None, checkpoint_every: int = 1,
          resume: bool = False, weights_out_path: str = None):
    """
    Mini-batch Adam на 768 параметрах, MSE между sigmoid(eval) и target.
    Вся арифметика — sparse matmul по всему батчу разом (никакого
    python-цикла по позициям/фичам внутри эпохи).

    checkpoint_dir / checkpoint_every / resume: см. описание в шапке файла.
    weights_out_path: если задан, "человекочитаемые" веса (weights.json)
      дополнительно сохраняются туда после каждого чекпоинта — не только
      в самом конце обучения.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    w = (init_weights.copy() if init_weights is not None
         else np.zeros(N_PARAMS, dtype=np.float64))

    m = np.zeros(N_PARAMS)
    v = np.zeros(N_PARAMS)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t_step = 0
    start_epoch = 0

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if resume and checkpoint_dir:
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            start_epoch, w, m, v, t_step = load_checkpoint(latest)
            log(f"[resume] найден чекпоинт {latest}, продолжаю с эпохи {start_epoch + 1}/{epochs}")
        else:
            log("[resume] чекпоинтов не найдено в checkpoint_dir, начинаю с нуля")

    if start_epoch >= epochs:
        log(f"[resume] чекпоинт уже на эпохе {start_epoch} >= epochs={epochs}, обучать нечего")
        return w

    idx_all = np.arange(n)
    for epoch in range(start_epoch, epochs):
        rng.shuffle(idx_all)
        total_loss = 0.0
        t0 = time.time()

        for start in range(0, n, batch_size):
            batch_idx = idx_all[start:start + batch_size]
            Xb = X[batch_idx]                 # (b, 768) sparse
            yb = targets[batch_idx]           # (b,)

            pred_raw = Xb @ w                 # (b,) — tapered eval для всего батча разом
            pred = 1.0 / (1.0 + np.exp(-K * pred_raw))
            err = pred - yb
            total_loss += float(np.sum(err * err))

            # d(loss)/d(pred_raw) для каждого сэмпла батча
            dloss_draw = 2 * err * pred * (1 - pred) * K   # (b,)
            grad = (Xb.T @ dloss_draw) / max(len(batch_idx), 1)   # (768,)

            t_step += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        mse = total_loss / n
        log(f"epoch {epoch + 1}/{epochs}  mse={mse:.6f}  time={time.time() - t0:.1f}s")

        is_last = (epoch + 1) == epochs
        if checkpoint_dir and ((epoch + 1) % checkpoint_every == 0 or is_last):
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch + 1:04d}.npz")
            save_checkpoint(ckpt_path, epoch + 1, w, m, v, t_step)
            log(f"  -> чекпоинт сохранён: {ckpt_path}")
            if weights_out_path:
                save_weights(w, weights_out_path)
                log(f"  -> текущие веса сохранены: {weights_out_path}")

    return w


def save_weights(weights, path="weights.json"):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"weights": weights.tolist()}, f)
    os.replace(tmp_path, path)


def load_weights(path="weights.json"):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["weights"], dtype=np.float64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="jsonl файл с {fen, cp} после prepare_data.py")
    ap.add_argument("--limit", type=int, default=None, help="ограничить число позиций (для теста)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out", default="weights.json",
                     help="куда сохранять итоговые (и промежуточные, после каждого чекпоинта) веса")
    ap.add_argument("--init-material", action="store_true",
                     help="начать с материальных весов вместо нулей (быстрее сходится); "
                          "игнорируется при --resume, если найден чекпоинт")
    ap.add_argument("--workers", type=int, default=1,
                     help="число процессов для разбора FEN (полезно на сервере с большим датасетом)")
    ap.add_argument("--checkpoint-dir", default=None,
                     help="директория для чекпоинтов (веса + состояние Adam); рекомендуется на Google Диске")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                     help="сохранять чекпоинт каждые N эпох (по умолчанию — каждую)")
    ap.add_argument("--resume", action="store_true",
                     help="продолжить с последнего чекпоинта в --checkpoint-dir, если он есть")
    args = ap.parse_args()

    log(f"Loading dataset from {args.data} (limit={args.limit}, workers={args.workers})...")
    t0 = time.time()
    X, targets = load_dataset(args.data, limit=args.limit, n_workers=args.workers)
    log(f"Loaded {X.shape[0]} positions, {X.nnz} nonzeros in design matrix "
        f"({time.time() - t0:.1f}s).")

    init_w = default_weights() if args.init_material else None
    w = train(X, targets, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, init_weights=init_w,
              checkpoint_dir=args.checkpoint_dir,
              checkpoint_every=args.checkpoint_every,
              resume=args.resume,
              weights_out_path=args.out)
    save_weights(w, args.out)
    log(f"Saved weights to {args.out}")
