"""
train_dual.py — supervised training of DualHeadResNet on Lichess games.

Two steps:

  1. prepare: parse PGN (.pgn or .pgn.zst) into compact shards
       python train_dual.py prepare lichess-2400.pgn.zst --out data/ --max-positions 5000000

     For every position the shard stores the input planes (bit-packed), the index of
     the move that was played (policy target) and a value target from White's point
     of view: tanh(cp / 400) when the game has [%eval] annotations, otherwise the
     game result (+1 / 0 / -1).

  2. train: fit the network
       python train_dual.py train data/ --epochs 3 --out chess_dualhead_best.pth

     Mixed precision on CUDA, AdamW with cosine decay, a checkpoint after every epoch
     (and with --resume training continues from it).

The resulting checkpoint is loaded by chess_engine.load_model and used by uci_engine.py.
"""

from __future__ import annotations

import argparse
import glob
import io
import math
import os
import random
import time

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chess_engine import DualHeadResNet, ModelConfig, encode_move, fen_to_planes, load_model, save_checkpoint

SHARD_SIZE = 100_000


# ----------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------

def _open_pgn(path: str):
    if path.endswith(".zst"):
        import zstandard as zstd
        stream = zstd.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _result_value(result: str):
    return {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(result)


def _write_shard(out_dir: str, idx: int, planes, policy, value) -> None:
    np.savez(
        os.path.join(out_dir, f"shard_{idx:05d}.npz"),
        planes=np.stack(planes), policy=np.array(policy, dtype=np.int16), value=np.array(value, dtype=np.float16),
    )


def prepare(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    history_len = args.history_len
    planes, policy, value = [], [], []
    shard, total, games = 0, 0, 0
    pgn = _open_pgn(args.pgn)

    while total < args.max_positions:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        result = _result_value(game.headers.get("Result", "*"))
        if result is None:
            continue
        try:
            elo = min(int(game.headers.get("WhiteElo", 0)), int(game.headers.get("BlackElo", 0)))
        except ValueError:
            elo = 0
        if elo < args.min_elo:
            continue

        games += 1
        board = game.board()
        history: list[str] = []
        for ply, node in enumerate(game.mainline()):
            move = node.move
            if ply >= args.skip_plies:
                eval_ = node.eval()
                if eval_ is not None:
                    cp = eval_.white().score(mate_score=2000)
                    target = math.tanh(cp / 400.0)
                else:
                    target = result
                x = fen_to_planes(board.fen(), history, history_len)
                planes.append(np.packbits(x.astype(np.uint8), axis=None))
                policy.append(encode_move(move))
                value.append(target)
                total += 1
                if len(planes) == SHARD_SIZE:
                    _write_shard(args.out, shard, planes, policy, value)
                    shard += 1
                    planes, policy, value = [], [], []
                    print(f"shard {shard}: {total} positions from {games} games", flush=True)
            history = ([board.fen()] + history)[:history_len]
            board.push(move)
            if total >= args.max_positions:
                break

    if planes:
        _write_shard(args.out, shard, planes, policy, value)
    print(f"done: {total} positions from {games} games in {args.out}")


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------

class ShardDataset(Dataset):
    def __init__(self, folder: str, in_planes: int):
        files = sorted(glob.glob(os.path.join(folder, "shard_*.npz")))
        if not files:
            raise FileNotFoundError(f"No shard_*.npz files in {folder}; run 'prepare' first")
        data = [np.load(f) for f in files]
        self.planes = np.concatenate([d["planes"] for d in data])
        self.policy = np.concatenate([d["policy"] for d in data]).astype(np.int64)
        self.value = np.concatenate([d["value"] for d in data]).astype(np.float32)
        self.shape = (in_planes, 8, 8)

    def __len__(self) -> int:
        return len(self.policy)

    def __getitem__(self, i):
        x = np.unpackbits(self.planes[i])[: np.prod(self.shape)].reshape(self.shape).astype(np.float32)
        return torch.from_numpy(x), self.policy[i], self.value[i]


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    start_epoch = 0
    if args.resume and os.path.exists(args.out):
        model, _ = load_model(args.out, device)
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        start_epoch = int(ckpt.get("epoch", 0))
    else:
        cfg = ModelConfig(channels=args.channels, num_blocks=args.blocks, history_len=args.history_len,
                          in_planes=18 + 12 * args.history_len)
        model = DualHeadResNet(cfg).to(device)
        ckpt = None
    model.train()

    data = ShardDataset(args.data, model.config.in_planes)
    split = max(1, int(len(data) * 0.02))
    indices = list(range(len(data)))
    random.shuffle(indices)
    val_set = torch.utils.data.Subset(data, indices[:split])
    train_set = torch.utils.data.Subset(data, indices[split:])
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(total_steps, 1),
                                                    pct_start=0.05)
    if ckpt and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={device} positions={len(data)} params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        for step, (x, p, v) in enumerate(loader, 1):
            x, p, v = x.to(device, non_blocking=True), p.to(device), v.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, value = model(x)
                policy_loss = F.cross_entropy(logits.float(), p)
                value_loss = F.mse_loss(value.float(), v)
                loss = policy_loss + args.value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if step % args.log_every == 0:
                print(f"epoch {epoch + 1} step {step}/{len(loader)} policy {policy_loss.item():.3f} "
                      f"value {value_loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{step * args.batch_size / (time.time() - t0):.0f} pos/s", flush=True)

        model.eval()
        correct, mse, n = 0, 0.0, 0
        with torch.no_grad():
            for x, p, v in val_loader:
                logits, value = model(x.to(device))
                correct += (logits.argmax(1).cpu() == p).sum().item()
                mse += F.mse_loss(value.float().cpu(), v, reduction="sum").item()
                n += len(p)
        print(f"epoch {epoch + 1}: val move accuracy {correct / n:.3f}, val value mse {mse / n:.4f}")
        save_checkpoint(args.out, model, epoch=epoch + 1, optimizer=optimizer.state_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="PGN -> training shards")
    p.add_argument("pgn")
    p.add_argument("--out", default="data")
    p.add_argument("--max-positions", type=int, default=5_000_000)
    p.add_argument("--min-elo", type=int, default=0)
    p.add_argument("--skip-plies", type=int, default=6, help="skip the first plies of each game")
    p.add_argument("--history-len", type=int, default=3)

    t = sub.add_parser("train", help="train DualHeadResNet")
    t.add_argument("data")
    t.add_argument("--out", default="chess_dualhead_best.pth")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--channels", type=int, default=256)
    t.add_argument("--blocks", type=int, default=16)
    t.add_argument("--history-len", type=int, default=3)
    t.add_argument("--value-weight", type=float, default=1.0)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--log-every", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true")

    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else train(args)


if __name__ == "__main__":
    main()
