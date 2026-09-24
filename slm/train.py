"""Обучение мини-SLM.

    python train.py [--iters 2000] [--batch 32] ...

Пишет чекпоинты в out/: best_ckpt.pt, last_ckpt.pt, config.json.
"""
import argparse
import math
import os
import time
import torch
from config import Config
from model import MiniSLM


def get_batch(cfg: Config, split: str):
    data = np_load(os.path.join(cfg.bin_dir, f"{split}.bin"))
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    xs, ys = [], []
    for i in ix:
        chunk = data[i:i + cfg.block_size + 1].astype("int64")
        xs.append(torch.from_numpy(chunk[:-1]))
        ys.append(torch.from_numpy(chunk[1:]))
    return torch.stack(xs), torch.stack(ys)


def np_load(path):
    import numpy as np
    return np.memmap(path, dtype="uint16", mode="r")


@torch.no_grad()
def evaluate(model, cfg: Config, device):
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(cfg, "val")
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    val_loss = sum(losses) / len(losses)
    return val_loss, math.exp(val_loss)  # perplexity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    if args.iters: cfg.max_iters = args.iters
    if args.batch: cfg.batch_size = args.batch
    if args.lr: cfg.lr = args.lr

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    # vocab из метаданных данных
    import json
    meta = json.load(open(os.path.join(cfg.bin_dir, "meta.json")))
    cfg.vocab_size = meta["vocab_size"]

    model = MiniSLM(cfg).to(device)
    n_params = model.count_params()
    print(f"Параметров: {n_params/1e6:.2f}M | {cfg.n_params_note}")

    # AdamW с разделением: weight decay только на матрицы весов
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    step = 0
    best_val = float("inf")
    if args.resume and os.path.exists(os.path.join(cfg.out_dir, "last_ckpt.pt")):
        ck = torch.load(os.path.join(cfg.out_dir, "last_ckpt.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        print(f"Продолжаем с шага {step}")

    def lr_at(s):
        if s < cfg.warmup_steps:
            return cfg.lr * s / max(1, cfg.warmup_steps)
        # cosine decay до min_lr_frac
        frac = (s - cfg.warmup_steps) / max(1, cfg.max_iters - cfg.warmup_steps)
        frac = min(1.0, frac)
        return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))

    t0 = time.time()
    tokens_seen = 0
    while step < cfg.max_iters:
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step)
        x, y = get_batch(cfg, "train")
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tokens_seen += x.numel()
        step += 1

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            print(f"iter {step:5d}/{cfg.max_iters} | loss {loss.item():.4f} | "
                  f"lr {lr_at(step):.2e} | {tokens_seen/dt/1e3:.1f}k tok/s")

        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            vl, ppl = evaluate(model, cfg, device)
            print(f"   >>> eval loss {vl:.4f} | perplexity {ppl:.1f}")
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "best_val": best_val, "cfg": cfg},
                           os.path.join(cfg.out_dir, "best_ckpt.pt"))
                print("   >>> сохранён лучший чекпоинт")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "best_val": best_val, "cfg": cfg},
               os.path.join(cfg.out_dir, "last_ckpt.pt"))
    print(f"Готово за {(time.time()-t0)/60:.1f} мин. Лучший val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
