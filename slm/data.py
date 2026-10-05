"""Загрузка/токенизация датасета и подготовка train.bin / val.bin (uint16).

Использование:
    python data.py                # токенизирует corpus.txt, пишет .bin + метаданные
"""
import os
import pickle
import numpy as np
from config import Config
import tok_loader  # общий доступ к токен-функциям


def prepare(cfg: Config):
    raw = open(cfg.data_path, encoding="utf-8").read()
    ids = tok_loader.encode(raw)
    arr = np.array(ids, dtype=np.uint16)
    n = len(arr)
    split = int(n * 0.9)
    os.makedirs(cfg.bin_dir, exist_ok=True)
    arr[:split].tofile(os.path.join(cfg.bin_dir, "train.bin"))
    arr[split:].tofile(os.path.join(cfg.bin_dir, "val.bin"))
    meta = {"n_tokens": n, "train": split, "val": n - split,
            "vocab_size": tok_loader.vocab_size()}
    with open(os.path.join(cfg.bin_dir, "meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print(f"Токенов всего: {n:,} | train: {split:,} | val: {n-split:,} | vocab: {meta['vocab_size']}")


if __name__ == "__main__":
    cfg = Config()
    if not os.path.exists(cfg.data_path):
        raise SystemExit(f"Нет данных: {cfg.data_path}. Скачай/положи корпус туда.")
    prepare(cfg)
