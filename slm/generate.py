"""Генерация текста обученной моделью.

    python generate.py --prompt "Жил был" --tokens 100 --temp 0.8
"""
import argparse
import os
import torch
from config import Config
from model import MiniSLM
import tok_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/best_ckpt.pt")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--n", type=int, default=3, help="сколько примеров сгенерировать")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = MiniSLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"Загружен {args.ckpt} (шаг {ck['step']}, val loss {ck['best_val']:.3f})")

    for i in range(args.n):
        ids = tok_loader.encode(args.prompt) if args.prompt else []
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=args.tokens,
                             temperature=args.temp, top_k=args.top_k)
        text = tok_loader.decode(out[0].tolist())
        print(f"\n--- пример {i+1} ---\n{text}")


if __name__ == "__main__":
    main()
