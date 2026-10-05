"""Обёртка над train_tokenizer (BPE, как у GPT-2 / nanoGPT).

Использование:
    python tokenizer.py --data data/corpus.txt --dir tokenizer --vocab 4096
"""
import argparse
import os
import pickle


def _ensure_train_bpe():
    """Пробуем подтянуть классический train_bpe из репо karpathy/minbpe/nanoGPT."""
    try:
        from train_bpe import BPE  # если положишь minbpe/nanoGPT рядом — сработает
        return BPE
    except ImportError:
        return None


def build_fallback_wordpiece(corpus_path: str, vocab_size: int):
    """Запасной вариант: простейший word-level словарь + редкие слова в <unk>.

    Это НЕ BPE, но для практики «от А до Я» на первом прогоне достаточно:
    модель учится предсказывать следующие СЛОВО. Позже заменим на нормальный BPE.
    """
    from collections import Counter
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    counts = Counter(words)
    most_common = [w for w, _ in counts.most_common(vocab_size - 2)]
    itos = ["<unk>", "<|endoftext|>"] + most_common
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode_words(text: str, stoi: dict) -> list[int]:
    unk = stoi["<unk>"]
    return [stoi.get(w, unk) for w in text.split()]


def decode_words(ids, stoi_inv: list[str]) -> str:
    return " ".join(s for s in (stoi_inv[i] for i in ids) if not s.startswith("<|"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--dir", default="tokenizer")
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    BPE = _ensure_train_bpe()
    meta = {"type": None}
    if BPE is not None:
        print("Найден train_bpe — строим BPE-токенизатор...")
        with open(args.data, encoding="utf-8") as f:
            corpus = f.read()
        bpe = BPE(corpus, args.vocab)
        with open(os.path.join(args.dir, "bpe.pkl"), "wb") as f:
            pickle.dump(bpe, f)
        meta = {"type": "bpe", "vocab_size": args.vocab}
    else:
        print("train_bpe не найден — строим fallback word-level токенизатор.")
        stoi, itos = build_fallback_wordpiece(args.data, args.vocab)
        with open(os.path.join(args.dir, "wordpiece.pkl"), "wb") as f:
            pickle.dump({"stoi": stoi, "itos": itos}, f)
        meta = {"type": "word", "vocab_size": len(itos)}

    with open(os.path.join(args.dir, "meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)
    print(f"Готово. vocab={meta['vocab_size']} тип={meta['type']} -> {args.dir}/")
