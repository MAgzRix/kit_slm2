"""Единая точка доступа к токенизатору (word-level fallback или BPE).

encode(text) -> list[int];  decode(ids) -> str;  vocab_size() -> int
"""
import os
import pickle
from typing import List

_TOK_DIR = os.path.join(os.path.dirname(__file__), "tokenizer")
_cache = {}


def _load():
    if "kind" in _cache:
        return
    meta_path = os.path.join(_TOK_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("Токенизатор не построен. Сначала: python tokenizer.py --data ...")
    import json
    meta = json.load(open(meta_path))
    if meta["type"] == "word":
        d = pickle.load(open(os.path.join(_TOK_DIR, "wordpiece.pkl"), "rb"))
        _cache.update(kind="word", stoi=d["stoi"], itos=d["itos"])
    else:
        bpe = pickle.load(open(os.path.join(_TOK_DIR, "bpe.pkl"), "rb"))
        _cache.update(kind="bpe", bpe=bpe)


def encode(text: str) -> List[int]:
    _load()
    if _cache["kind"] == "word":
        unk = _cache["stoi"]["<unk>"]
        return [_cache["stoi"].get(w, unk) for w in text.split()]
    return _cache["bpe"].encode(text)


def decode(ids) -> str:
    _load()
    if _cache["kind"] == "word":
        itos = _cache["itos"]
        return " ".join(itos[i] for i in ids if itos[i] not in ("<unk>", "<|endoftext|>"))
    return _cache["bpe"].decode(ids)


def vocab_size() -> int:
    _load()
    if _cache["kind"] == "word":
        return len(_cache["itos"])
    return len(_cache["bpe"].encoder)
