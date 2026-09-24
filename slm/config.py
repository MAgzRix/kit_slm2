"""Конфигурация мини-SLM. Всё в одном месте — меняй и экспериментируй."""
from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # --- Архитектура ---
    vocab_size: int = 4096      # размер словаря BPE
    d_model: int = 192          # размер эмбеддингов / скрытой размерности
    n_layers: int = 4           # слоёв трансформера
    n_heads: int = 6            # голов внимания (d_model % n_heads == 0)
    d_ff_mult: int = 4          # FFN = d_model * d_ff_mult
    dropout: float = 0.0        # на маленьком датасете часто не нужен
    block_size: int = 256       # длина контекста (токенов)
    rope_base: float = 10000.0

    # --- Токенизатор ---
    tokenizer_dir: str = "tokenizer"

    # --- Данные ---
    data_path: str = "data/corpus.txt"
    bin_dir: str = "data"       # сюда пишем train.bin / val.bin

    # --- Обучение ---
    batch_size: int = 16
    lr: float = 3e-4            # пиковый learning rate
    min_lr_frac: float = 0.1    # до какого множителя от lr затухает
    warmup_steps: int = 50
    max_iters: int = 2000       # шагов обучения
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100    # раз в N шагов — валидация
    eval_batches: int = 10      # сколько батчей усредняем при оценке
    log_interval: int = 25
    out_dir: str = "out"
    seed: int = 1337

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    @property
    def n_params_note(self) -> str:
        return f"~{(12 * self.n_layers * self.d_model**2 + 2 * self.d_model * self.vocab_size)/1e6:.1f}M параметров"
