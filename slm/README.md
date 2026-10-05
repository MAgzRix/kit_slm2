# Мини-SLM (small language model) с нуля

Полный цикл: данные → токенизация → модель → обучение → генерация.
База знаний/план — в [NOTION.md](NOTION.md) (готов к переносу в Notion).

## Быстрый старт

```bash
python tokenizer.py --data data/corpus.txt --vocab 4096   # словарь
python data.py                                            # train.bin / val.bin
python train.py --iters 2000                              # обучение
python generate.py --prompt "Кот" --tokens 100            # генерация
```

## Структура

- `config.py` — все гиперпараметры в одном месте
- `tokenizer.py` — учим BPE (если установлен minbpe/nanoGPT) или word-level fallback
- `tok_loader.py` — единый encode/decode для всего проекта
- `data.py` — корпус → бинарные токен-массивы, сплит 90/10
- `model.py` — GPT-декодер: RoPE + RMSNorm + SwiGLU + tied embeddings (~2M параметров)
- `train.py` — AdamW, warmup+cosine LR, grad clip, eval по val loss/perplexity
- `generate.py` — sampling с temperature/top-k
- `data/corpus.txt` — учебный корпус (сейчас синтетический демо-текст; замени на свой!)

## Статус прогона (CPU, 400 итераций, ~5 мин)

val loss 0.96 | perplexity 2.6 | модель связно достраивает предложения шаблонов.

## Что дальше

Смотри раздел «7. Идеи для экспериментов» в NOTION.md.
