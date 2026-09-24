# 🧠 Мини-SLM: от А до Я

План-конспект и база знаний по проекту «small language model с нуля».
Этот файл — то, что можно целиком перенести в Notion (каждая секция = страница/заголовок).

---

## 0. Roadmap: 7 этапов

| # | Этап | Что делаем | Результат |
|---|------|-----------|-----------|
| 1 | Данные | Собираем корпус текста | `data/corpus.txt` |
| 2 | Токенизация | BPE или word-level словарь | `tokenizer/` |
| 3 | Подготовка данных | Токенизируем, делим train/val | `train.bin`, `val.bin` |
| 4 | Модель | Трансформер-декодер | `model.py` (~1–5M параметров) |
| 5 | Обучение | AdamW + cosine LR + eval loop | `out/best_ckpt.pt` |
| 6 | Генерация | Sampling: temperature/top-k | `generate.py` |
| 7 | Эксперименты | Масштабирование, ablations | заметки ниже |

Команды запуска (в папке `slm/`):

```bash
python tokenizer.py --data data/corpus.txt --vocab 4096   # шаг 2
python data.py                                            # шаг 3
python train.py --iters 2000                              # шаг 5
python generate.py --prompt "Жил был" --tokens 100        # шаг 6
```

---

## 1. Где брать данные

**Где делать:** Kaggle / HuggingFace Datasets / Wikipedia dumps / Project Gutenberg.

Варианты для русскоязычной модели:
- **Russian National Corpus (ruscorpora)** — академично, но сложно скачать.
- **Огранизации из RUSLEANDERA / librusec** — художественная литература (Gutenberg-подобные).
- **Википедия**: `https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2` → распарсить `wikitextextractor`.
- **HuggingFace**: датасеты типа `google/flores`, `mcamabrol/russian-wiki-news`, `citiec/ru_sentiment_dataset`.
- Самый быстрый старт: пара глав книги Толстого/Достоевского (public domain) → `data/corpus.txt`.

**Сколько нужно?** Для практики: 1–5 МБ текста (~200K–1M слов) уже даёт видимую связность на word-level модели.

⚠️ Практическое правило: **объём данных важнее размера модели** для маленького SLM. Лучше 1M токенов на модели 1M параметров, чем 100K токенов на 10M параметрах (переобучится).

---

## 2. Токенизация

**Что это:** превращаем текст в последовательность целых чисел. Модель видит только числа.

Уровни:
- **character-level**: словарь ~100 символов. Просто, но последовательности длинные, модель мучается.
- **word-level** (наш fallback): словарь = самые частые слова, редкие → `<unk>`. Быстро понять суть, но OOV проблема.
- **BPE (Byte-Pair Encoding)**: как GPT-2/Llama — подсловарь из частотных кусков. Золотой стандарт.

**Как учить BPE:** репо `karpathy/minbpe` (`pip install minbpe`) или скрипт `train_bpe.py` из nanoGPT. Положил рядом — `tokenizer.py` сам его подхватит.

Ключевые метрики для заметок:
- vocab size: 4K–50K (для мини-эксперимента 4096 достаточно)
- bits per character / tokens per word — качество компрессии токенизатора

---

## 3. Архитектура модели (что и почему)

Декодер-only трансформер (как GPT). Компоненты и обоснование:

| Компонент | Выбор | Почему |
|-----------|-------|--------|
| Positional encoding | **RoPE** | лучше обобщает на длину, без learnable-таблицы |
| Normalization | **Pre-norm RMSNorm** | стабильнее post-norm, дешевле LayerNorm |
| FFN | **SwiGLU** | современная замена ReLU-FFN (Llama/PaLM), +~качество при тех же FLOPs |
| Attention | causal, SDPA (`is_causal=True`) | PyTorch сам выберет flash/mem-efficient ядро |
| Embeddings | **tied** (вход=выход) | экономия ~vocab×d_model параметров |
| Инициализация | std=0.02, зануление residual-проекций | стабильный старт обучения |

Размер для CPU-практики: d_model=192, 4 слоя, 6 голов ≈ **1.3M параметров**.
Формула параметров (приближённо): `12·L·d² + 2·d·V`.

---

## 4. Обучение

**Где делать:** локально на CPU (медленно, но аутентично) или бесплатная GPU:
- Google Colab (T4, 12ч сессия)
- Kaggle Kernels (30ч GPU/неделю бесплатно)
- Lightning AI Studio

Гиперпараметры и смысл:
- **AdamW**, lr=3e-4, betas=(0.9, 0.95) — константы nanoGPT, хорошо работают на мелких моделях
- **warmup 5% шагов + cosine decay** — не даём обучению «взорваться» в начале, плавно затухаем к концу
- **grad clip 1.0** — защита от всплесков градиента
- **weight decay 0.1 только на матрицы весов** (не на norm-векторы) — стандарт Llama
- batch_size 32 × block_size 256 = 8K токенов на шаг. На CPU это ~секунда на шаг.

**Когда останавливать:** val loss перестал падать 5–10 оценок подряд → переобучение близко. Сравнивай train vs val loss: расхождение > 0.2 nat — мало данных или велика модель.

Метрики:
- **loss** (cross-entropy, nat) — основная
- **perplexity = exp(loss)** — «эффективное число вариантов»; для word-level модели на 1M слов PPL 30–80 — норм результат

---

## 5. Генерация (inference)

- **temperature**: 0.7–1.0 — баланс креативности/связности; T→0 = greedy
- **top-k**: обрезаем хвост распределения (k=50 типично)
- top-p (nucleus) — продвинутый вариант, сделать как упражнение

Эксперимент-заметка: при T<0.3 word-level модель зацикливается на частых словах; при T>1.2 — бред. Это видно живьём, интересно поиграться.

---

## 6. Чек-лист «Я всё сделал?»

- [ ] Корпус ≥ 500KB уникального текста лежит в `data/corpus.txt`
- [ ] Токенизатор построен, vocab записан в `tokenizer/meta.json`
- [ ] `train.bin`/`val.bin` созданы, отношение 90/10
- [ ] Обучение: loss монотонно падает первые 100 шагов (иначе — баг!)
- [ ] Val loss < ln(vocab) — модель учится, а не угадывает равномерно
- [ ] Генерация выдаёт осмысленные n-gram'ы после промпта
- [ ] Записал 3 ablation-эксперимента (см. ниже)

---

## 7. Идеи для экспериментов (продолжение практики)

1. **BPE вместо word-level** — установи `minbpe`, пересобери токенизатор, сравни PPL.
2. **Scaling law на пальцах**: прогони обучение при d_model ∈ {96, 192, 384}, построй график val_loss vs params.
3. **Длина контекста**: block_size 64 vs 256 — как влияет на связность генерации.
4. **Дообучение на диалогах** (инструктаж лайт): собери пары «вопрос-ответ», fine-tune с маской на prompt.
5. **KV-cache** в `generate()` — ускорение генерации в разы (классическое упражнение).
6. **Eval на held-out**: реализуй next-word accuracy и сравнись со случайным baseline 1/vocab.

---

## 8. Полезные ссылки

- nanoGPT (Karpathy) — эталонный минималистичный код: https://github.com/karpathy/nanoGPT
- «Let's build GPT» — видео-разбор той же архитектуры: YouTube Karpathy
- minbpe: https://github.com/karpathy/minbpe
- Illustrated Transformer (Jay Alammar): https://jalammar.github.io/illustrated-transformer/
- RoPE объяснение: https://blog.eleuther.ai/rotary-embeddings/
- Chinchilla paper (про масштабирование данные↔модель): arXiv 2203.15556
