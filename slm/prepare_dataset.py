"""Подготовка произвольного датасета к виду data/corpus.txt (UTF-8, один текст).

Умеем читать: .txt/.md (как есть), .json (список/список словарей), .jsonl (построчно),
из папки — рекурсивно все эти файлы. Из JSON-словарей берём текстовые поля
(title/text/content/question/answer/body/page_content и т.п.).

Использование:
    python prepare_dataset.py --src <путь_к_файлу_или_папке> [--out data/corpus.txt] [--max-mb 20]

Пример (Kaggle):
    path = kagglehub.dataset_download("ryemint/kit-knowledge-base")  # напечатает путь
    python prepare_dataset.py --src "<этот путь>"
После этого: python tokenizer.py --data data/corpus.txt --vocab 4096 && python data.py
"""
import argparse
import json
import os
import re

TEXT_FIELDS = ("text", "content", "answer", "question", "body", "abstract",
               "title", "description", "page_content", "output", "input", "context")


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)                 # HTML-теги
    s = re.sub(r"https?://\S+", " ", s)            # ссылки
    s = re.sub(r"[ \t]+", " ", s)                  # лишние пробелы
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _from_obj(obj, out: list):
    """Рекурсивно вытаскиваем строки из JSON-структур."""
    if isinstance(obj, str):
        if len(obj) > 3:
            out.append(_clean(obj))
    elif isinstance(obj, dict):
        for k in TEXT_FIELDS:
            if k in obj:
                _from_obj(obj[k], out)
        # если ни одного знакомого поля — берём все строковые значения
        if not any(k in obj for k in TEXT_FIELDS):
            for v in obj.values():
                if isinstance(v, str):
                    _from_obj(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _from_obj(v, out)


def collect(path: str) -> list:
    docs = []
    files = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            for n in names:
                if n.lower().endswith((".txt", ".md", ".json", ".jsonl")):
                    files.append(os.path.join(root, n))
    else:
        files = [path]

    for f in sorted(files):
        ext = f.lower().rsplit(".", 1)[-1]
        try:
            if ext in ("txt", "md"):
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    docs.append(_clean(fh.read()))
            elif ext == "json":
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    out = []
                    _from_obj(json.load(fh), out)
                    docs.extend(out)
            elif ext == "jsonl":
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        out = []
                        try:
                            _from_obj(json.loads(line), out)
                        except json.JSONDecodeError:
                            out.append(_clean(line))
                        docs.extend(out)
        except Exception as e:
            print(f"пропуск {f}: {e}")
    return [d for d in docs if len(d) > 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="файл/папка после скачивания датасета")
    ap.add_argument("--out", default=os.path.join("data", "corpus.txt"))
    ap.add_argument("--max-mb", type=float, default=50.0, help="ограничить размер корпуса")
    args = ap.parse_args()

    docs = collect(args.src)
    if not docs:
        raise SystemExit("Ничего текстового не найдено. Покажи содержимое папки: ls <path>")
    text = "\n\n".join(docs)
    limit = int(args.max_mb * 1024 * 1024)
    if len(text.encode("utf-8")) > limit:
        text = text[:limit // 2]  # грубо, по символам
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    uniq = len(set(text.split()))
    print(f"Документов: {len(docs):,} | знаков: {len(text):,} | уникальных слов: {uniq:,}")
    print(f"Записано в {args.out}. Дальше:\n"
          f"  python tokenizer.py --data {args.out} --vocab 4096\n"
          f"  python data.py")


if __name__ == "__main__":
    main()
