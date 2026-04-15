from pathlib import Path

from .normalization import normalize_word


def load_word_list(path: Path) -> list[str]:
    """
    Читает файл со списком слов (по одному на строку).
    Нормализует каждое слово и убирает дубликаты.
    Возвращает список уникальных нормализованных слов.
    """
    words: list[str] = []
    already_seen: set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            normalized = normalize_word(raw_line)
            if not normalized or normalized in already_seen:
                continue
            already_seen.add(normalized)
            words.append(normalized)

    return words
