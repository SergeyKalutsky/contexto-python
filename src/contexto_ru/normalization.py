import re


WORD_RE = re.compile(r"[^\w\-]+", re.UNICODE)

# pymorphy3 — необязательная библиотека для лемматизации (приведения слова к начальной форме).
# Если она не установлена, просто используем базовую нормализацию.
try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
except ImportError:
    _morph = None


def normalize_word(text: str) -> str:
    """
    Приводит введённое слово к стандартному виду:
      1. Убирает пробелы и лишние символы
      2. Переводит в нижний регистр
      3. Заменяет «ё» на «е» (так слова хранятся в модели)
      4. Если установлен pymorphy3 — приводит к начальной форме (лемме)
    """
    cleaned = WORD_RE.sub("", text.strip().lower().replace("ё", "е"))
    if not cleaned:
        return ""

    if _morph is None:
        return cleaned

    parsed = _morph.parse(cleaned)
    if not parsed:
        return cleaned

    return parsed[0].normal_form
