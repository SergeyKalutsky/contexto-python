import math
import re
from pathlib import Path

from navec import Navec


# Ошибка, которую мы бросаем, если что-то пошло не так при загрузке модели
class EmbeddingModelError(RuntimeError):
    pass


# Регулярное выражение: слово должно состоять только из русских букв (и дефиса внутри)
RUSSIAN_WORD_RE = re.compile(r"^[а-яё]+(-[а-яё]+)*$")


def _is_valid_russian_word(word: str) -> bool:
    """Возвращает True, если слово подходит для игрового словаря."""
    if len(word) < 2:
        return False
    if word.startswith("-") or word.endswith("-"):
        return False
    return bool(RUSSIAN_WORD_RE.fullmatch(word))


class WordEmbeddings:
    """
    Загружает компактную русскую модель navec (~50 МБ) и хранит векторы слов.
    В отличие от fastText — грузится быстро и не требует отдельного кэша.
    """

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise EmbeddingModelError(f"Файл модели не найден: {model_path}")

        print(f"Загружаю модель {model_path} ...")
        navec = Navec.load(str(model_path))

        # Отбираем только подходящие русские слова и сохраняем их векторы
        self._vectors: dict[str, list[float]] = {}
        for word in navec.vocab.words:
            cleaned = word.strip().lower().replace("ё", "е")
            if _is_valid_russian_word(cleaned) and cleaned not in self._vectors:
                self._vectors[cleaned] = navec[word].tolist()

        self.words = list(self._vectors.keys())
        print(f"Готово! Слов в словаре: {len(self.words)}")

    def vector(self, word: str) -> list[float]:
        """Возвращает числовой вектор для слова."""
        if word not in self._vectors:
            raise EmbeddingModelError(f"Слово не найдено в модели: {word}")
        return self._vectors[word]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Считает косинусное сходство двух векторов — число от 0 до 1.
    Чем ближе к 1, тем слова «похожее» по смыслу.
    """
    if not vec_a or not vec_b:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    len_a = math.sqrt(sum(a * a for a in vec_a))
    len_b = math.sqrt(sum(b * b for b in vec_b))

    if len_a == 0.0 or len_b == 0.0:
        return 0.0

    return dot_product / (len_a * len_b)
