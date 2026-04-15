import gzip
import math
import pickle
import re
from pathlib import Path


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


def _load_cache(cache_path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Загружает уже сохранённый словарь из файла-кэша."""
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data["words"], data["vectors"]


def _save_cache(cache_path: Path, words: list[str], vectors: dict[str, list[float]]) -> None:
    """Сохраняет словарь в файл-кэш, чтобы не читать модель заново."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"words": words, "vectors": vectors}, f, protocol=pickle.HIGHEST_PROTOCOL)


def _build_vocabulary_from_model(
    model_path: Path,
    max_words: int,
) -> tuple[list[str], dict[str, list[float]]]:
    """
    Читает файл модели fastText (.vec.gz) и собирает словарь русских слов.
    Возвращает два объекта:
      - words:   список слов в порядке встречаемости
      - vectors: словарь {слово -> список чисел (вектор)}
    """
    words: list[str] = []
    vectors: dict[str, list[float]] = {}
    already_seen: set[str] = set()  # чтобы не добавлять одно слово дважды

    print(f"Строю словарь из модели {model_path} ...")

    with gzip.open(model_path, "rt", encoding="utf-8", newline="\n") as f:
        f.readline()  # первая строка — служебный заголовок, пропускаем

        for line_number, line in enumerate(f, start=1):
            parts = line.rstrip().split(" ")

            # строка должна содержать слово + числа; если частей меньше 10 — пропускаем
            if len(parts) < 10:
                continue

            word = parts[0].strip().lower().replace("ё", "е")

            if not _is_valid_russian_word(word):
                continue
            if word in already_seen:
                continue

            already_seen.add(word)
            words.append(word)
            vectors[word] = [float(v) for v in parts[1:] if v]

            if len(words) >= max_words:
                break

            if line_number % 500_000 == 0:
                print(f"  Проверено строк: {line_number}, найдено слов: {len(words)}")

    if not words:
        raise EmbeddingModelError("Не удалось найти русские слова в файле модели.")

    print(f"Готово! Слов в словаре: {len(words)}")
    return words, vectors


class FastTextEmbeddings:
    """
    Хранит слова и их векторы (числовые представления).
    Умеет загружаться из кэша или строить кэш из файла модели.
    """

    def __init__(self, words: list[str], vectors: dict[str, list[float]]):
        if not words:
            raise EmbeddingModelError("Словарь пуст.")
        self.words = words
        self._vectors = vectors

    @classmethod
    def from_model_cache(
        cls,
        model_path: Path,
        cache_path: Path,
        max_words: int,
    ) -> "FastTextEmbeddings":
        """
        Загружает модель. Если кэш уже есть — берёт из него (быстро).
        Если нет — читает модель и создаёт кэш (медленно, только первый раз).
        """
        if cache_path.exists():
            words, vectors = _load_cache(cache_path)
            return cls(words, vectors)

        if not model_path.exists():
            raise EmbeddingModelError(f"Файл модели не найден: {model_path}")

        words, vectors = _build_vocabulary_from_model(model_path, max_words)
        _save_cache(cache_path, words, vectors)
        return cls(words, vectors)

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
