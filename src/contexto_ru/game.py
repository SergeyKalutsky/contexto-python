import random

from .embeddings import WordEmbeddings, cosine_similarity
from .normalization import normalize_word


class GuessResult:
    """Результат одной попытки угадать слово."""

    def __init__(self, original_guess: str, normalized_guess: str, similarity: float, rank: int, is_exact: bool):
        self.original_guess = original_guess      # слово как ввёл пользователь
        self.normalized_guess = normalized_guess  # слово после нормализации
        self.similarity = similarity              # косинусное сходство (от 0 до 1)
        self.rank = rank                          # место слова среди всего словаря
        self.is_exact = is_exact                  # True, если слово угадано точно


class ContextoGame:
    """
    Основная логика игры.
    Загаданное слово выбирается случайно из списка целевых слов.
    Для каждого слова в словаре заранее вычисляется его ранг (место по близости к загаданному).
    """

    def __init__(
        self,
        embedding_model: WordEmbeddings,
        vocabulary: list[str] | None = None,
        targets: list[str] | None = None,
        target_word: str | None = None,
    ):
        # Если словарь не передан — берём все слова из модели
        vocabulary = vocabulary or embedding_model.words
        targets = targets or vocabulary

        if not vocabulary:
            raise ValueError("Словарь игры пуст.")
        if not targets:
            raise ValueError("Список целевых слов пуст.")

        self.embedding_model = embedding_model
        self.vocabulary = vocabulary

        # Оставляем только те целевые слова, которые есть в словаре
        vocabulary_set = set(vocabulary)
        self.targets = [word for word in targets if word in vocabulary_set]
        if not self.targets:
            raise ValueError("Целевые слова не пересекаются с игровым словарем.")

        # Кэш векторов: чтобы не запрашивать один и тот же вектор дважды
        self._vector_cache: dict[str, list[float]] = {}

        # Кэш рангов: {слово -> (ранг, близость)}
        self._ranking_cache: dict[str, tuple[int, float]] = {}

        self.target_word = target_word or random.choice(self.targets)
        self._prepare_round(self.target_word)

    def _get_vector(self, word: str) -> list[float]:
        """Возвращает вектор слова, используя кэш."""
        if word not in self._vector_cache:
            self._vector_cache[word] = self.embedding_model.vector(word)
        return self._vector_cache[word]

    def _prepare_round(self, target_word: str) -> None:
        """
        Вычисляет ранги всех слов словаря относительно загаданного слова.
        Вызывается один раз при старте или сбросе игры.
        """
        self.target_word = target_word
        target_vector = self._get_vector(target_word)

        # Считаем близость каждого слова к загаданному
        scored: list[tuple[str, float]] = []
        for word in self.vocabulary:
            similarity = cosine_similarity(target_vector, self._get_vector(word))
            scored.append((word, similarity))

        # Сортируем по убыванию близости: самое похожее слово — ранг 1
        scored.sort(key=lambda item: item[1], reverse=True)

        # Сохраняем ранги в словарь для быстрого поиска
        self._ranking_cache = {
            word: (index + 1, similarity)
            for index, (word, similarity) in enumerate(scored)
        }

    def reset(self, target_word: str | None = None) -> None:
        """Начинает новую игру. Если слово не указано — выбирается случайное."""
        self._prepare_round(target_word or random.choice(self.targets))

    def guess(self, raw_guess: str) -> GuessResult:
        """
        Обрабатывает попытку игрока.
        Возвращает GuessResult с рангом и близостью введённого слова.
        """
        normalized = normalize_word(raw_guess)
        if not normalized:
            raise ValueError("Введите слово.")
        if normalized not in self._ranking_cache:
            raise ValueError("Слова нет в словаре игры.")

        rank, similarity = self._ranking_cache[normalized]
        return GuessResult(
            original_guess=raw_guess,
            normalized_guess=normalized,
            similarity=similarity,
            rank=rank,
            is_exact=(normalized == self.target_word),
        )
