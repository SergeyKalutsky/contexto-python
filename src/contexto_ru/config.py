from pathlib import Path


class AppConfig:
    """Все пути и настройки приложения в одном месте."""

    def __init__(self):
        # Корень проекта — два уровня выше папки с этим файлом
        root = Path(__file__).resolve().parents[2]

        self.base_dir = root
        self.vocab_path = root / "data" / "vocab_ru.txt"
        self.targets_path = root / "data" / "targets_ru.txt"
        self.fasttext_model_path = root / "models" / "cc.ru.300.vec.gz"
        self.fasttext_model_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz"
        self.embedding_cache_path = root / "models" / "ru_vocab_cache.pkl"
        self.max_vocabulary_words = 50_000
