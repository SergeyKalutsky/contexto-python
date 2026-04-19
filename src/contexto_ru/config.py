from pathlib import Path


class AppConfig:
    """Все пути и настройки приложения в одном месте."""

    def __init__(self):
        # Корень проекта — два уровня выше папки с этим файлом
        root = Path(__file__).resolve().parents[2]

        self.base_dir = root
        self.vocab_path = root / "data" / "vocab_ru.txt"
        self.targets_path = root / "data" / "targets_ru.txt"

        # Модель navec (~50 МБ) — намного легче прежней fastText (~1.3 ГБ)
        self.model_path = root / "models" / "navec_hudlit_v1_12B_500K_300d_100q.tar"
        self.model_url = (
            "https://storage.yandexapis.com/natasha-navec/packs/"
            "navec_hudlit_v1_12B_500K_300d_100q.tar"
        )
