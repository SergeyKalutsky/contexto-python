from urllib.request import urlretrieve

from .config import AppConfig
from .embeddings import EmbeddingModelError, WordEmbeddings
from .game import ContextoGame
from .ui import ContextoApp
from .vocab import load_word_list


def ensure_model_exists(config: AppConfig) -> None:
    """Скачивает файл модели, если его ещё нет."""
    if config.model_path.exists():
        return

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю модель в {config.model_path} ...")
    urlretrieve(config.model_url, config.model_path)
    print("Скачивание завершено.")


def main() -> None:
    config = AppConfig()
    ensure_model_exists(config)

    try:
        embeddings = WordEmbeddings(config.model_path)
    except EmbeddingModelError as error:
        raise SystemExit(
            f"{error}\n\n"
            "Перед запуском игры нужен файл модели в папке models/."
        ) from error

    # Список всех слов словаря
    vocabulary = embeddings.words

    # Загаданные слова: берём из файла, или весь словарь если файла нет
    if config.targets_path.exists():
        targets = load_word_list(config.targets_path)
    else:
        targets = vocabulary

    game = ContextoGame(
        embedding_model=embeddings,
        vocabulary=vocabulary,
        targets=targets,
    )
    print(f"[DEBUG] Загаданное слово: {game.target_word}")
    app = ContextoApp(game)
    app.run()


if __name__ == "__main__":
    main()
