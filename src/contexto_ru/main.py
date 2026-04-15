from urllib.request import urlretrieve

from .config import AppConfig
from .embeddings import EmbeddingModelError, FastTextEmbeddings
from .game import ContextoGame
from .ui import ContextoApp
from .vocab import load_word_list


def ensure_model_exists(config: AppConfig) -> None:
    """Скачивает файл модели, если его ещё нет."""
    if config.fasttext_model_path.exists():
        return

    config.fasttext_model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю модель в {config.fasttext_model_path} ...")
    urlretrieve(config.fasttext_model_url, config.fasttext_model_path)
    print("Скачивание завершено.")


def main() -> None:
    config = AppConfig()
    ensure_model_exists(config)

    try:
        embeddings = FastTextEmbeddings.from_model_cache(
            config.fasttext_model_path,
            config.embedding_cache_path,
            max_words=config.max_vocabulary_words,
        )
    except EmbeddingModelError as error:
        raise SystemExit(
            f"{error}\n\n"
            "Перед запуском игры нужен файл русской fastText-модели models/cc.ru.300.vec.gz."
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
    app = ContextoApp(game)
    app.run()


if __name__ == "__main__":
    main()
