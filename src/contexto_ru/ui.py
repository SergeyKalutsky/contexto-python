import tkinter as tk
from tkinter import messagebox, ttk

from .game import ContextoGame, GuessResult


class ContextoApp:
    """Графический интерфейс игры на tkinter."""

    def __init__(self, game: ContextoGame):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Контексто RU")
        self.root.geometry("760x540")

        # Текст подсказки над полем ввода
        self.status_var = tk.StringVar(
            value="Угадайте скрытое русское слово. Чем меньше ранг, тем ближе вы к ответу."
        )
        self.guess_var = tk.StringVar()

        # Все попытки текущей игры: {нормализованное_слово -> GuessResult}
        self.guess_results: dict[str, GuessResult] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """Создаёт все элементы интерфейса."""
        style = ttk.Style(self.root)
        style.configure("Treeview", rowheight=26)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        ttk.Label(outer, text="Контексто RU", font=("Segoe UI", 22, "bold")).pack(anchor=tk.W)

        # Строка подсказки
        ttk.Label(outer, textvariable=self.status_var, wraplength=700).pack(anchor=tk.W, pady=(8, 16))

        # Строка ввода: поле + кнопки
        input_row = ttk.Frame(outer)
        input_row.pack(fill=tk.X)

        guess_entry = ttk.Entry(input_row, textvariable=self.guess_var, font=("Segoe UI", 12))
        guess_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        guess_entry.bind("<Return>", self._submit_guess)
        guess_entry.focus_set()

        ttk.Button(input_row, text="Проверить", command=self._submit_guess).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(input_row, text="Новое слово", command=self._new_game).pack(side=tk.LEFT, padx=(8, 0))

        # Таблица с результатами
        columns = ("guess", "rank", "similarity")
        self.results_table = ttk.Treeview(outer, columns=columns, show="headings", height=18)
        self.results_table.heading("guess", text="Слово")
        self.results_table.heading("rank", text="Ранг")
        self.results_table.heading("similarity", text="Близость")
        self.results_table.column("guess", width=320)
        self.results_table.column("rank", width=120, anchor=tk.CENTER)
        self.results_table.column("similarity", width=120, anchor=tk.CENTER)
        self.results_table.pack(fill=tk.BOTH, expand=True, pady=(16, 0))

        # Цвета строк таблицы в зависимости от ранга
        self.results_table.tag_configure("exact",      background="#8fce00", foreground="#102000")
        self.results_table.tag_configure("very_close", background="#cdeccf", foreground="#17351d")
        self.results_table.tag_configure("close",      background="#e7f6d5", foreground="#314121")
        self.results_table.tag_configure("medium",     background="#fff1c7", foreground="#4e3f14")
        self.results_table.tag_configure("far",        background="#f9d7d3", foreground="#4d1f19")

    def _submit_guess(self, _event=None) -> None:
        """Вызывается при нажатии кнопки «Проверить» или Enter."""
        guess = self.guess_var.get()
        try:
            result = self.game.guess(guess)
        except ValueError as error:
            messagebox.showerror("Ошибка", str(error))
            return

        self.guess_results[result.normalized_guess] = result
        self._render_results()
        self.guess_var.set("")  # очищаем поле ввода

        if result.is_exact:
            self.status_var.set(f"Точно! Ответ: {result.normalized_guess}. Ранг {result.rank}.")
            messagebox.showinfo("Победа", f"Вы угадали слово: {result.normalized_guess}")
        else:
            self.status_var.set(
                f"Последняя попытка: {result.normalized_guess}. "
                f"Ранг {result.rank}, близость {result.similarity:.4f}."
            )

    def _render_results(self) -> None:
        """Перерисовывает таблицу с результатами (сортировка по рангу)."""
        # Удаляем все строки
        for item_id in self.results_table.get_children():
            self.results_table.delete(item_id)

        # Добавляем заново, отсортировав по рангу
        sorted_results = sorted(self.guess_results.values(), key=lambda r: r.rank)
        for result in sorted_results:
            self.results_table.insert(
                "",
                tk.END,
                values=(result.normalized_guess, result.rank, f"{result.similarity:.4f}"),
                tags=(self._color_tag(result.rank, result.is_exact),),
            )

    def _color_tag(self, rank: int, is_exact: bool) -> str:
        """Возвращает название цветового тега в зависимости от ранга слова."""
        if is_exact or rank == 1:
            return "exact"
        if rank <= 100:
            return "very_close"
        if rank <= 1000:
            return "close"
        if rank <= 10000:
            return "medium"
        return "far"

    def _new_game(self) -> None:
        """Сбрасывает игру и начинает заново с новым загаданным словом."""
        self.game.reset()
        self.guess_results.clear()
        self._render_results()
        self.status_var.set("Новая игра запущена. Угадайте скрытое слово.")

    def run(self) -> None:
        """Запускает главный цикл интерфейса."""
        self.root.mainloop()
