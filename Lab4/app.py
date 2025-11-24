import tkinter as tk
from tkinter import ttk
import os
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import glob


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Criteria Decision Analysis - Methods Comparison")
        self.geometry("1280x760")

        # Ścieżka do folderu z danymi
        self.base_path = os.path.join(os.path.dirname(__file__), '..', 'Lab2', 'data')
        self.data_folders = self.get_data_folders()
        self.current_folder = None
        self.current_files = []
        self.current_file_index = 0
        self.current_data = None

        # Słownik kierunków optymalizacji
        self.optimization_directions = {
            'x1': 'min',
            'x2': 'min'
        }

        # Wyniki metod
        self.method_results = {}

        self.create_gui()

    def get_data_folders(self):
        return [item for item in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, item))]

    def load_data_from_folder(self, folder_name):
        self.current_folder = folder_name
        folder_path = os.path.join(self.base_path, folder_name)
        self.current_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        self.current_file_index = 0
        if self.current_files:
            self.load_current_file()
            self.update_file_counter()

    def load_current_file(self):
        if 0 <= self.current_file_index < len(self.current_files):
            self.current_data = pd.read_csv(self.current_files[self.current_file_index])
            self.clear_results()
            self.refresh_plot()
            self.update_file_counter()

    def create_gui(self):
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lewa strona - panel kontrolny
        left_frame = ttk.LabelFrame(main_container, text="Control Panel")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Wybór danych
        data_frame = ttk.LabelFrame(left_frame, text="Data Selection")
        data_frame.pack(fill=tk.X, padx=5, pady=5)

        self.folder_combo = ttk.Combobox(data_frame, values=self.data_folders, state='readonly')
        self.folder_combo.pack(fill=tk.X, padx=5, pady=5)
        self.folder_combo.bind('<<ComboboxSelected>>',
                               lambda e: self.load_data_from_folder(e.widget.get()))

        # Nawigacja między plikami
        nav_frame = ttk.Frame(data_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(nav_frame, text="Previous", command=self.prev_file).pack(side=tk.LEFT, padx=5)
        self.file_counter_label = ttk.Label(nav_frame, text="")
        self.file_counter_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_file).pack(side=tk.LEFT, padx=5)

        # Panel optymalizacji
        opt_frame = ttk.LabelFrame(left_frame, text="Optimization Directions")
        opt_frame.pack(fill=tk.X, padx=5, pady=5)

        for criterion in ['x1', 'x2']:
            frame = ttk.Frame(opt_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"{criterion}:").pack(side=tk.LEFT)
            combo = ttk.Combobox(frame, values=['min', 'max'], state='readonly', width=10)
            combo.set(self.optimization_directions[criterion])
            combo.pack(side=tk.RIGHT)
            combo.bind('<<ComboboxSelected>>',
                       lambda e, crit=criterion: self.update_optimization(crit, e.widget.get()))

        # Panel metod
        methods_frame = ttk.LabelFrame(left_frame, text="Methods")
        methods_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(methods_frame, text="TOPSIS",
                   command=self.run_topsis).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(methods_frame, text="RSM",
                   command=self.run_rsm).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(methods_frame, text="UTA-STAR",
                   command=self.run_utastar).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(methods_frame, text="Compare All Methods",
                   command=self.compare_methods).pack(fill=tk.X, padx=5, pady=2)

        # Wyniki
        results_frame = ttk.LabelFrame(left_frame, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_text = tk.Text(results_frame, height=20, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Prawa strona - wizualizacja
        right_frame = ttk.LabelFrame(main_container, text="Visualization")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_file_counter(self):
        if self.current_files:
            current = self.current_file_index + 1
            total = len(self.current_files)
            self.file_counter_label.config(text=f"File {current}/{total}")
        else:
            self.file_counter_label.config(text="No files")

    def next_file(self):
        if self.current_files and self.current_file_index < len(self.current_files) - 1:
            self.current_file_index += 1
            self.load_current_file()

    def prev_file(self):
        if self.current_files and self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_current_file()

    def clear_results(self):
        self.method_results = {}
        self.results_text.delete(1.0, tk.END)

    def update_optimization(self, criterion, direction):
        self.optimization_directions[criterion] = direction
        self.refresh_plot()

    def refresh_plot(self):
        if self.current_data is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Używamy pierwszych dwóch kolumn do wizualizacji
        columns = self.current_data.columns[:2]

        # Rysowanie punktów
        scatter = ax.scatter(self.current_data[columns[0]],
                             self.current_data[columns[1]],
                             c='blue', label='Points', alpha=0.5)

        # Rysowanie wyników metod
        for method_name, results in self.method_results.items():
            if 'ranking' in results:
                colors = results['ranking'] / np.max(results['ranking'])
                scatter = ax.scatter(self.current_data[columns[0]],
                                     self.current_data[columns[1]],
                                     c=colors, cmap='viridis',
                                     label=f'{method_name} Ranking')

        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.legend()

        # Dodanie informacji o aktualnym pliku
        if self.current_files:
            current_file = os.path.basename(self.current_files[self.current_file_index])
            ax.set_title(f"File: {current_file}")

        self.canvas.draw()

    def run_topsis(self):
        if self.current_data is None:
            return

        # Używamy pierwszych dwóch kolumn
        columns = self.current_data.columns[:2]
        data = self.current_data[columns].values

        weights = np.array([0.5, 0.5])

        # Normalizacja
        normalized = data / np.sqrt(np.sum(data ** 2, axis=0))

        # Ważona macierz znormalizowana
        weighted = normalized * weights

        # Ideal i anti-ideal
        is_benefit = [d == 'max' for d in self.optimization_directions.values()]
        ideal = np.max(weighted, axis=0) * np.array(is_benefit) + \
                np.min(weighted, axis=0) * ~np.array(is_benefit)
        anti_ideal = np.min(weighted, axis=0) * np.array(is_benefit) + \
                     np.max(weighted, axis=0) * ~np.array(is_benefit)

        # Odległości
        d_ideal = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
        d_anti = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))

        # Ranking
        ranking = d_anti / (d_ideal + d_anti)

        self.method_results['TOPSIS'] = {'ranking': ranking}
        self.update_results("TOPSIS", ranking)
        self.refresh_plot()

    def run_rsm(self):
        if self.current_data is None:
            return

        # Używamy pierwszych dwóch kolumn
        columns = self.current_data.columns[:2]
        data = self.current_data[columns].values

        # Punkt referencyjny (ideal point)
        ref_point = np.min(data, axis=0) if self.optimization_directions['x1'] == 'min' else np.max(data, axis=0)

        # Obliczanie odległości od punktu referencyjnego
        distances = np.sqrt(np.sum((data - ref_point) ** 2, axis=1))

        # Ranking (mniejsza odległość = lepszy wynik)
        ranking = 1 - (distances / np.max(distances))

        self.method_results['RSM'] = {'ranking': ranking}
        self.update_results("RSM", ranking)
        self.refresh_plot()

    def run_utastar(self):
        if self.current_data is None:
            return

        # Używamy pierwszych dwóch kolumn
        columns = self.current_data.columns[:2]
        data = self.current_data[columns].values

        # Normalizacja danych
        normalized = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

        # Funkcje użyteczności (liniowe)
        if self.optimization_directions['x1'] == 'min':
            normalized[:, 0] = 1 - normalized[:, 0]
        if self.optimization_directions['x2'] == 'min':
            normalized[:, 1] = 1 - normalized[:, 1]

        # Całkowita użyteczność
        ranking = np.mean(normalized, axis=1)

        self.method_results['UTA-STAR'] = {'ranking': ranking}
        self.update_results("UTA-STAR", ranking)
        self.refresh_plot()

    def compare_methods(self):
        self.clear_results()
        self.run_topsis()
        self.run_rsm()
        self.run_utastar()

        # Porównanie rankingów
        self.results_text.insert(tk.END, "\nMethods Comparison:\n")
        for method1 in self.method_results:
            for method2 in self.method_results:
                if method1 < method2:
                    corr = np.corrcoef(self.method_results[method1]['ranking'],
                                       self.method_results[method2]['ranking'])[0, 1]
                    self.results_text.insert(tk.END,
                                             f"{method1} vs {method2} correlation: {corr:.3f}\n")

    def update_results(self, method_name, ranking):
        self.results_text.insert(tk.END, f"\n{method_name} Results:\n")
        self.results_text.insert(tk.END, f"Best solution index: {np.argmax(ranking)}\n")
        self.results_text.insert(tk.END, f"Best solution value: {ranking[np.argmax(ranking)]:.3f}\n")

        # Dodaj top 5 najlepszych rozwiązań
        top_indices = np.argsort(ranking)[-5:][::-1]
        self.results_text.insert(tk.END, "\nTop 5 solutions:\n")
        for i, idx in enumerate(top_indices, 1):
            self.results_text.insert(tk.END,
                                     f"{i}. Index: {idx}, Value: {ranking[idx]:.3f}\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()
