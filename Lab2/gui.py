"""
Pareto GUI application
- Implements three Pareto front algorithms (naive no filter, naive with filter, ideal-point)
- Allows per-dimension min/max selection
- Data generator with distributions: uniform, normal, exponential, poisson
- Visualization for 2D and 3D and table for higher dims
- Load / Save CSV, benchmark simple stats

Run: python pareto_gui.py
Dependencies: Python 3.8+, numpy, matplotlib
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import math
import random
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

# ----------------- Core algorithms (adapted from user's code) -----------------

def get_dimensionality(points):
    if not points:
        return 0
    return len(points[0])


def _normalize_point_for_direction(point, directions):
    """Return a transformed point where every criterion is normalized so that
    smaller-is-better for all coordinates. directions is a list of 'min'/'max'.
    For max criteria we negate the value to convert to minimization.
    """
    return tuple((-v if d == 'max' else v) for v, d in zip(point, directions))


def naive_no_filter(points, directions=None):
    if not points:
        return []
    if directions is None:
        directions = ['min'] * get_dimensionality(points)
    pts = [ _normalize_point_for_direction(p, directions) for p in points ]

    pareto_front = []
    for idx, p1 in enumerate(pts):
        dominated = False
        for jdx, p2 in enumerate(pts):
            if idx != jdx and all(x2 <= x1 for x1, x2 in zip(p1, p2)) and any(x2 < x1 for x1, x2 in zip(p1, p2)):
                dominated = True
                break
        if not dominated:
            pareto_front.append(points[idx])
    return pareto_front


def naive_with_filter(points, directions=None):
    if not points:
        return []
    if directions is None:
        directions = ['min'] * get_dimensionality(points)
    filtered_points = []
    for p in points:
        p_norm = _normalize_point_for_direction(p, directions)
        # check if p is dominated by any already in filtered_points
        if not any(all(x <= y for x, y in zip(_normalize_point_for_direction(other, directions), p_norm)) and any(x < y for x, y in zip(_normalize_point_for_direction(other, directions), p_norm))
                   for other in filtered_points):
            # remove from filtered_points any points dominated by p
            filtered_points = [point for point in filtered_points
                               if not (all(x <= y for x, y in zip(p_norm, _normalize_point_for_direction(point, directions))) and
                                       any(x < y for x, y in zip(p_norm, _normalize_point_for_direction(point, directions))))]
            filtered_points.append(p)
    return filtered_points


def find_ideal_point(points, directions=None):
    if not points:
        return None
    if directions is None:
        directions = ['min'] * get_dimensionality(points)
    dim = get_dimensionality(points)
    # transform points so that smaller is better, then min per coordinate
    transformed = [ _normalize_point_for_direction(p, directions) for p in points ]
    ideal_point = []
    for d in range(dim):
        ideal_point.append(min(point[d] for point in transformed))
    return tuple(ideal_point)


def calculate_distance(point, ideal_point, directions=None):
    # squared Euclidean in transformed space
    p = _normalize_point_for_direction(point, directions) if directions else point
    return sum((p_i - i_i) ** 2 for p_i, i_i in zip(p, ideal_point))


def ideal_point_algorithm(points, directions=None):
    if not points:
        return []
    if directions is None:
        directions = ['min'] * get_dimensionality(points)
    ideal_point = find_ideal_point(points, directions)
    points_with_distances = [(point, calculate_distance(point, ideal_point, directions)) for point in points]
    sorted_points = sorted(points_with_distances, key=lambda x: x[1])

    pareto_front = []
    transformed_front = []
    for point, _ in sorted_points:
        dominated = False
        p_trans = _normalize_point_for_direction(point, directions)
        for p in transformed_front:
            if all(x <= y for x, y in zip(p, p_trans)):
                dominated = True
                break
        if not dominated:
            pareto_front.append(point)
            transformed_front.append(p_trans)
    return pareto_front

# ----------------- Data generator -----------------

def generate_dataset(n_points, n_dims, dist, params):
    """dist: 'uniform', 'normal', 'exponential', 'poisson'
    params is dict of distribution-specific parameters
    """
    rng = np.random.default_rng()
    if dist == 'uniform':
        a = params.get('a', 0.0)
        b = params.get('b', 1.0)
        data = rng.uniform(a, b, size=(n_points, n_dims))
    elif dist == 'normal':
        mu = params.get('mu', 0.0)
        sigma = params.get('sigma', 1.0)
        data = rng.normal(mu, sigma, size=(n_points, n_dims))
    elif dist == 'exponential':
        scale = params.get('scale', 1.0)
        data = rng.exponential(scale, size=(n_points, n_dims))
    elif dist == 'poisson':
        lam = params.get('lam', 2.0)
        data = rng.poisson(lam, size=(n_points, n_dims)).astype(float)
    else:
        raise ValueError('Unknown distribution')
    return [tuple(row.tolist()) for row in data]

# ----------------- Utilities -----------------

def save_to_csv(points, filename):
    if not points:
        return
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'c{i+1}' for i in range(get_dimensionality(points))])
        for p in points:
            writer.writerow(list(p))


def load_from_csv(filename):
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            return []
        # skip header if non-numeric
        start = 0
        try:
            float(rows[0][0])
        except Exception:
            start = 1
        data = []
        for r in rows[start:]:
            if not r:
                continue
            data.append(tuple(float(x) for x in r))
        return data

# ----------------- GUI -----------------

class ParetoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Pareto Front Explorer')
        self.geometry('1100x700')

        self.points = []
        self.directions = []  # per-dimension 'min'/'max'

        self._build_ui()

    def _build_ui(self):
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ---------------- left controls ----------------
        gen_frame = ttk.LabelFrame(left, text='Generator danych')
        gen_frame.pack(fill=tk.X, pady=4)

        ttk.Label(gen_frame, text='Liczba punktów:').grid(row=0, column=0)
        self.n_points_var = tk.IntVar(value=100)
        ttk.Entry(gen_frame, textvariable=self.n_points_var, width=7).grid(row=0, column=1)

        ttk.Label(gen_frame, text='Wymiar:').grid(row=1, column=0)
        self.n_dims_var = tk.IntVar(value=2)
        ttk.Spinbox(gen_frame, from_=1, to=10, textvariable=self.n_dims_var, width=5).grid(row=1, column=1)

        ttk.Label(gen_frame, text='Rozkład:').grid(row=2, column=0)
        self.dist_var = tk.StringVar(value='uniform')
        ttk.Combobox(gen_frame, values=['uniform', 'normal', 'exponential', 'poisson'], textvariable=self.dist_var, state='readonly', width=10).grid(row=2, column=1)

        ttk.Button(gen_frame, text='Generuj', command=self.on_generate).grid(row=3, column=0, columnspan=2, pady=6)

        load_frame = ttk.LabelFrame(left, text='Dane')
        load_frame.pack(fill=tk.X, pady=4)
        ttk.Button(load_frame, text='Wczytaj CSV', command=self.on_load).pack(fill=tk.X, pady=2)
        ttk.Button(load_frame, text='Zapisz wynik (CSV)', command=self.on_save).pack(fill=tk.X, pady=2)

        criteria_frame = ttk.LabelFrame(left, text='Kierunki kryteriów')
        criteria_frame.pack(fill=tk.X, pady=4)
        ttk.Button(criteria_frame, text='Ustaw wg wymiaru', command=self.on_set_directions).pack(fill=tk.X, pady=2)
        self.directions_label = ttk.Label(criteria_frame, text='Nie ustawione')
        self.directions_label.pack(fill=tk.X, pady=2)

        algo_frame = ttk.LabelFrame(left, text='Algorytm i uruchomienie')
        algo_frame.pack(fill=tk.X, pady=4)
        self.algo_var = tk.StringVar(value='naive_no_filter')
        ttk.Radiobutton(algo_frame, text='Naiwny (bez filtracji)', variable=self.algo_var, value='naive_no_filter').pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text='Naiwny (z filtracją)', variable=self.algo_var, value='naive_with_filter').pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text='Punkt idealny', variable=self.algo_var, value='ideal_point').pack(anchor=tk.W)
        ttk.Button(algo_frame, text='Wylicz front Pareto', command=self.on_solve).pack(fill=tk.X, pady=6)

        benchmark_frame = ttk.LabelFrame(left, text='Benchmark (prosty)')
        benchmark_frame.pack(fill=tk.X, pady=4)
        ttk.Button(benchmark_frame, text='Uruchom prosty benchmark', command=self.on_benchmark).pack(fill=tk.X)

        # ---------------- right: plotting and table ----------------
        top_right = ttk.Frame(right)
        top_right.pack(fill=tk.BOTH, expand=True)
        bottom_right = ttk.Frame(right)
        bottom_right.pack(fill=tk.X)

        # Figure
        self.fig = Figure(figsize=(6,5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=top_right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Table-like listbox
        self.table = ttk.Treeview(bottom_right, columns=('index', 'point'), show='headings')
        self.table.heading('index', text='Nr')
        self.table.heading('point', text='Wektor')
        self.table.pack(fill=tk.BOTH, expand=True)

    def on_generate(self):
        n = max(1, self.n_points_var.get())
        d = max(1, self.n_dims_var.get())
        dist = self.dist_var.get()
        params = {}
        # Use reasonable defaults
        if dist == 'uniform':
            params = {'a': 0.0, 'b': 10.0}
        elif dist == 'normal':
            params = {'mu': 0.0, 'sigma': 3.0}
        elif dist == 'exponential':
            params = {'scale': 1.0}
        elif dist == 'poisson':
            params = {'lam': 3.0}
        self.points = generate_dataset(n, d, dist, params)
        # default directions (all minimize)
        self.directions = ['min'] * d
        self.directions_label.config(text=str(self.directions))
        self._refresh_view()

    def on_load(self):
        fn = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if not fn:
            return
        try:
            self.points = load_from_csv(fn)
            if self.points:
                d = get_dimensionality(self.points)
                self.directions = ['min'] * d
                self.directions_label.config(text=str(self.directions))
            self._refresh_view()
        except Exception as e:
            messagebox.showerror('Błąd', f'Nie udało się wczytać pliku:\n{e}')

    def on_save(self):
        if not self.points:
            messagebox.showinfo('Info', 'Brak punktów do zapisania')
            return
        fn = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')])
        if not fn:
            return
        save_to_csv(self.points, fn)
        messagebox.showinfo('Zapis', 'Zapisano dane do pliku')

    def on_set_directions(self):
        d = max(1, self.n_dims_var.get())
        win = tk.Toplevel(self)
        win.title('Ustaw kierunki')
        vars = []
        for i in range(d):
            v = tk.StringVar(value='min' if i < len(self.directions) and self.directions else 'min')
            vars.append(v)
            ttk.Label(win, text=f'Wymiar {i+1}').grid(row=i, column=0)
            ttk.Combobox(win, values=['min','max'], textvariable=v, state='readonly', width=6).grid(row=i, column=1)
        def apply():
            self.directions = [v.get() for v in vars]
            self.directions_label.config(text=str(self.directions))
            win.destroy()
        ttk.Button(win, text='Zastosuj', command=apply).grid(row=d, column=0, columnspan=2, pady=6)

    def on_solve(self):
        if not self.points:
            messagebox.showinfo('Info', 'Brak punktów')
            return
        algo = self.algo_var.get()
        if algo == 'naive_no_filter':
            pf = naive_no_filter(self.points, self.directions)
        elif algo == 'naive_with_filter':
            pf = naive_with_filter(self.points, self.directions)
        else:
            pf = ideal_point_algorithm(self.points, self.directions)
        self._show_results(pf)

    def on_benchmark(self):
        if not self.points:
            messagebox.showinfo('Info', 'Wygeneruj lub wczytaj dane najpierw')
            return
        import time
        results = {}
        for name, func in [('naive_no_filter', naive_no_filter), ('naive_with_filter', naive_with_filter), ('ideal_point', ideal_point_algorithm)]:
            t0 = time.perf_counter()
            pf = func(self.points, self.directions)
            t1 = time.perf_counter()
            results[name] = {'time_ms': (t1-t0)*1000.0, 'pareto_count': len(pf)}
        s = '\n'.join(f"{k}: czas={v['time_ms']:.2f} ms, |P|={v['pareto_count']}" for k,v in results.items())
        messagebox.showinfo('Benchmark', s)

    def _show_results(self, pareto_front):
        # mark Pareto points and visualize
        self.pareto = set(pareto_front)
        self._refresh_view()

    def _refresh_view(self):
        # update table
        for i in self.table.get_children():
            self.table.delete(i)
        for idx, p in enumerate(self.points, start=1):
            tag = ()
            if hasattr(self, 'pareto') and p in self.pareto:
                tag = ('pareto',)
            self.table.insert('', 'end', values=(idx, p), tags=tag)
        self.table.tag_configure('pareto', background='#c8f7c5')

        # update plot
        self.ax.clear()
        d = get_dimensionality(self.points) if self.points else 0
        if d == 1:
            xs = [p[0] for p in self.points]
            ys = list(range(len(xs)))
            self.ax.plot(ys, xs, marker='o')
        elif d == 2:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax.scatter(xs, ys, label='points')
            if hasattr(self, 'pareto'):
                px = [p[0] for p in self.pareto]
                py = [p[1] for p in self.pareto]
                self.ax.scatter(px, py, s=80, edgecolors='k', label='Pareto', facecolors='none')
            self.ax.set_xlabel('c1')
            self.ax.set_ylabel('c2')
            self.ax.legend()
        elif d == 3:
            # create a 3D axes
            self.fig.delaxes(self.ax)
            self.ax = self.fig.add_subplot(111, projection='3d')
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            zs = [p[2] for p in self.points]
            self.ax.scatter(xs, ys, zs)
            if hasattr(self, 'pareto'):
                px = [p[0] for p in self.pareto]
                py = [p[1] for p in self.pareto]
                pz = [p[2] for p in self.pareto]
                self.ax.scatter(px, py, pz, s=80)
            self.ax.set_xlabel('c1')
            self.ax.set_ylabel('c2')
            self.ax.set_zlabel('c3')
        else:
            # higher dims: show simple projection (pairwise first two dims)
            if d >= 2:
                xs = [p[0] for p in self.points]
                ys = [p[1] for p in self.points]
                self.ax.scatter(xs, ys)
                if hasattr(self, 'pareto'):
                    px = [p[0] for p in self.pareto]
                    py = [p[1] for p in self.pareto]
                    self.ax.scatter(px, py, s=80)
                self.ax.set_xlabel('c1')
                self.ax.set_ylabel('c2')
                self.ax.set_title(f'Projekcja: pierwszy i drugi wymiar (wymiarów={d})')
            else:
                self.ax.text(0.5, 0.5, 'Brak danych')
        self.canvas.draw()


if __name__ == '__main__':
    app = ParetoApp()
    app.mainloop()
