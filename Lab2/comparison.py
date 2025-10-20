import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# Pomocnicze funkcje
# ----------------------------------------------------------

def load_csv_points(filename):
    """Wczytuje punkty z pliku CSV, ignorując puste wiersze i metadane."""
    points = []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0]:
                continue
            try:
                floats = [float(x) for x in row if x.strip()]
                if floats:
                    points.append(tuple(floats))
            except ValueError:
                continue
    return points


def extract_metadata(filename):
    """Odczytuje runtime i liczbę operacji z końca pliku CSV."""
    runtime, ops = None, None
    with open(filename, 'r', newline='') as f:
        for line in f:
            if "Execution time" in line:
                try:
                    runtime = float(re.findall(r"[\d.]+", line)[0])
                except:
                    pass
            elif "Number of operations" in line:
                try:
                    ops = int(re.findall(r"\d+", line)[0])
                except:
                    pass
    return runtime, ops


# ----------------------------------------------------------
# Analiza pojedynczego folderu eksperymentu
# ----------------------------------------------------------

def analyze_experiment(folder):
    """Zbiera dane o rozmiarze frontu, czasie i liczbie operacji z folderu."""
    summary = []

    for algo in ['naive_no_filter', 'naive_with_filter', 'ideal_point']:
        algo_files = sorted(
            [f for f in os.listdir(folder)
             if f.startswith(algo + "_pareto_") and f.endswith(".csv")]
        )

        sizes, times, ops_list = [], [], []

        for file in algo_files:
            filepath = os.path.join(folder, file)
            points = load_csv_points(filepath)
            size = len(points)
            runtime, ops = extract_metadata(filepath)
            if size > 0:
                sizes.append(size)
            if runtime is not None:
                times.append(runtime)
            if ops is not None:
                ops_list.append(ops)

        summary.append({
            'algorithm': algo,
            'avg_size': np.mean(sizes) if sizes else 0,
            'total_size': np.sum(sizes) if sizes else 0,
            'total_ops': np.sum(ops_list) if ops_list else 0,
            'total_time': np.sum(times) if times else 0,
        })

    return summary


# ----------------------------------------------------------
# Wykresy porównawcze dla folderu eksperymentu
# ----------------------------------------------------------

def plot_comparison(summary, folder):
    folder_name = os.path.basename(folder)
    plot_dir = os.path.join("plots", folder_name)
    os.makedirs(plot_dir, exist_ok=True)

    algos = [s['algorithm'] for s in summary]
    times = [s['total_time'] for s in summary]
    ops = [s['total_ops'] for s in summary]
    sizes = [s['avg_size'] for s in summary]

    # --- Czas wykonania ---
    plt.figure()
    plt.bar(algos, times)
    plt.ylabel("Czas całkowity [ms]")
    plt.title(f"Czas wykonania - {folder_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "time.png"))
    plt.close()

    # --- Liczba operacji ---
    plt.figure()
    plt.bar(algos, ops, color='orange')
    plt.ylabel("Liczba operacji")
    plt.title(f"Liczba operacji - {folder_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ops.png"))
    plt.close()

    # --- Średni rozmiar frontu Pareto ---
    plt.figure()
    plt.bar(algos, sizes, color='green')
    plt.ylabel("Średni rozmiar frontu Pareto")
    plt.title(f"Średni rozmiar frontu - {folder_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "size.png"))
    plt.close()

    print(f"📊 Wykresy zapisane do: {plot_dir}/")


# ----------------------------------------------------------
# Porównanie frontów Pareto (tylko jeśli różne)
# ----------------------------------------------------------

def compare_pareto_sets(folder):
    """Porównuje fronty Pareto z pierwszego zbioru (dataset_1) dla różnych algorytmów."""
    folder_name = os.path.basename(folder)
    plot_dir = os.path.join("plots", folder_name)
    os.makedirs(plot_dir, exist_ok=True)

    paretos = {}
    for algo in ['naive_no_filter', 'naive_with_filter', 'ideal_point']:
        file = os.path.join(folder, f"{algo}_pareto_1.csv")
        if os.path.exists(file):
            pts = load_csv_points(file)
            if pts:
                # zaokrąglamy lekko, by uniknąć błędów floatów
                paretos[algo] = {tuple(round(x, 6) for x in p) for p in pts}

    if not paretos:
        print(f"⚠️ Brak plików Pareto do porównania w {folder_name}.")
        return

    # porównaj fronty — czy wszystkie identyczne
    all_sets = list(paretos.values())
    identical = all(all_sets[0] == s for s in all_sets[1:])

    if identical:
        # wszystkie fronty identyczne
        print(f"✅ Fronty Pareto są identyczne w {folder_name}. Rysuję tylko przykładowy wykres.")
        ref_algo = next(iter(paretos.keys()))
        pts = list(paretos[ref_algo])
        plt.figure()
        plt.title(f"Front Pareto (identyczny dla wszystkich) - {folder_name}")
        if len(pts[0]) >= 2:
            xs, ys = zip(*pts)
            plt.scatter(xs, ys, color='blue', label=ref_algo)
        else:
            xs = [p[0] for p in pts]
            plt.scatter(xs, [0]*len(xs), color='blue', label=ref_algo)
        plt.xlabel("c1")
        plt.ylabel("c2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "pareto_identical.png"))
        plt.close()
        return

    # fronty się różnią → rysujemy porównanie
    print(f"🔍 Różnice między frontami Pareto wykryte w {folder_name}. Generuję wykres porównawczy.")
    plt.figure()
    plt.title(f"Porównanie frontów Pareto (dataset_1) - {folder_name}")
    colors = {'naive_no_filter': 'blue', 'naive_with_filter': 'orange', 'ideal_point': 'green'}

    for algo, pts in paretos.items():
        d = len(next(iter(pts)))
        if d >= 2:
            xs, ys = zip(*pts)
            plt.scatter(xs, ys, alpha=0.6, label=algo, color=colors.get(algo, 'gray'))
        else:
            xs = [p[0] for p in pts]
            plt.scatter(xs, [0]*len(xs), alpha=0.6, label=algo, color=colors.get(algo, 'gray'))

    plt.xlabel("c1")
    plt.ylabel("c2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pareto_diff.png"))
    plt.close()
    print(f"📈 Zapisano wykres różnic do: {plot_dir}/pareto_diff.png")


# ----------------------------------------------------------
# Główne wykonanie
# ----------------------------------------------------------

def main():
    os.makedirs("plots", exist_ok=True)
    data_root = "data"
    folders = [os.path.join(data_root, f)
               for f in os.listdir(data_root)
               if os.path.isdir(os.path.join(data_root, f))]

    if not folders:
        print("❌ Brak folderów eksperymentów w ./data/")
        return

    for folder in folders:
        folder_name = os.path.basename(folder)
        print(f"\n🔍 Analiza folderu: {folder_name}")
        summary = analyze_experiment(folder)

        for s in summary:
            print(f"  - {s['algorithm']}: czas={s['total_time']:.2f} ms, "
                  f"operacje={s['total_ops']}, średni rozmiar={s['avg_size']:.1f}")

        plot_comparison(summary, folder)
        compare_pareto_sets(folder)


if __name__ == "__main__":
    main()
