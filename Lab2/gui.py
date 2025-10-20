import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import numpy as np
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ----------------- Core Pareto algorithms -----------------

def get_dimensionality(points):
    return len(points[0]) if points else 0

def _normalize_point_for_direction(point, directions):
    return tuple((-v if d=='max' else v) for v,d in zip(point, directions))

def naive_no_filter(points, directions=None):
    if not points:
        return [], 0
    if directions is None:
        directions = ['min']*get_dimensionality(points)
    pts = [_normalize_point_for_direction(p, directions) for p in points]

    pareto_front = []
    ops = 0
    for idx, p1 in enumerate(pts):
        dominated = False
        for jdx, p2 in enumerate(pts):
            ops += 1
            if idx != jdx and all(x2 <= x1 for x1,x2 in zip(p1,p2)) and any(x2 < x1 for x1,x2 in zip(p1,p2)):
                dominated = True
                break
        if not dominated:
            pareto_front.append(points[idx])
    return pareto_front, ops

def naive_with_filter(points, directions=None):
    if not points:
        return [],0
    if directions is None:
        directions = ['min']*get_dimensionality(points)
    filtered_points = []
    ops = 0
    for p in points:
        p_norm = _normalize_point_for_direction(p, directions)
        dominated = False
        for other in filtered_points:
            ops += 1
            o_norm = _normalize_point_for_direction(other, directions)
            if all(x <= y for x,y in zip(o_norm, p_norm)) and any(x<y for x,y in zip(o_norm,p_norm)):
                dominated = True
                break
        if not dominated:
            new_filtered = []
            for point in filtered_points:
                ops += 1
                pt_norm = _normalize_point_for_direction(point, directions)
                if not (all(x<=y for x,y in zip(p_norm,pt_norm)) and any(x<y for x,y in zip(p_norm,pt_norm))):
                    new_filtered.append(point)
            filtered_points = new_filtered + [p]
    return filtered_points, ops

def find_ideal_point(points, directions=None):
    if not points:
        return None
    if directions is None:
        directions = ['min']*get_dimensionality(points)
    transformed = [_normalize_point_for_direction(p,directions) for p in points]
    return tuple(min(p[i] for p in transformed) for i in range(len(points[0])))

def calculate_distance(point, ideal_point, directions=None):
    p = _normalize_point_for_direction(point,directions) if directions else point
    return sum((p_i-i_i)**2 for p_i,i_i in zip(p,ideal_point))

def ideal_point_algorithm(points, directions=None):
    if not points:
        return [],0
    if directions is None:
        directions = ['min']*get_dimensionality(points)
    ideal_point = find_ideal_point(points,directions)
    points_with_distances = [(p, calculate_distance(p,ideal_point,directions)) for p in points]
    sorted_points = sorted(points_with_distances, key=lambda x:x[1])

    pareto_front = []
    transformed_front = []
    ops = 0
    for p,_ in sorted_points:
        p_trans = _normalize_point_for_direction(p,directions)
        dominated=False
        for q in transformed_front:
            ops+=1
            if all(x<=y for x,y in zip(q,p_trans)):
                dominated=True
                break
        if not dominated:
            pareto_front.append(p)
            transformed_front.append(p_trans)
    return pareto_front, ops

# ----------------- Data generator -----------------

def generate_dataset(n_points, n_dims, dist, params):
    rng = np.random.default_rng()
    # Validate parameters
    for k,v in params.items():
        if not isinstance(v,(int,float)):
            raise ValueError(f"Parametr {k} musi być liczbowy")
    if dist=='uniform':
        a=params.get('a',0.0)
        b=params.get('b',1.0)
        if a>=b:
            raise ValueError("Parametry uniform: a < b wymagane")
        data=rng.uniform(a,b,(n_points,n_dims))
    elif dist=='normal':
        mu=params.get('mu',0.0)
        sigma=params.get('sigma',1.0)
        if sigma<=0:
            raise ValueError("sigma musi być >0")
        data=rng.normal(mu,sigma,(n_points,n_dims))
    elif dist=='exponential':
        scale=params.get('scale',1.0)
        if scale<=0:
            raise ValueError("scale musi być >0")
        data=rng.exponential(scale,(n_points,n_dims))
    elif dist=='poisson':
        lam=params.get('lam',2.0)
        if lam<=0:
            raise ValueError("lam musi być >0")
        data=rng.poisson(lam,(n_points,n_dims)).astype(float)
    else:
        raise ValueError("Nieznany rozkład")
    return [tuple(row.tolist()) for row in data]

# ----------------- File utilities -----------------

def save_to_csv(points, filename, runtime_ms=None, n_ops=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not points:
        return
    with open(filename,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow([f'c{i+1}' for i in range(len(points[0]))])
        for p in points:
            writer.writerow(list(p))
        if runtime_ms is not None:
            writer.writerow([])
            writer.writerow(['Execution time (ms):', f'{runtime_ms:.3f}'])
        if n_ops is not None:
            writer.writerow(['Number of operations:', n_ops])

def load_from_csv(filename):
    with open(filename,'r',newline='') as f:
        reader=csv.reader(f)
        rows=list(reader)
        if not rows:
            return []
        start=0
        try:
            float(rows[0][0])
        except:
            start=1
        return [tuple(float(x) for x in r) for r in rows[start:] if r]

# ----------------- GUI -----------------

class ParetoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pareto Front Explorer")
        self.geometry("1200x750")
        self.points=[]
        self.directions=[]
        self.last_runtime_ms=None
        self._build_ui()

    def _build_ui(self):
        left=ttk.Frame(self)
        left.pack(side=tk.LEFT,fill=tk.Y,padx=8,pady=8)
        right=ttk.Frame(self)
        right.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True,padx=8,pady=8)

        # --- Generator ---
        gen_frame=ttk.LabelFrame(left,text="Generator danych")
        gen_frame.pack(fill=tk.X,pady=4)
        ttk.Label(gen_frame,text="Liczba punktów:").grid(row=0,column=0)
        self.n_points_var=tk.IntVar(value=100)
        ttk.Spinbox(gen_frame,from_=1,to=10000,textvariable=self.n_points_var,width=8).grid(row=0,column=1)
        ttk.Label(gen_frame,text="Wymiar:").grid(row=1,column=0)
        self.n_dims_var=tk.IntVar(value=2)
        ttk.Spinbox(gen_frame,from_=1,to=10,textvariable=self.n_dims_var,width=5).grid(row=1,column=1)
        ttk.Label(gen_frame,text="Rozkład:").grid(row=2,column=0)
        self.dist_var=tk.StringVar(value='uniform')
        dist_box=ttk.Combobox(gen_frame,values=['uniform','normal','exponential','poisson'],
                              textvariable=self.dist_var,state='readonly',width=12)
        dist_box.grid(row=2,column=1)
        dist_box.bind("<<ComboboxSelected>>",lambda e:self._update_param_fields())
        self.param_frame=ttk.LabelFrame(gen_frame,text="Parametry rozkładu")
        self.param_frame.grid(row=3,column=0,columnspan=2,pady=4)
        self.param_entries={}
        self._update_param_fields()
        ttk.Button(gen_frame,text="Generuj",command=self.on_generate).grid(row=4,column=0,columnspan=2,pady=6)

        # --- Dane ---
        load_frame=ttk.LabelFrame(left,text="Dane")
        load_frame.pack(fill=tk.X,pady=4)
        ttk.Button(load_frame,text="Wczytaj CSV",command=self.on_load).pack(fill=tk.X,pady=2)
        ttk.Button(load_frame,text="Zapisz wynik",command=self.on_save_results).pack(fill=tk.X,pady=2)

        # --- Kierunki ---
        crit_frame=ttk.LabelFrame(left,text="Kierunki kryteriów")
        crit_frame.pack(fill=tk.X,pady=4)
        ttk.Button(crit_frame,text="Ustaw wg wymiaru",command=self.on_set_directions).pack(fill=tk.X,pady=2)
        self.directions_label=ttk.Label(crit_frame,text="Nie ustawione")
        self.directions_label.pack(fill=tk.X)

        # --- Algorytm ---
        algo_frame=ttk.LabelFrame(left,text="Algorytm")
        algo_frame.pack(fill=tk.X,pady=4)
        self.algo_var=tk.StringVar(value='naive_no_filter')
        for txt,val in [("Naiwny (bez filtracji)","naive_no_filter"),
                        ("Naiwny (z filtracją)","naive_with_filter"),
                        ("Punkt idealny","ideal_point")]:
            ttk.Radiobutton(algo_frame,text=txt,variable=self.algo_var,value=val).pack(anchor=tk.W)
        ttk.Button(algo_frame,text="Wylicz front Pareto",command=self.on_solve).pack(fill=tk.X,pady=6)

        # --- Benchmark ---
        bench_frame=ttk.LabelFrame(left,text="Benchmark")
        bench_frame.pack(fill=tk.X,pady=4)
        ttk.Button(bench_frame,text="Uruchom benchmark",command=self.on_benchmark).pack(fill=tk.X,pady=4)

        # --- Wykres ---
        self.fig=Figure(figsize=(6,5),dpi=100)
        self.ax=self.fig.add_subplot(111)
        self.canvas=FigureCanvasTkAgg(self.fig,master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)

    # ----------------- Generator -----------------

    def _update_param_fields(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        dist=self.dist_var.get()
        self.param_entries={}
        params=[]
        if dist=='uniform':
            params=[('a',0,100),('b',0,100)]
        elif dist=='normal':
            params=[('mu',-100,100),('sigma',0.1,50)]
        elif dist=='exponential':
            params=[('scale',0.1,50)]
        elif dist=='poisson':
            params=[('lam',0.1,50)]
        for i,(name,low,high) in enumerate(params):
            ttk.Label(self.param_frame,text=f"{name}:").grid(row=i,column=0)
            v=tk.DoubleVar(value=(low+high)/2)
            ttk.Spinbox(self.param_frame,from_=low,to=high,increment=0.1,textvariable=v,width=8).grid(row=i,column=1)
            self.param_entries[name]=v

    def on_generate(self):
        n=self.n_points_var.get()
        d=self.n_dims_var.get()
        dist=self.dist_var.get()
        params={k:v.get() for k,v in self.param_entries.items()}
        try:
            self.points=generate_dataset(n,d,dist,params)
        except Exception as e:
            messagebox.showerror("Błąd parametrów",str(e))
            return
        self.directions=['min']*d
        self.directions_label.config(text=str(self.directions))
        os.makedirs("data",exist_ok=True)
        param_str="_".join(f"{k}{params[k]}" for k in params)
        save_to_csv(self.points,f"data/dataset_{param_str}.csv")
        self._refresh_view()

    # ----------------- Core actions -----------------

    def on_set_directions(self):
        if not self.points:
            return
        d=len(self.points[0])
        win=tk.Toplevel(self)
        win.title("Ustaw kierunki")
        vars_=[]
        for i in range(d):
            ttk.Label(win,text=f"Wymiar {i+1}").grid(row=i,column=0)
            v=tk.StringVar(value=self.directions[i] if i<len(self.directions) else 'min')
            vars_.append(v)
            ttk.Combobox(win,values=['min','max'],textvariable=v,state='readonly',width=6).grid(row=i,column=1)
        def apply():
            self.directions=[v.get() for v in vars_]
            self.directions_label.config(text=str(self.directions))
            win.destroy()
        ttk.Button(win,text="Zastosuj",command=apply).grid(row=d,column=0,columnspan=2,pady=6)

    def on_solve(self):
        if not self.points:
            return
        algo_name=self.algo_var.get()
        start=time.perf_counter()
        if algo_name=='naive_no_filter':
            pf,n_ops=naive_no_filter(self.points,self.directions)
        elif algo_name=='naive_with_filter':
            pf,n_ops=naive_with_filter(self.points,self.directions)
        else:
            pf,n_ops=ideal_point_algorithm(self.points,self.directions)
        end=time.perf_counter()
        self.last_runtime_ms=(end-start)*1000.0
        self.pareto=set(pf)
        self._refresh_view()
        param_str="_".join(f"{k}{v.get()}" for k,v in self.param_entries.items())
        save_to_csv(list(self.pareto),f"data/pareto_result_{param_str}.csv",runtime_ms=self.last_runtime_ms,n_ops=n_ops)
        messagebox.showinfo("Zakończono",f"Front Pareto obliczony w {self.last_runtime_ms:.3f} ms\nLiczba operacji: {n_ops}")

    def on_load(self):
        fn=filedialog.askopenfilename(filetypes=[('CSV files','*.csv')])
        if not fn:
            return
        self.points=load_from_csv(fn)
        if self.points:
            self.directions=['min']*len(self.points[0])
            self.directions_label.config(text=str(self.directions))
        self._refresh_view()

    def on_save_results(self):
        if not hasattr(self,'pareto') or not self.pareto:
            messagebox.showinfo("Info","Brak wyników do zapisania")
            return
        param_str="_".join(f"{k}{v.get()}" for k,v in self.param_entries.items())
        save_to_csv(list(self.pareto),f"data/pareto_result_{param_str}.csv",runtime_ms=self.last_runtime_ms)
        messagebox.showinfo("Zapisano","Wyniki zapisano w katalogu data/")

    # ----------------- Visualization -----------------

    def _refresh_view(self):
        if not self.points:
            return
        d=len(self.points[0])
        self.fig.clf()
        if d>=3:
            self.ax=self.fig.add_subplot(111,projection='3d')
            xs,ys,zs=zip(*[(p[0],p[1],p[2]) for p in self.points])
            self.ax.scatter(xs,ys,zs,alpha=0.5,label='punkty')
            if hasattr(self,'pareto') and self.pareto:
                ppx,ppy,ppz=zip(*[(p[0],p[1],p[2]) for p in self.pareto])
                self.ax.scatter(ppx,ppy,ppz,color='red',label='Pareto')
            self.ax.set_xlabel('c1')
            self.ax.set_ylabel('c2')
            self.ax.set_zlabel('c3')
            self.ax.legend()
        elif d==2:
            self.ax=self.fig.add_subplot(111)
            xs,ys=zip(*self.points)
            self.ax.scatter(xs,ys,alpha=0.5,label='punkty')
            if hasattr(self,'pareto') and self.pareto:
                px,py=zip(*self.pareto)
                self.ax.scatter(px,py,color='red',label='Pareto')
            self.ax.set_xlabel('c1')
            self.ax.set_ylabel('c2')
            self.ax.legend()
        else:
            self.ax=self.fig.add_subplot(111)
            xs=[p[0] for p in self.points]
            self.ax.scatter(xs,[0]*len(xs))
            self.ax.set_title('1D')
        self.canvas.draw()

    # ----------------- Benchmark -----------------

    def on_benchmark(self):
        if not self.points:
            messagebox.showinfo("Info","Najpierw wygeneruj dane.")
            return
        algos={'naive_no_filter':naive_no_filter,'naive_with_filter':naive_with_filter,'ideal_point':ideal_point_algorithm}
        results=[]
        param_str="_".join(f"{k}{v.get()}" for k,v in self.param_entries.items())
        for name,func in algos.items():
            start=time.perf_counter()
            pf,n_ops=func(self.points,self.directions)
            end=time.perf_counter()
            runtime_ms=(end-start)*1000.0
            results.append((name,len(pf),runtime_ms,n_ops))
            save_to_csv(pf,f"data/{name}_pareto_{param_str}.csv",runtime_ms=runtime_ms,n_ops=n_ops)
        os.makedirs("data",exist_ok=True)
        with open(f"data/benchmark_results_{param_str}.csv",'w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(["Algorytm","Rozmiar frontu","Czas [ms]","Liczba operacji"])
            for r in results:
                writer.writerow(r)
        win=tk.Toplevel(self)
        win.title("Wyniki benchmarku")
        tree=ttk.Treeview(win,columns=("alg","size","time","ops"),show="headings",height=6)
        for col,text in zip(("alg","size","time","ops"),("Algorytm","Rozmiar frontu","Czas [ms]","Liczba operacji")):
            tree.heading(col,text=text)
        tree.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)
        for r in results:
            tree.insert("",tk.END,values=r)

if __name__=='__main__':
    app=ParetoApp()
    app.mainloop()
