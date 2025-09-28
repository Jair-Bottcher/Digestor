# pip install pandas openpyxl plotly scipy

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import plotly.express as px

# =========================================================
# 1) Leitura do arquivo
# =========================================================
try:
    pasta = Path(__file__).parent
except NameError:
    pasta = Path.cwd()   # Jupyter/Interactive não tem __file__

arquivo = pasta / "digestor_sintetico_batelada_final.xlsx"

print(f"Lendo: {arquivo.name}")

try:
    df = pd.read_excel(arquivo, sheet_name="digestor_sintetico", engine="openpyxl")
except ValueError:
    df = pd.read_excel(arquivo, engine="openpyxl")

print(df.shape)      # (linhas, colunas)
print(df.columns)    # nomes das colunas
print(df.head(3))    # amostra

if "ID_Batelada" in df.columns:
    df["ID_Batelada"] = df["ID_Batelada"].astype("category")

numeric_df = df.select_dtypes(include="number")

# =========================================================
# 2) SIMULAÇÃO (EDO simples) + métricas
# =========================================================
def simular_digestor(temp_bulk_c, carga_alcalina, sulfidez, tempo_coccao_min):
    """
    EDO simplificada: dL/dt = -k * L, com k dependente de Temp, Carga e Sulfidez.
    Retorna (kappa_sim, L_end).
    """
    T_ref = 155.0
    a, b, c = 0.04, 0.03, 0.20
    k0 = 0.015

    k = k0 * np.exp(a * (temp_bulk_c - T_ref)) \
        * (1.0 + c * (carga_alcalina - 18)/18.0) \
        * (1.0 + b * (sulfidez - 30)/30.0)

    def dLdt(t, L):
        return -k * L

    L0 = [100.0]
    t_span = (0.0, float(tempo_coccao_min))
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(dLdt, t_span, L0, t_eval=t_eval, method="RK45")
    L_end = sol.y[0, -1]

    theta = 3.3
    kappa_sim = float(np.clip(L_end / theta, 10, 30))
    return kappa_sim, float(L_end)

def rendimento_estimado(kappa, relacao_licor_madeira=5.5):
    return float(np.clip(48 + 0.30*(kappa - 18) - 0.05*(relacao_licor_madeira - 5.5), 40, 55))

# =========================================================
# 3) Problema de otimização
# =========================================================
vars_decisao = ["Temp_Bulk_C", "Carga_Alcalina_madeira", "Sulfidez", "Tempo_Coccao_min"]
for v in vars_decisao:
    if v not in numeric_df.columns:
        raise ValueError(f"Coluna necessária '{v}' não encontrada no arquivo.")

def bounds(col):
    return float(numeric_df[col].min()), float(numeric_df[col].max())

bounds_dict = {v: bounds(v) for v in vars_decisao}

kappa_sp = float(numeric_df["Kappa_Number"].median()) if "Kappa_Number" in numeric_df.columns else 18.0
rel_licor_col = "Relacao_Licor_Madeira_m3_BDt"
rel_licor_med = float(numeric_df[rel_licor_col].median()) if rel_licor_col in numeric_df.columns else 5.5

def avaliar_objetivos(x):
    Tbulk, carga, sulf, tcoc = x
    kappa_sim, L_end = simular_digestor(Tbulk, carga, sulf, tcoc)
    rend_sim = rendimento_estimado(kappa_sim, relacao_licor_madeira=rel_licor_med)
    f1 = (kappa_sim - kappa_sp)**2      # minimizar
    f2 = -rend_sim                      # minimizar (equivalente a maximizar rendimento)
    return np.array([f1, f2], dtype=float), {"Kappa": kappa_sim, "Rendimento": rend_sim, "L_end": L_end}

# =========================================================
# 4) SPEA2 — implementação didática
# =========================================================
rng = np.random.default_rng(123)

def amostrar_inicial(N):
    return [np.array([rng.uniform(*bounds_dict[v]) for v in vars_decisao], dtype=float) for _ in range(N)]

def reparar(x):
    x_corr = x.copy()
    for i, v in enumerate(vars_decisao):
        lo, hi = bounds_dict[v]
        x_corr[i] = np.clip(x_corr[i], lo, hi)
    return x_corr

def cruzamento_aritmetico(p1, p2, pc=0.9):
    if rng.random() > pc:
        return p1.copy(), p2.copy()
    alpha = rng.uniform(0, 1, size=len(p1))
    c1 = alpha*p1 + (1-alpha)*p2
    c2 = alpha*p2 + (1-alpha)*p1
    return reparar(c1), reparar(c2)

def mutacao_gauss(x, pm=0.2, sigma_frac=0.05):
    y = x.copy()
    for i, v in enumerate(vars_decisao):
        if rng.random() < pm:
            lo, hi = bounds_dict[v]
            sigma = (hi - lo)*sigma_frac
            y[i] = y[i] + rng.normal(0, sigma)
    return reparar(y)

def domina(fa, fb):
    return np.all(fa <= fb) and np.any(fa < fb)

def calcular_fitness_SPEA2(pop_objs):
    F = np.array(pop_objs)  # N x M
    N = F.shape[0]
    S = np.zeros(N, dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j and domina(F[i], F[j]):
                S[i] += 1.0
    R = np.zeros(N, dtype=float)
    for i in range(N):
        dominadores = [j for j in range(N) if j != i and domina(F[j], F[i])]
        R[i] = np.sum(S[dominadores]) if dominadores else 0.0
    k_idx = int(np.sqrt(N)) or 1
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        di = np.linalg.norm(F - F[i], axis=1)
        D[i] = di
        D[i, i] = np.inf
    d_k = np.partition(D, k_idx, axis=1)[:, k_idx]
    density = 1.0 / (d_k + 2.0)
    fitness = R + density
    return fitness, density

def truncar_por_aglomeracao(indices, F, Nmax):
    idx = indices.copy()
    while len(idx) > Nmax:
        M = len(idx)
        subF = F[idx]
        D = np.full((M, M), np.inf, dtype=float)
        for i in range(M):
            for j in range(i+1, M):
                d = np.linalg.norm(subF[i] - subF[j])
                D[i, j] = D[j, i] = d
        i_min, j_min = np.unravel_index(np.argmin(D), D.shape)
        remove = i_min if np.mean(D[i_min]) < np.mean(D[j_min]) else j_min
        del idx[remove]
    return idx

def selecao_torneio(indices, fitness, k=2, size=1):
    esc = []
    for _ in range(size):
        cand = rng.choice(indices, size=k, replace=True)
        best = cand[np.argmin(fitness[cand])]
        esc.append(best)
    return esc

# =========================================================
# 5) EXECUÇÃO DO SPEA2
# =========================================================
N = 60
N_arch = 30
T = 25
pc = 0.9
pm = 0.2

P = amostrar_inicial(N)
objs, extras = [], []
for x in P:
    f, m = avaliar_objetivos(x)
    objs.append(f); extras.append(m)

A, A_objs, A_ext = [], [], []

print("\n=== Iniciando SPEA2 ===")
for gen in range(1, T+1):
    P_all = (A + P) if len(A) > 0 else P
    Objs_all = (A_objs + objs) if len(A_objs) > 0 else objs
    Ext_all  = (A_ext + extras) if len(A_ext) > 0 else extras

    fitness, density = calcular_fitness_SPEA2(Objs_all)
    idx_all = list(range(len(P_all)))
    ord_all = sorted(idx_all, key=lambda i: (fitness[i], density[i]))
    A_idx = [i for i in ord_all if fitness[i] < np.median(fitness)]
    if len(A_idx) < N_arch:
        for i in ord_all:
            if i not in A_idx:
                A_idx.append(i)
            if len(A_idx) >= N_arch:
                break
    if len(A_idx) > N_arch:
        A_idx = truncar_por_aglomeracao(A_idx, np.array(Objs_all), N_arch)

    A = [P_all[i].copy() for i in A_idx]
    A_objs = [Objs_all[i].copy() for i in A_idx]
    A_ext  = [Ext_all[i].copy() for i in A_idx]

    f1_vals = [f[0] for f in A_objs]
    f2_vals = [f[1] for f in A_objs]
    print(f"[Geração {gen:02d}] Arquivo externo: {len(A)} soluções | "
          f"f1(min,med,max)=({np.min(f1_vals):.4f}, {np.median(f1_vals):.4f}, {np.max(f1_vals):.4f}) | "
          f"Rend(máx)≈{(-np.min(f2_vals)):.2f}%")

    fit_A, dens_A = calcular_fitness_SPEA2(A_objs)
    pais_idx = selecao_torneio(list(range(len(A))), fit_A, k=2, size=N)

    filhos = []
    for i in range(0, N, 2):
        p1 = A[pais_idx[i % len(pais_idx)]]
        p2 = A[pais_idx[(i+1) % len(pais_idx)]]
        c1, c2 = cruzamento_aritmetico(p1, p2, pc=pc)
        c1 = mutacao_gauss(c1, pm=pm, sigma_frac=0.05)
        c2 = mutacao_gauss(c2, pm=pm, sigma_frac=0.05)
        filhos.extend([c1, c2])
    P = filhos[:N]

    objs, extras = [], []
    for x in P:
        f, m = avaliar_objetivos(x)
        objs.append(f); extras.append(m)

print("\n=== SPEA2 concluído ===")
print(f"Soluções no arquivo externo (Pareto aproximado): {len(A)}")

# =========================================================
# 6) PLOT Fronteira de Pareto (erro de Kappa vs Rendimento)
# =========================================================
A_objs_arr = np.array(A_objs)
fig_pareto = px.scatter(
    x=A_objs_arr[:,0],
    y=-A_objs_arr[:,1],  # -f2 = rendimento
    title="Fronteira de Pareto (aprox.): erro de Kappa vs Rendimento",
    labels={"x": "(Kappa - Kappa_SP)^2  (menor = melhor)", "y": "Rendimento (%)  (maior = melhor)"}
)
fig_pareto.show()

# =========================================================
# 7) LISTAR TOP 15 SOLUÇÕES
# =========================================================
df_solucoes = pd.DataFrame([
    {
        "Temp_Bulk_C": x[0],
        "Carga_Alcalina_madeira": x[1],
        "Sulfidez": x[2],
        "Tempo_Coccao_min": x[3],
        "Kappa": m["Kappa"],
        "Rendimento": m["Rendimento"],
        "Erro_Kappa2": f[0],
        "Score_Proxy": -(f[0]) + (-f[1])  # ranking simples
    }
    for x, f, m in zip(A, A_objs, A_ext)
])

df_top15 = df_solucoes.sort_values(
    by=["Erro_Kappa2", "Rendimento"], ascending=[True, False]
).head(15)

print("\n=== Top 15 soluções encontradas ===")
print(df_top15.to_string(index=False, float_format="%.3f"))
