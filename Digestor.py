# pip install pandas openpyxl pandasgui plotly scipy

from pathlib import Path
#from pandasgui import show   # deixe comentado se estiver em ambiente sem GUI
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path
import plotly.io as pio
pio.renderers.default = "notebook_connected"   # funciona bem no VSCode Interactive

# =========================================================
# 1) Leitura do arquivo
# =========================================================
# Se estiver no Jupyter/VSCode Interactive, use o diretório atual
try:
    pasta = Path(__file__).parent
except NameError:
    pasta = Path.cwd()   # Jupyter/Interactive não tem __file__

arquivo = pasta / "digestor_sintetico_batelada_final.xlsx"

print(f"Lendo: {arquivo}")
if not arquivo.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")

print(f"Lendo: {arquivo.name}")

try:
    df = pd.read_excel(arquivo, sheet_name="digestor_sintetico", engine="openpyxl")
except ValueError:
    df = pd.read_excel(arquivo, engine="openpyxl")

print(df.shape)      # (linhas, colunas)
print(df.columns)    # nomes das colunas
print(df.head(3))    # amostra

# Ajustar tipo categórico para batelada (se existir)
if "ID_Batelada" in df.columns:
    df["ID_Batelada"] = df["ID_Batelada"].astype("category")

# =========================================================
# 2) (Opcional) Exploração rápida
# =========================================================
# show(df)  # descomente se quiser abrir a GUI interativa

# Matriz de correlação completa (opcional)
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Matriz de Correlação (Pearson) — Completa",
    color_continuous_scale="RdBu",
    zmin=-1, zmax=1
)
fig_corr.update_layout(margin=dict(l=40, r=40, t=60, b=40))
fig_corr.show()

# Matriz (variáveis mais fortes com Kappa_Number) — opcional
target_col = "Kappa_Number"
if target_col in numeric_df.columns:
    abs_with_target = corr[target_col].abs().sort_values(ascending=False)
    abs_no_target = abs_with_target.drop(index=target_col, errors="ignore")
    limiar = 0.20
    fortes = abs_no_target[abs_no_target >= limiar].index.tolist()
    if len(fortes) == 0:
        fortes = abs_no_target.head(8).index.tolist()
    subset_cols = [target_col] + fortes
    corr_subset = corr.loc[subset_cols, subset_cols]
    fig_corr_strong = px.imshow(
        corr_subset,
        text_auto=True,
        aspect="auto",
        title=f"Matriz de Correlação — Variáveis mais fortes (|corr com {target_col}| ≥ {limiar} ou Top 8)",
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1
    )
    fig_corr_strong.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    fig_corr_strong.show()
else:
    print(f"Atenção: coluna alvo '{target_col}' não encontrada. Pulei o heatmap das variáveis mais fortes.")

# =========================================================
# 3) SIMULAÇÃO BASEADA EM DERIVADAS (EDO)
#    Modelo simplificado da cinética (bulk) para fins didáticos
# =========================================================
def simular_digestor(temp_bulk_c, carga_alcalina, sulfidez, tempo_coccao_min):
    """
    Simula a variação da lignina (L) no tempo (t) via EDO: dL/dt = -k * L
    k depende de temperatura (Temp_Bulk_C), carga alcalina (% madeira) e sulfidez (%).
    Retorna:
      - kappa_sim (aprox do Kappa_Number)
      - L_end (lignina final % relativa ao início)
    """
    # Parâmetros simplificados (didáticos)
    # Conversões e fatores
    T_ref = 155.0  # °C de referência
    a, b = 0.04, 0.03   # pesos de sensibilidade (temperatura e sulfidez) (adimensionais)
    c = 0.20            # peso da carga alcalina (adimensional)
    k0 = 0.015          # base de taxa [1/min], ajustada para dar L_end plausível

    # Fator cinético (arrhenius-like simplificado/empírico)
    # Cresce com Temp, Carga e Sulfidez
    k = k0 * np.exp(a * (temp_bulk_c - T_ref)) * (1.0 + c * (carga_alcalina - 18)/18.0) * (1.0 + b * (sulfidez - 30)/30.0)

    # EDO: dL/dt = -k * L
    def dLdt(t, L):
        return -k * L

    # Condição inicial: L0 = 100 (unidade relativa)
    L0 = [100.0]
    t_span = (0.0, float(tempo_coccao_min))
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(dLdt, t_span, L0, t_eval=t_eval, method="RK45")
    L_end = sol.y[0, -1]

    # Mapear L_end -> Kappa (aproximação linear L/θ)
    theta = 3.3  # fator de escala para dar kappa ~ 18 em condições típicas
    kappa_sim = np.clip(L_end / theta, 10, 30)

    return float(kappa_sim), float(L_end)

def rendimento_estimado(kappa, relacao_licor_madeira=5.5):
    """
    Proxy de rendimento: cresce quando kappa é maior (menos deslignificação),
    e reduz levemente com relação licor/madeira mais alta.
    """
    return np.clip(48 + 0.30 * (kappa - 18) - 0.05 * (relacao_licor_madeira - 5.5), 40, 55)

# =========================================================
# 4) LOOP DE OTIMIZAÇÃO (random search didático)
#    Objetivo: minimizar (kappa - kappa_sp)^2 e maximizar rendimento
# =========================================================
# Definir bounds a partir do DataFrame
def bounds(col):
    return float(numeric_df[col].min()), float(numeric_df[col].max())

# Variáveis de decisão (usamos as principais do cozimento)
vars_decisao = ["Temp_Bulk_C", "Carga_Alcalina_madeira", "Sulfidez", "Tempo_Coccao_min"]
for v in vars_decisao:
    if v not in numeric_df.columns:
        raise ValueError(f"Coluna necessária '{v}' não encontrada no arquivo.")

bounds_dict = {v: bounds(v) for v in vars_decisao}

# Setpoint de Kappa
kappa_sp = float(numeric_df["Kappa_Number"].median()) if "Kappa_Number" in numeric_df.columns else 18.0

# Peso dos objetivos (ajuste conforme preferência: qualidade vs rendimento)
alpha = 1.0   # peso do erro de kappa
beta  = 0.5   # peso do rendimento (com sinal para maximizar)

# Relação licor/madeira média para o proxy de rendimento (se existir no DF, usar mediana)
rel_licor_col = "Relacao_Licor_Madeira_m3_BDt"
rel_licor_med = float(numeric_df[rel_licor_col].median()) if rel_licor_col in numeric_df.columns else 5.5

# Random search
N_ITER = 30
melhor_score = None
melhor_sol = None

print("\n=== Iniciando otimização (random search) ===")
print(f"Setpoint de Kappa: {kappa_sp:.2f}")
print(f"Variáveis e limites:")
for v, (lo, hi) in bounds_dict.items():
    print(f"  - {v}: [{lo:.3f}, {hi:.3f}]")

for it in range(1, N_ITER + 1):
    # Amostrar candidato dentro dos limites (uniforme)
    cand = {v: np.random.uniform(bounds_dict[v][0], bounds_dict[v][1]) for v in vars_decisao}

    # Simular digestor
    kappa_sim, L_end = simular_digestor(
        temp_bulk_c=cand["Temp_Bulk_C"],
        carga_alcalina=cand["Carga_Alcalina_madeira"],
        sulfidez=cand["Sulfidez"],
        tempo_coccao_min=cand["Tempo_Coccao_min"]
    )

    # Calcular rendimento estimado
    rend_sim = rendimento_estimado(kappa_sim, relacao_licor_madeira=rel_licor_med)

    # Função objetivo: minimizar erro de kappa e maximizar rendimento
    # Score = -(alpha*(erro_kappa)^2) + beta*(rend)   (maximização implícita via score)
    erro = (kappa_sim - kappa_sp)
    score = -(alpha * erro**2) + beta * (rend_sim)

    # Logging do passo
    print(
        f"[Iter {it:02d}] "
        f"Tbulk={cand['Temp_Bulk_C']:.2f} °C | "
        f"Carga={cand['Carga_Alcalina_madeira']:.2f} | "
        f"Sulfidez={cand['Sulfidez']:.2f} | "
        f"Tcoc={cand['Tempo_Coccao_min']:.1f} min || "
        f"Kappa={kappa_sim:.2f} (erro={erro:+.2f}) | "
        f"Rend={rend_sim:.2f}% | "
        f"Score={score:.3f}"
    )

    # Atualizar melhor solução
    if (melhor_score is None) or (score > melhor_score):
        melhor_score = score
        melhor_sol = (cand, kappa_sim, rend_sim, L_end)

# Resultado final
print("\n=== Resultado da Otimização ===")
if melhor_sol is not None:
    cand, kappa_sim, rend_sim, L_end = melhor_sol
    print("Melhor conjunto de variáveis de decisão encontrado:")
    print(f"  Temp_Bulk_C            : {cand['Temp_Bulk_C']:.2f} °C")
    print(f"  Carga_Alcalina_madeira : {cand['Carga_Alcalina_madeira']:.2f}")
    print(f"  Sulfidez               : {cand['Sulfidez']:.2f}")
    print(f"  Tempo_Coccao_min       : {cand['Tempo_Coccao_min']:.1f} min")
    print(f"Resultados simulados:")
    print(f"  Kappa_Number (sim)     : {kappa_sim:.2f}")
    print(f"  Rendimento (sim)       : {rend_sim:.2f}%")
    print(f"  Lignina final L_end    : {L_end:.2f} (unid. relativa)")
    print(f"  Score                  : {melhor_score:.3f}")
else:
    print("Nenhuma solução encontrada (verifique limites/variáveis).")
