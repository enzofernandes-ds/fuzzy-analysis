"""
Segmentação Fuzzy de Clientes Atacadistas - versão final (Streamlit + Plotly)
Base: Wholesale Customers (UCI) | Algoritmo: Fuzzy C-Means implementado em NumPy

Como executar (na pasta onde estão este arquivo e o CSV):
    pip install streamlit plotly numpy pandas scikit-learn
    streamlit run fuzzy_wholesale_app.py

Os resultados (índices de validação e resumo dos grupos) também são impressos no terminal
sempre que o modelo é recalculado. Todos os gráficos são gerados a partir dos dados lidos.
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:  # garante acentos corretos no terminal do Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------------------------
ARQUIVO_PADRAO = "Wholesale_customers_data.csv"
SEMENTE = 42
COLUNAS_GASTO = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
NOMES_PT = {"Fresh": "Frescos", "Milk": "Laticínios", "Grocery": "Mercearia",
            "Frozen": "Congelados", "Detergents_Paper": "Limpeza/Papel", "Delicassen": "Delicatessen"}
CHANNEL_NOMES = {1: "Horeca", 2: "Varejo"}
REGION_NOMES = {1: "Lisboa", 2: "Porto", 3: "Outra"}
CORES = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#ec4899"]

st.set_page_config(page_title="Fuzzy C-Means | Wholesale Customers", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------------------------
# Fuzzy C-Means e índices de validação
# ----------------------------------------------------------------------------------------------
def fcm(X, c, m=2.0, max_iter=300, tol=1e-6, seed=0):
    """Retorna centros (c x d), matriz de pertinência U (n x c) e nº de iterações."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n, c))
    U /= U.sum(axis=1, keepdims=True)
    for it in range(1, max_iter + 1):
        Um = U ** m
        centros = (Um.T @ X) / Um.sum(axis=0)[:, None]
        d = np.linalg.norm(X[:, None, :] - centros[None, :, :], axis=2)
        d = np.fmax(d, 1e-12)
        inv = d ** (-2.0 / (m - 1.0))
        U_novo = inv / inv.sum(axis=1, keepdims=True)
        if np.linalg.norm(U_novo - U) < tol:
            U = U_novo
            break
        U = U_novo
    return centros, U, it


def melhor_execucao(X, c, m=2.0, n_init=10):
    """Várias inicializações; mantém a de menor função objetivo J."""
    melhor = None
    for s in range(n_init):
        centros, U, it = fcm(X, c, m, seed=SEMENTE + s)
        d = np.linalg.norm(X[:, None, :] - centros[None, :, :], axis=2)
        J = np.sum((U ** m) * d ** 2)
        if melhor is None or J < melhor[0]:
            melhor = (J, centros, U, it)
    return melhor


def coeficiente_particao(U):
    return np.sum(U ** 2) / U.shape[0]


def entropia_particao(U):
    return -np.sum(U * np.log(np.fmax(U, 1e-12))) / U.shape[0]


def xie_beni(X, centros, U, m=2.0):
    d2 = np.linalg.norm(X[:, None, :] - centros[None, :, :], axis=2) ** 2
    dist_centros = np.linalg.norm(centros[:, None, :] - centros[None, :, :], axis=2) ** 2
    sep = np.fmax(np.min(dist_centros[np.triu_indices(len(centros), k=1)]), 1e-12)
    return np.sum((U ** m) * d2) / (X.shape[0] * sep)


def indice_silhueta(X, rotulos):
    if len(np.unique(rotulos)) < 2:
        return float("nan")
    return float(silhouette_score(X, rotulos))


@st.cache_data(show_spinner="Calculando índices de validação (c = 2 a 8)...")
def calcular_validacao(X, m):
    linhas = []
    for c in range(2, 9):
        J, centros, U, _ = melhor_execucao(X, c, m, n_init=10)
        linhas.append({"c": c, "J": J, "PC": coeficiente_particao(U), "PE": entropia_particao(U),
                       "XB": xie_beni(X, centros, U, m), "Silhueta": indice_silhueta(X, U.argmax(axis=1))})
    return pd.DataFrame(linhas).set_index("c")


@st.cache_data(show_spinner="Executando Fuzzy C-Means...")
def rodar_modelo(X, media, desvio, c, m):
    J, centros, U, it = melhor_execucao(X, c, m, n_init=30)
    centros_orig = np.expm1(centros * desvio + media)
    ordem = np.argsort(centros_orig.sum(axis=1))  # grupos ordenados do menor ao maior gasto total
    return J, centros[ordem], centros_orig[ordem], U[:, ordem], it


def escolher_c(validacao):
    """Maior silhueta; se c=2 vencer mas c=3 ficar a menos de 10% de distância, adota c=3."""
    sil = validacao["Silhueta"].dropna()
    if sil.empty:
        return 2
    c = int(sil.idxmax())
    if c == 2 and 3 in sil.index and sil.loc[3] >= 0.9 * sil.loc[2]:
        c = 3
    return c


# ----------------------------------------------------------------------------------------------
# Funções de gráficos (Plotly)
# ----------------------------------------------------------------------------------------------
def mostrar(fig, chave, altura=420):
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=50, b=10))
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch", key=chave)
    else:
        st.plotly_chart(fig, use_container_width=True, key=chave)


def fig_validacao(validacao, c_sel):
    titulos = ("Coef. de Partição (↑)", "Entropia de Partição (↓)", "Xie-Beni (↓)", "Silhueta (↑)")
    fig = make_subplots(rows=1, cols=4, subplot_titles=titulos)
    for i, col in enumerate(["PC", "PE", "XB", "Silhueta"], start=1):
        fig.add_trace(go.Scatter(x=validacao.index, y=validacao[col], mode="lines+markers", showlegend=False,
                                 line=dict(color="#2563eb"),
                                 hovertemplate="c = %{x}<br>" + col + " = %{y:.3f}<extra></extra>"), row=1, col=i)
        fig.add_trace(go.Scatter(x=[c_sel], y=[validacao.loc[c_sel, col]], mode="markers", showlegend=False,
                                 marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="#ef4444", width=3)),
                                 hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(title_text="Nº de grupos (c)", dtick=1, row=1, col=i)
    fig.update_yaxes(type="log", row=1, col=3)  # XB explode quando dois centros quase coincidem
    return fig


def fig_dispersao(Z, Zc, U, rotulos, nomes, var_exp, modo_3d, colorir):
    n, c = U.shape
    dados = np.column_stack([np.arange(1, n + 1), U])
    hover = "Cliente %{customdata[0]:.0f}<br>" + "<br>".join(
        f"{nomes[k]}: %{{customdata[{k + 1}]:.1%}}" for k in range(c)) + "<extra></extra>"
    fig = go.Figure()
    if colorir == "Grupo dominante":
        for k in range(c):
            idx = rotulos == k
            if not idx.any():
                continue
            alfa = 0.25 + 0.75 * (U[idx, k] - U[idx, k].min()) / (np.ptp(U[idx, k]) + 1e-12)
            if modo_3d:
                fig.add_trace(go.Scatter3d(x=Z[idx, 0], y=Z[idx, 1], z=Z[idx, 2], mode="markers",
                                           name=f"{nomes[k]} (n={idx.sum()})", customdata=dados[idx],
                                           hovertemplate=hover,
                                           marker=dict(size=4, color=CORES[k], opacity=0.85)))
            else:
                fig.add_trace(go.Scatter(x=Z[idx, 0], y=Z[idx, 1], mode="markers",
                                         name=f"{nomes[k]} (n={idx.sum()})", customdata=dados[idx],
                                         hovertemplate=hover,
                                         marker=dict(size=8, color=CORES[k], opacity=alfa)))
    else:
        k = nomes.index(colorir.replace("Pertinência a ", ""))
        marcador = dict(size=4 if modo_3d else 8, color=U[:, k], colorscale="Viridis", cmin=0, cmax=1,
                        colorbar=dict(title=f"Pertinência<br>{nomes[k]}"))
        if modo_3d:
            fig.add_trace(go.Scatter3d(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], mode="markers", name="Clientes",
                                       customdata=dados, hovertemplate=hover, marker=marcador))
        else:
            fig.add_trace(go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers", name="Clientes",
                                     customdata=dados, hovertemplate=hover, marker=marcador))
    for k in range(c):  # centros dos grupos
        if modo_3d:
            fig.add_trace(go.Scatter3d(x=[Zc[k, 0]], y=[Zc[k, 1]], z=[Zc[k, 2]], mode="markers",
                                       name=f"Centro {nomes[k]}", showlegend=False,
                                       marker=dict(size=9, color="black", symbol="diamond")))
        else:
            fig.add_trace(go.Scatter(x=[Zc[k, 0]], y=[Zc[k, 1]], mode="markers", name=f"Centro {nomes[k]}",
                                     showlegend=False,
                                     marker=dict(size=16, color="black", symbol="x", line=dict(width=2))))
    rot = [f"Componente {i + 1} ({v:.0%} da variância)" for i, v in enumerate(var_exp)]
    if modo_3d:
        fig.update_layout(scene=dict(xaxis_title=rot[0], yaxis_title=rot[1], zaxis_title=rot[2]),
                          legend=dict(orientation="h", y=-0.05))
    else:
        fig.update_layout(xaxis_title=rot[0], yaxis_title=rot[1], legend=dict(orientation="h", y=1.08))
    return fig


def fig_perfil(centros_orig, colunas, nomes, escala_log):
    fig = go.Figure()
    for k, nome in enumerate(nomes):
        fig.add_trace(go.Bar(x=[NOMES_PT[c] for c in colunas], y=centros_orig[k], name=nome,
                             marker_color=CORES[k], hovertemplate="%{x}: %{y:,.0f}<extra>" + nome + "</extra>"))
    fig.update_layout(barmode="group", yaxis_title="Gasto anual típico (unidades monetárias)",
                      legend=dict(orientation="h", y=1.1))
    if escala_log:
        fig.update_yaxes(type="log")
    return fig


def fig_heatmap(U, nomes, n_mostrar):
    amb = np.argsort(U.max(axis=1))[:n_mostrar]
    amb = amb[np.argsort(U.argmax(axis=1)[amb], kind="stable")]
    fig = go.Figure(go.Heatmap(z=U[amb], x=nomes, y=[f"Cliente {i + 1}" for i in amb], zmin=0, zmax=1,
                               colorscale="YlGnBu", colorbar=dict(title="Pertinência"),
                               hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", type="category")
    return fig


def fig_histograma(U, limiar):
    fig = go.Figure(go.Histogram(x=U.max(axis=1), nbinsx=25, marker_color="#2563eb",
                                 hovertemplate="Pertinência máxima: %{x}<br>Clientes: %{y}<extra></extra>"))
    fig.add_vline(x=limiar, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Limiar de fronteira ({limiar:.0%})", annotation_position="top right")
    fig.update_layout(xaxis_title="Maior grau de pertinência do cliente", yaxis_title="Nº de clientes")
    return fig


def fig_funcoes(z1, U, nomes):
    ordem = np.argsort(z1)
    fig = go.Figure()
    for k, nome in enumerate(nomes):
        fig.add_trace(go.Scatter(x=z1[ordem], y=U[ordem, k], mode="lines", name=nome, line=dict(color=CORES[k], width=2),
                                 hovertemplate="Componente 1: %{x:.2f}<br>Pertinência: %{y:.1%}<extra>" + nome + "</extra>"))
    fig.update_layout(xaxis_title="Componente 1 (PCA)", yaxis_title="Grau de pertinência",
                      yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1))
    return fig


def fig_cliente(u_cliente, nomes):
    fig = go.Figure(go.Bar(x=nomes, y=u_cliente, marker_color=CORES[:len(nomes)],
                           text=[f"{v:.1%}" for v in u_cliente], textposition="outside",
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_layout(yaxis=dict(range=[0, 1.1], tickformat=".0%", title="Grau de pertinência"))
    return fig


def fig_composicao(ct, titulo):
    fig = go.Figure([go.Bar(name=str(col), x=ct.index, y=ct[col],
                            hovertemplate="%{x}<br>" + str(col) + ": %{y:.1f}%<extra></extra>") for col in ct.columns])
    fig.update_layout(barmode="stack", title=titulo, yaxis_title="% do grupo", legend=dict(orientation="h", y=-0.2))
    return fig


# ----------------------------------------------------------------------------------------------
# Interface: barra lateral e carga dos dados
# ----------------------------------------------------------------------------------------------
st.title("Segmentação Fuzzy de Clientes Atacadistas")
st.caption("Fuzzy C-Means · base Wholesale Customers (UCI) · aprendizado não supervisionado: "
           "`Channel` e `Region` ficam fora do modelo e só aparecem na interpretação final.")

with st.sidebar:
    st.header("Dados")
    arquivo = st.file_uploader("CSV da base (opcional)", type="csv",
                               help=f"Se nada for enviado, o app procura '{ARQUIVO_PADRAO}' na pasta do programa.")

df = None
if arquivo is not None:
    df = pd.read_csv(arquivo)
else:
    for pasta in (Path(__file__).resolve().parent, Path.cwd()):
        if (pasta / ARQUIVO_PADRAO).exists():
            df = pd.read_csv(pasta / ARQUIVO_PADRAO)
            break
if df is None:
    st.error(f"Arquivo '{ARQUIVO_PADRAO}' não encontrado. Coloque-o na mesma pasta deste programa "
             "ou envie o CSV pela barra lateral.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
faltando = [c for c in COLUNAS_GASTO if c not in df.columns]
if faltando:
    st.error(f"Colunas ausentes no CSV: {', '.join(faltando)}. Use o arquivo original da base "
             "(colunas em inglês: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen).")
    st.stop()

with st.sidebar:
    st.header("Parâmetros")
    sel = st.multiselect("Variáveis de gasto usadas no modelo", COLUNAS_GASTO, default=COLUNAS_GASTO,
                         format_func=lambda c: NOMES_PT[c])
    m = st.slider("Fuzzificador (m)", 1.1, 3.0, 2.0, 0.1,
                  help="m próximo de 1 deixa os grupos quase nítidos; valores maiores aumentam a sobreposição.")
    auto = st.checkbox("Escolher nº de grupos automaticamente", value=True,
                       help="Usa o índice de silhueta (ver aba Visão geral).")
    c_manual = st.slider("Nº de grupos (c)", 2, 8, 2, disabled=auto)
    limiar = st.slider("Limiar de fronteira", 0.50, 0.90, 0.60, 0.05,
                       help="Clientes cuja maior pertinência fica abaixo deste valor são considerados ambíguos.")

colunas = [c for c in COLUNAS_GASTO if c in sel]
if len(colunas) < 2:
    st.warning("Selecione pelo menos 2 variáveis de gasto na barra lateral.")
    st.stop()

# ----------------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------------
X_log = np.log1p(df[colunas].astype(float).values)  # log reduz o peso dos clientes gigantes
media, desvio = X_log.mean(axis=0), X_log.std(axis=0)
X = (X_log - media) / desvio

validacao = calcular_validacao(X, m)
c_auto = escolher_c(validacao)
N_GRUPOS = c_auto if auto else c_manual
J, centros, centros_orig, U, it = rodar_modelo(X, media, desvio, N_GRUPOS, m)
rotulos = U.argmax(axis=1)
nomes_grupo = [f"Grupo {i + 1}" for i in range(N_GRUPOS)]
max_pert = U.max(axis=1)
fronteira = max_pert < limiar

sil = indice_silhueta(X, rotulos)
pc, pe, xb = coeficiente_particao(U), entropia_particao(U), xie_beni(X, centros, U, m)

n_comp = min(3, len(colunas))
pca = PCA(n_components=n_comp, random_state=SEMENTE).fit(X)
Z, Zc = pca.transform(X), pca.transform(centros)
var_exp = pca.explained_variance_ratio_

tabela = pd.DataFrame(U, columns=[f"Pertinência {g}" for g in nomes_grupo]).round(4)
tabela.insert(0, "Cliente", np.arange(1, len(df) + 1))
tabela["Grupo dominante"] = [nomes_grupo[i] for i in rotulos]
tabela["Maior pertinência"] = max_pert.round(4)
tabela["Fronteira (ambíguo)"] = np.where(fronteira, "Sim", "Não")
extras = [c for c in ("Channel", "Region") if c in df.columns]
tabela = pd.concat([tabela, df[extras + colunas].reset_index(drop=True)], axis=1)

composicoes = []
for col, nomes_cat, titulo in (("Channel", CHANNEL_NOMES, "Channel (1=Horeca, 2=Varejo)"),
                               ("Region", REGION_NOMES, "Region (1=Lisboa, 2=Porto, 3=Outra)")):
    if col in df.columns:
        ct = pd.crosstab(tabela["Grupo dominante"], df[col].values, normalize="index") * 100
        ct.columns = [nomes_cat.get(x, str(x)) for x in ct.columns]
        composicoes.append((col, titulo, ct.round(1)))

# ----------------------------------------------------------------------------------------------
# Resumo em texto (mesmo conteúdo do protótipo) e impressão no terminal
# ----------------------------------------------------------------------------------------------
linhas = [f"FUZZY C-MEANS | base Wholesale Customers | {len(df)} clientes | c={N_GRUPOS} | m={m} | "
          f"J={J:.2f} | iteracoes={it}",
          f"Silhueta (hard): {sil:.3f} | PC: {pc:.3f} | PE: {pe:.3f} | XB: {xb:.3f}",
          f"Clientes em fronteira (maior pertinencia < {limiar:.0%}): {fronteira.sum()} ({fronteira.mean():.1%})",
          "\nCentros (gasto anual tipico, escala original):",
          pd.DataFrame(centros_orig, columns=[NOMES_PT[c] for c in colunas], index=nomes_grupo)
          .round(0).astype(int).to_string(),
          "\nTamanho dos grupos (grupo dominante):",
          tabela["Grupo dominante"].value_counts().sort_index().to_string()]
for col, titulo, ct in composicoes:
    linhas += [f"\nComposicao por {titulo}, em % do grupo:", ct.to_string()]
resumo = "\n".join(linhas)

chave_print = (tuple(colunas), N_GRUPOS, m, limiar)
if st.session_state.get("_chave_print") != chave_print:  # imprime só quando o modelo muda
    st.session_state["_chave_print"] = chave_print
    print("\nIndices de validacao:\n", validacao.round(3), flush=True)
    print("\n" + resumo, flush=True)

# ----------------------------------------------------------------------------------------------
# Abas
# ----------------------------------------------------------------------------------------------
aba1, aba2, aba3, aba4, aba5 = st.tabs(["Visão geral", "Grupos", "Pertinência", "Tabela", "Interpretação"])

with aba1:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clientes", len(df))
    k2.metric("Grupos (c)", N_GRUPOS)
    k3.metric("Silhueta", "n/d" if np.isnan(sil) else f"{sil:.3f}")
    k4.metric("Coef. de partição", f"{pc:.3f}")
    k5.metric("Clientes em fronteira", f"{fronteira.sum()} ({fronteira.mean():.0%})")
    st.subheader("Escolha do número de grupos")
    st.caption("(↑) maior é melhor · (↓) menor é melhor · círculo vermelho = valor de c em uso. "
               f"Sugestão automática: c = {c_auto}.")
    mostrar(fig_validacao(validacao, N_GRUPOS), "validacao", 360)
    st.dataframe(validacao.round(4))
    with st.expander("Estatísticas descritivas dos gastos (escala original)"):
        st.dataframe(df[colunas].rename(columns=NOMES_PT).describe().T.round(1))

with aba2:
    c1, c2 = st.columns([1, 2])
    modo = c1.radio("Visualização", ["2D", "3D"] if n_comp >= 3 else ["2D"], horizontal=True)
    opcoes_cor = ["Grupo dominante"] + [f"Pertinência a {g}" for g in nomes_grupo]
    colorir = c2.selectbox("Colorir por", opcoes_cor)
    st.caption("Cada ponto é um cliente projetado por PCA. No modo 'Grupo dominante', quanto mais transparente o ponto, "
               "menor a pertinência ao seu grupo. Passe o mouse para ver as pertinências do cliente.")
    mostrar(fig_dispersao(Z, Zc, U, rotulos, nomes_grupo, var_exp, modo == "3D", colorir), "dispersao", 560)
    st.subheader("Perfil de consumo de cada grupo")
    log_y = st.checkbox("Escala logarítmica no eixo Y", value=False)
    mostrar(fig_perfil(centros_orig, colunas, nomes_grupo, log_y), "perfil", 420)
    st.dataframe(pd.DataFrame(centros_orig, columns=[NOMES_PT[c] for c in colunas], index=nomes_grupo).round(0).astype(int))

with aba3:
    a, b = st.columns(2)
    with a:
        st.subheader("Nitidez dos grupos")
        mostrar(fig_histograma(U, limiar), "histograma", 380)
    with b:
        st.subheader("Funções de pertinência")
        mostrar(fig_funcoes(Z[:, 0], U, nomes_grupo), "funcoes", 380)
    st.subheader("Clientes mais ambíguos")
    n_amb = st.slider("Quantidade de clientes no mapa de calor", 10, 100, 40, 5)
    mostrar(fig_heatmap(U, nomes_grupo, n_amb), "heatmap", max(380, 14 * n_amb + 120))
    st.subheader("Consultar um cliente")
    cli = st.number_input("Nº do cliente", min_value=1, max_value=len(df), value=1, step=1)
    ca, cb = st.columns([2, 3])
    with ca:
        mostrar(fig_cliente(U[int(cli) - 1], nomes_grupo), "cliente", 340)
    with cb:
        st.dataframe(df.loc[[int(cli) - 1], extras + colunas].rename(columns=NOMES_PT), hide_index=True)
        st.write(f"Grupo dominante: **{nomes_grupo[rotulos[int(cli) - 1]]}** "
                 f"({max_pert[int(cli) - 1]:.1%}) · "
                 f"{'cliente em fronteira' if fronteira[int(cli) - 1] else 'cliente bem definido'}")

with aba4:
    f1, f2 = st.columns([3, 2])
    grupos_sel = f1.multiselect("Grupo dominante", nomes_grupo, default=nomes_grupo)
    so_fronteira = f2.checkbox("Mostrar apenas clientes em fronteira")
    vis = tabela[tabela["Grupo dominante"].isin(grupos_sel)]
    if so_fronteira:
        vis = vis[vis["Fronteira (ambíguo)"] == "Sim"]
    st.caption(f"{len(vis)} clientes exibidos")
    st.dataframe(vis, hide_index=True)
    st.download_button("Baixar tabela de pertinência (CSV)", tabela.to_csv(index=False).encode("utf-8-sig"),
                       file_name="tabela_pertinencia.csv", mime="text/csv")

with aba5:
    st.subheader("Leitura dos grupos")
    ref = np.expm1(media)  # padrão (centro geométrico) da base, mesma escala dos centros
    for k in range(N_GRUPOS):
        razao = centros_orig[k] / ref
        acima = [NOMES_PT[c] for c, r in zip(colunas, razao) if r >= 1.25]
        abaixo = [NOMES_PT[c] for c, r in zip(colunas, razao) if r <= 0.80]
        partes = []
        if acima:
            partes.append("gasto acima do padrão da base em " + ", ".join(acima))
        if abaixo:
            partes.append("abaixo do padrão em " + ", ".join(abaixo))
        if not partes:
            partes.append("consumo próximo ao padrão da base em todas as categorias")
        n_k = int((rotulos == k).sum())
        st.markdown(f"- **{nomes_grupo[k]}** ({n_k} clientes): " + "; ".join(partes) + ".")
    if composicoes:
        st.subheader("Validação externa (Channel e Region)")
        st.caption("Essas colunas não entraram no modelo; aqui servem só para checar se os grupos fazem sentido.")
        cols = st.columns(len(composicoes))
        for i, (col, titulo, ct) in enumerate(composicoes):
            with cols[i]:
                mostrar(fig_composicao(ct, titulo), f"comp_{col}", 360)
    st.subheader("Resumo (igual ao impresso no terminal)")
    st.code(resumo, language="text")
