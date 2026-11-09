"""
Lógica Fuzzy em base não supervisionada: Wholesale Customers (UCI)
Algoritmo: Fuzzy C-Means (FCM) implementado em NumPy (sem dependência de scikit-fuzzy)

Como usar:
    1) Coloque 'Wholesale_customers_data.csv' na mesma pasta deste arquivo
    2) python fuzzy_wholesale.py
    Saídas (pasta 'resultados_fuzzy'): gráficos .png, tabela de pertinência .csv e resumo .txt
"""

# %% 1. Imports e configuração
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ARQUIVO = "Wholesale_customers_data.csv"
PASTA_SAIDA = "resultados_fuzzy"
M = 2.0              # fuzzificador (m=2 é o valor padrão da literatura)
SEMENTE = 42
LIMIAR_AMBIGUO = 0.60  # cliente cuja maior pertinência é menor que isso = "fronteira"
os.makedirs(PASTA_SAIDA, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

COLUNAS_GASTO = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
NOMES_PT = {"Fresh": "Frescos", "Milk": "Laticínios", "Grocery": "Mercearia",
            "Frozen": "Congelados", "Detergents_Paper": "Limpeza/Papel", "Delicassen": "Delicatessen"}


# %% 2. Implementação do Fuzzy C-Means
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
    """Roda o FCM com várias inicializações e mantém a de menor função objetivo J."""
    melhor = None
    for s in range(n_init):
        centros, U, it = fcm(X, c, m, seed=SEMENTE + s)
        d = np.linalg.norm(X[:, None, :] - centros[None, :, :], axis=2)
        J = np.sum((U ** m) * d ** 2)
        if melhor is None or J < melhor[0]:
            melhor = (J, centros, U, it)
    return melhor


# %% 3. Índices de validação (escolha do número de grupos)
def coeficiente_particao(U):
    return np.sum(U ** 2) / U.shape[0]          # PC: maior = melhor (1 = partição nítida)


def entropia_particao(U):
    return -np.sum(U * np.log(np.fmax(U, 1e-12))) / U.shape[0]  # PE: menor = melhor


def xie_beni(X, centros, U, m=2.0):
    d2 = np.linalg.norm(X[:, None, :] - centros[None, :, :], axis=2) ** 2
    num = np.sum((U ** m) * d2)
    dist_centros = np.linalg.norm(centros[:, None, :] - centros[None, :, :], axis=2) ** 2
    sep = np.min(dist_centros[np.triu_indices(len(centros), k=1)])
    return num / (X.shape[0] * sep)             # XB: menor = melhor


# %% 4. Carga e preparação dos dados
df = pd.read_csv(ARQUIVO)
print(f"Base carregada: {df.shape[0]} clientes x {df.shape[1]} colunas | valores nulos: {int(df.isna().sum().sum())}")

# 'Channel' e 'Region' são categorias (rótulos): ficam FORA do modelo para manter o caráter não supervisionado.
# Só serão usadas no final, para interpretar (validação externa) os grupos encontrados.
X_bruto = df[COLUNAS_GASTO].astype(float)

# Gastos são muito assimétricos (poucos clientes enormes): log1p + padronização evitam que outliers dominem.
X_log = np.log1p(X_bruto.values)
escala = StandardScaler().fit(X_log)
X = escala.transform(X_log)

# Estatísticas descritivas
X_bruto.rename(columns=NOMES_PT).describe().T.round(1).to_csv(f"{PASTA_SAIDA}/estatisticas_descritivas.csv", encoding="utf-8-sig")

# %% 5. Escolha do número de grupos (c)
faixa_c = range(2, 9)
linhas = []
for c in faixa_c:
    J, centros, U, it = melhor_execucao(X, c, M)
    rotulos = U.argmax(axis=1)
    linhas.append({"c": c, "J": J, "PC": coeficiente_particao(U), "PE": entropia_particao(U),
                   "XB": xie_beni(X, centros, U, M), "Silhueta": silhouette_score(X, rotulos)})
validacao = pd.DataFrame(linhas).set_index("c")
validacao.round(4).to_csv(f"{PASTA_SAIDA}/indices_validacao.csv", encoding="utf-8-sig")
print("\nÍndices de validação:\n", validacao.round(3))

fig, eixos = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (col, titulo, dica) in zip(eixos, [("PC", "Coef. de Partição", "↑ maior é melhor"),
                                           ("PE", "Entropia de Partição", "↓ menor é melhor"),
                                           ("XB", "Xie-Beni", "↓ menor é melhor"),
                                           ("Silhueta", "Silhueta", "↑ maior é melhor")]):
    ax.plot(validacao.index, validacao[col], marker="o", color="#2563eb")
    ax.set_title(f"{titulo}\n({dica})", fontsize=10)
    ax.set_xlabel("Nº de grupos (c)")
    if col == "XB":
        ax.set_yscale("log")  # XB explode quando dois centros quase coincidem (c alto)
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/01_indices_validacao.png")
plt.close()

# Escolha de c: maior silhueta. Se c=2 vencer mas c=3 ficar a menos de 10% de distância, adota-se c=3 (mais informativo).
# Ajuste N_GRUPOS manualmente para testar outro valor.
N_GRUPOS = int(validacao["Silhueta"].idxmax())
if N_GRUPOS == 2 and validacao.loc[3, "Silhueta"] >= 0.9 * validacao.loc[2, "Silhueta"]:
    N_GRUPOS = 3
print(f"\nNúmero de grupos adotado: c = {N_GRUPOS}")

# %% 6. Modelo final
J, centros, U, it = melhor_execucao(X, N_GRUPOS, M, n_init=30)
print(f"FCM final: c={N_GRUPOS}, m={M}, J={J:.2f}, convergiu em {it} iterações")

# Reordena os grupos pelo gasto total médio (do menor para o maior) para rótulos estáveis
centros_orig = np.expm1(escala.inverse_transform(centros))
ordem = np.argsort(centros_orig.sum(axis=1))
centros, centros_orig, U = centros[ordem], centros_orig[ordem], U[:, ordem]
rotulos = U.argmax(axis=1)
nomes_grupo = [f"Grupo {i + 1}" for i in range(N_GRUPOS)]

# %% 7. Tabela de pertinência
tabela = pd.DataFrame(U, columns=[f"Pertinência {g}" for g in nomes_grupo]).round(4)
tabela.insert(0, "Cliente", np.arange(1, len(df) + 1))
tabela["Grupo dominante"] = [nomes_grupo[i] for i in rotulos]
tabela["Maior pertinência"] = U.max(axis=1).round(4)
tabela["Fronteira (ambíguo)"] = np.where(U.max(axis=1) < LIMIAR_AMBIGUO, "Sim", "Não")
tabela = pd.concat([tabela, df[["Channel", "Region"] + COLUNAS_GASTO].reset_index(drop=True)], axis=1)
tabela.to_csv(f"{PASTA_SAIDA}/tabela_pertinencia.csv", index=False, encoding="utf-8-sig")

# %% 8. Gráficos
cores = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#ec4899"][:N_GRUPOS]

# 8.1 Dispersão em 2D (PCA): cor = grupo dominante, opacidade = grau de pertinência
pca = PCA(n_components=2, random_state=SEMENTE).fit(X)
Z = pca.transform(X)
Zc = pca.transform(centros)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for k in range(N_GRUPOS):
    idx = rotulos == k
    alfa = 0.25 + 0.75 * (U[idx, k] - U[idx, k].min()) / (np.ptp(U[idx, k]) + 1e-12)
    ax.scatter(Z[idx, 0], Z[idx, 1], s=28, color=cores[k], alpha=alfa, label=f"{nomes_grupo[k]} (n={idx.sum()})")
ax.scatter(Zc[:, 0], Zc[:, 1], marker="X", s=220, c="black", edgecolor="white", linewidth=1.2, label="Centros", zorder=5)
ax.set_xlabel(f"Componente 1 ({pca.explained_variance_ratio_[0]:.0%} da variância)")
ax.set_ylabel(f"Componente 2 ({pca.explained_variance_ratio_[1]:.0%} da variância)")
ax.set_title("Grupos fuzzy dos clientes (opacidade = grau de pertinência)")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/02_dispersao_grupos.png")
plt.close()

# 8.2 Perfil de consumo de cada grupo (gasto médio dos centros, escala original)
fig, ax = plt.subplots(figsize=(9, 4.8))
largura = 0.8 / N_GRUPOS
posx = np.arange(len(COLUNAS_GASTO))
for k in range(N_GRUPOS):
    ax.bar(posx + k * largura, centros_orig[k], width=largura, color=cores[k], label=nomes_grupo[k])
ax.set_xticks(posx + largura * (N_GRUPOS - 1) / 2)
ax.set_xticklabels([NOMES_PT[c] for c in COLUNAS_GASTO])
ax.set_ylabel("Gasto anual típico (unidades monetárias)")
ax.set_title("Perfil de consumo por grupo (centros do FCM)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/03_perfil_grupos.png")
plt.close()

# 8.3 Mapa de calor das pertinências dos 40 clientes mais "ambíguos" (os que o fuzzy mais ajuda a explicar)
amb = np.argsort(U.max(axis=1))[:40]
amb = amb[np.argsort(rotulos[amb], kind="stable")]
fig, ax = plt.subplots(figsize=(6.5, 7.5))
im = ax.imshow(U[amb], aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
ax.set_xticks(range(N_GRUPOS))
ax.set_xticklabels(nomes_grupo)
ax.set_yticks(range(len(amb)))
ax.set_yticklabels([f"Cliente {i + 1}" for i in amb], fontsize=7)
plt.colorbar(im, ax=ax, label="Grau de pertinência")
ax.set_title("Pertinência dos 40 clientes mais ambíguos")
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/04_heatmap_ambiguos.png")
plt.close()

# 8.4 Distribuição da maior pertinência (quão "nítidos" são os grupos)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(U.max(axis=1), bins=25, color="#2563eb", edgecolor="white")
ax.axvline(LIMIAR_AMBIGUO, color="#ef4444", linestyle="--", label=f"Limiar de fronteira ({LIMIAR_AMBIGUO:.0%})")
ax.set_xlabel("Maior grau de pertinência do cliente")
ax.set_ylabel("Nº de clientes")
ax.set_title("Nitidez dos grupos")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/05_distribuicao_pertinencia.png")
plt.close()

# 8.5 Funções de pertinência ao longo do eixo principal (visão clássica da lógica fuzzy)
ordem_x = np.argsort(Z[:, 0])
fig, ax = plt.subplots(figsize=(8, 4.2))
for k in range(N_GRUPOS):
    ax.plot(Z[ordem_x, 0], U[ordem_x, k], color=cores[k], linewidth=1.6, label=nomes_grupo[k])
ax.set_xlabel("Componente 1 (PCA)")
ax.set_ylabel("Grau de pertinência")
ax.set_title("Pertinência de cada grupo ao longo do eixo principal")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/06_funcoes_pertinencia.png")
plt.close()

# %% 9. Interpretação e validação externa (Channel/Region só entram aqui, depois do agrupamento)
resumo = []
resumo.append(f"FUZZY C-MEANS | base Wholesale Customers | {len(df)} clientes | c={N_GRUPOS} | m={M} | J={J:.2f} | iterações={it}")
resumo.append(f"Silhueta (hard): {silhouette_score(X, rotulos):.3f} | PC: {coeficiente_particao(U):.3f} | "
              f"PE: {entropia_particao(U):.3f} | XB: {xie_beni(X, centros, U, M):.3f}")
resumo.append(f"Clientes em fronteira (maior pertinência < {LIMIAR_AMBIGUO:.0%}): {(U.max(axis=1) < LIMIAR_AMBIGUO).sum()} "
              f"({(U.max(axis=1) < LIMIAR_AMBIGUO).mean():.1%})")
resumo.append("\nCentros (gasto anual típico, escala original):")
resumo.append(pd.DataFrame(centros_orig, columns=[NOMES_PT[c] for c in COLUNAS_GASTO], index=nomes_grupo).round(0).astype(int).to_string())
resumo.append("\nTamanho dos grupos (grupo dominante):")
resumo.append(pd.Series([nomes_grupo[i] for i in rotulos]).value_counts().sort_index().to_string())
resumo.append("\nComposição por Channel (1=Horeca, 2=Varejo), em % do grupo:")
resumo.append((pd.crosstab(tabela["Grupo dominante"], df["Channel"], normalize="index") * 100).round(1).to_string())
resumo.append("\nComposição por Region (1=Lisboa, 2=Porto, 3=Outra), em % do grupo:")
resumo.append((pd.crosstab(tabela["Grupo dominante"], df["Region"], normalize="index") * 100).round(1).to_string())
texto = "\n".join(resumo)
print("\n" + texto)
with open(f"{PASTA_SAIDA}/resumo.txt", "w", encoding="utf-8") as f:
    f.write(texto)
print(f"\nArquivos salvos em '{PASTA_SAIDA}/'")
