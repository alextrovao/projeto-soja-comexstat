"""
Projeto: previsão das exportações mensais de soja em grão (Comex Stat / MDIC)
Etapa 2: análise exploratória (EDA)

Entrada : dados_tratados/soja_mensal.csv   (gerado pelo script 01)
Saída   : graficos/*.png  +  resumo impresso no terminal e em dados_tratados/resumo_eda.txt
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # salva as figuras em arquivo, sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------------------------- parâmetros
PASTA_BASE = Path(__file__).resolve().parent
ARQUIVO = PASTA_BASE / "dados_tratados" / "soja_mensal.csv"
PASTA_GRAFICOS = PASTA_BASE / "graficos"

# Se o último mês da base estiver incompleto (compare com o mesmo mês do ano
# anterior), troque para True. O script 01 imprime o valor desse mês.
EXCLUIR_ULTIMO_MES = False

NOMES_MES = ["jan", "fev", "mar", "abr", "mai", "jun",
             "jul", "ago", "set", "out", "nov", "dez"]
linhas_resumo: list[str] = []


def registrar(texto: str = "") -> None:
    """Imprime e guarda o texto para o arquivo de resumo."""
    print(texto)
    linhas_resumo.append(texto)


# ------------------------------------------------------------------ carga
def carregar() -> pd.DataFrame:
    df = pd.read_csv(ARQUIVO, parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)
    if EXCLUIR_ULTIMO_MES:
        df = df.iloc[:-1].copy()
    df["valor_mi"] = df["vl_fob_usd"] / 1e6                    # US$ milhões
    df["mil_t"] = df["kg_liquido"] / 1e6                       # mil toneladas
    df["preco_t"] = (df["vl_fob_usd"] / (df["kg_liquido"] / 1000)).replace(
        [np.inf, -np.inf], np.nan
    )                                                           # US$ por tonelada
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    return df


# --------------------------------------------------------------- resumos
def resumo_geral(df: pd.DataFrame) -> None:
    registrar("=== RESUMO GERAL ===")
    registrar(f"Período: {df['data'].min():%Y-%m} a {df['data'].max():%Y-%m} "
              f"({len(df)} meses)")
    registrar(f"Valor mensal médio: US$ {df['valor_mi'].mean():,.0f} milhões")
    registrar(f"Valor mensal mínimo: US$ {df['valor_mi'].min():,.0f} milhões "
              f"({df.loc[df['valor_mi'].idxmin(), 'data']:%Y-%m})")
    registrar(f"Valor mensal máximo: US$ {df['valor_mi'].max():,.0f} milhões "
              f"({df.loc[df['valor_mi'].idxmax(), 'data']:%Y-%m})")
    registrar(f"Preço médio implícito: US$ {df['preco_t'].mean():,.0f}/t "
              f"(mín. {df['preco_t'].min():,.0f}; máx. {df['preco_t'].max():,.0f})")
    corr = df["valor_mi"].corr(df["mil_t"])
    registrar(f"Correlação entre valor e volume: {corr:.2f}")
    registrar()


def resumo_anual(df: pd.DataFrame) -> pd.DataFrame:
    """Total anual e comparação justa: mesmo período (jan até o último mês) em todos os anos."""
    ultimo_mes = int(df["mes"].iloc[-1])
    ultimo_ano = int(df["ano"].iloc[-1])

    anual = df.groupby("ano").agg(
        meses=("data", "count"), valor_mi=("valor_mi", "sum"), mil_t=("mil_t", "sum")
    )
    comparavel = (
        df[df["mes"] <= ultimo_mes].groupby("ano")["valor_mi"].sum().rename("jan_ate_ult_mes_mi")
    )
    anual = anual.join(comparavel)
    anual["var_vs_ano_ant_%"] = anual["jan_ate_ult_mes_mi"].pct_change() * 100

    registrar("=== TOTAIS POR ANO (US$ milhões / mil toneladas) ===")
    registrar(anual.round(1).to_string())
    registrar(f"\n'jan_ate_ult_mes_mi' soma janeiro a {NOMES_MES[ultimo_mes - 1]} de cada ano,")
    registrar(f"para comparar {ultimo_ano} com os anos anteriores no mesmo período.")
    registrar()
    return anual


def resumo_sazonal(df: pd.DataFrame) -> pd.Series:
    # Remove o efeito do tamanho do ano: valor do mês / média do ano (só anos completos)
    completos = df.groupby("ano")["data"].transform("count") == 12
    base = df[completos].copy()
    base["indice"] = base["valor_mi"] / base.groupby("ano")["valor_mi"].transform("mean")
    sazonal = base.groupby("mes")["indice"].mean()

    registrar("=== ÍNDICE SAZONAL (1,00 = média do ano; anos completos) ===")
    for m, v in sazonal.items():
        registrar(f"{NOMES_MES[m - 1]}: {v:.2f}")
    top = sazonal.sort_values(ascending=False).head(3).index
    registrar("Meses mais fortes: " + ", ".join(NOMES_MES[m - 1] for m in top))
    registrar()
    return sazonal


# --------------------------------------------------------------- gráficos
def salvar(fig, nome: str) -> None:
    PASTA_GRAFICOS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / nome, dpi=150)
    plt.close(fig)


def grafico_serie(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["data"], df["valor_mi"], color="#9bb7d4", label="Mensal")
    ax.plot(df["data"], df["valor_mi"].rolling(12).mean(), color="#1f4e79",
            linewidth=2.2, label="Média móvel de 12 meses")
    ax.set_title("Exportações de soja em grão do Brasil: valor mensal")
    ax.set_ylabel("US$ milhões (FOB)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(0, -0.18, "Fonte: Comex Stat / MDIC (NCM 1201).", transform=ax.transAxes, fontsize=8)
    salvar(fig, "01_serie_mensal.png")


def grafico_sazonalidade(df: pd.DataFrame) -> None:
    dados = [df.loc[df["mes"] == m, "valor_mi"].values for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.boxplot(dados, tick_labels=NOMES_MES, patch_artist=True,
               boxprops=dict(facecolor="#9bb7d4"), medianprops=dict(color="#1f4e79"))
    ax.set_title("Sazonalidade: distribuição do valor exportado por mês do ano")
    ax.set_ylabel("US$ milhões (FOB)")
    ax.grid(alpha=0.3, axis="y")
    salvar(fig, "02_sazonalidade.png")


def grafico_anual_comparavel(anual: pd.DataFrame, ultimo_mes: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(anual.index.astype(str), anual["jan_ate_ult_mes_mi"], color="#1f4e79")
    ax.set_title(f"Valor exportado de janeiro a {NOMES_MES[ultimo_mes - 1]}, por ano")
    ax.set_ylabel("US$ milhões (FOB)")
    ax.grid(alpha=0.3, axis="y")
    salvar(fig, "03_anual_mesmo_periodo.png")


def grafico_preco(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["data"], df["preco_t"], color="#c0504d")
    ax.set_title("Preço médio implícito da soja exportada")
    ax.set_ylabel("US$ por tonelada")
    ax.grid(alpha=0.3)
    salvar(fig, "04_preco_implicito.png")


def grafico_decomposicao(df: pd.DataFrame) -> None:
    serie = (
        df.set_index("data")["valor_mi"].asfreq("MS").interpolate()
    )  # preenche eventuais lacunas só para decompor
    res = seasonal_decompose(serie, model="multiplicative", period=12)
    fig = res.plot()
    fig.set_size_inches(11, 8)
    salvar(fig, "05_decomposicao.png")


# --------------------------------------------------------------------- main
def main() -> None:
    df = carregar()
    resumo_geral(df)
    anual = resumo_anual(df)
    resumo_sazonal(df)

    ultimo_mes = int(df["mes"].iloc[-1])
    grafico_serie(df)
    grafico_sazonalidade(df)
    grafico_anual_comparavel(anual, ultimo_mes)
    grafico_preco(df)
    grafico_decomposicao(df)

    (PASTA_BASE / "dados_tratados" / "resumo_eda.txt").write_text(
        "\n".join(linhas_resumo), encoding="utf-8"
    )
    print(f"Gráficos salvos em: {PASTA_GRAFICOS}")
    print("Resumo salvo em: dados_tratados/resumo_eda.txt")


if __name__ == "__main__":
    main()