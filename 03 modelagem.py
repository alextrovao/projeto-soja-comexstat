"""
Projeto: previsão das exportações mensais de soja em grão (Comex Stat / MDIC)
Etapa 3: modelagem e comparação de modelos

Entrada : dados_tratados/soja_mensal.csv   (gerado pelo script 01)
Saída   : graficos/06 a 08 *.png
          dados_tratados/resultados_modelos.csv, previsoes_teste.csv,
          previsao_proximos_12m.csv, resumo_modelagem.txt

Pergunta: quanto cada modelo melhora a previsão de "repetir o mesmo mês do ano anterior"?

Método: cada modelo prevê os 12 meses seguintes a uma data de corte, usando só o
que existia até essa data. Repetimos em 3 janelas de 12 meses, as mais recentes.
Métricas: MAE (erro médio absoluto) e WAPE (soma dos erros absolutos / soma dos
valores reais). O WAPE é preferido ao MAPE porque meses muito baixos, como
janeiro, tornariam o MAPE enganoso.

Modelos opcionais: a suavização exponencial precisa de `statsmodels` e o Random
Forest de `scikit-learn`. Sem a biblioteca, o modelo é pulado e o resto roda.
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ModuleNotFoundError:
    ExponentialSmoothing = None

try:
    from sklearn.ensemble import RandomForestRegressor
except ModuleNotFoundError:
    RandomForestRegressor = None

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- parâmetros
PASTA_BASE = Path(__file__).resolve().parent
ARQUIVO = PASTA_BASE / "dados_tratados" / "soja_mensal.csv"
PASTA_TRATADOS = PASTA_BASE / "dados_tratados"
PASTA_GRAFICOS = PASTA_BASE / "graficos"

ALVO = "valor_mi"          # "valor_mi" (US$ milhões) ou "mil_t" (mil toneladas)
HORIZONTE = 12             # meses previstos de uma vez
N_JANELAS = 3              # janelas de teste, da mais recente para trás
EXCLUIR_ULTIMO_MES = False  # True se o último mês da base estiver incompleto

ROTULO_ALVO = {"valor_mi": "US$ milhões (FOB)", "mil_t": "mil toneladas"}
REFERENCIA = "Referência: mesmo mês do ano anterior"

linhas_resumo: list[str] = []


def registrar(texto: str = "") -> None:
    print(texto)
    linhas_resumo.append(texto)


# ------------------------------------------------------------------ carga
def carregar() -> pd.Series:
    df = pd.read_csv(ARQUIVO, parse_dates=["data"]).sort_values("data")
    df["valor_mi"] = df["vl_fob_usd"] / 1e6
    df["mil_t"] = df["kg_liquido"] / 1e6
    if EXCLUIR_ULTIMO_MES:
        df = df.iloc[:-1]
    serie = df.set_index("data")[ALVO].asfreq("MS")
    if serie.isna().any():
        print(f"Aviso: {int(serie.isna().sum())} mês(es) faltando; interpolando.")
        serie = serie.interpolate()
    return serie


# ----------------------------------------------------------------- modelos
def prever_referencia(treino: pd.Series, datas_prev: pd.DatetimeIndex) -> np.ndarray:
    """Repete o valor do mesmo mês do ano anterior."""
    y = treino.values
    n = len(y)
    return np.array([y[n - 12 + (h - 1)] for h in range(1, len(datas_prev) + 1)])


def indice_sazonal(treino: pd.Series) -> pd.Series:
    """Peso de cada mês em relação à média do ano (só anos completos), média 1."""
    df = treino.to_frame("y")
    df["ano"] = df.index.year
    df["mes"] = df.index.month
    completos = df.groupby("ano")["y"].transform("count") == 12
    base = df[completos].copy()
    base["ind"] = base["y"] / base.groupby("ano")["y"].transform("mean")
    ind = base.groupby("mes")["ind"].mean()
    return ind / ind.mean()


def prever_indice_nivel(treino: pd.Series, datas_prev: pd.DatetimeIndex) -> np.ndarray:
    """Média dos últimos 12 meses multiplicada pelo índice sazonal do mês."""
    ind = indice_sazonal(treino)
    nivel = treino.iloc[-12:].mean()
    return np.array([nivel * ind[d.month] for d in datas_prev])


def prever_ets(treino: pd.Series, datas_prev: pd.DatetimeIndex) -> np.ndarray:
    """Suavização exponencial com tendência amortecida e sazonalidade, no logaritmo."""
    y = np.log(treino.clip(lower=1e-3).values)
    modelo = ExponentialSmoothing(
        y,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit()
    return np.exp(modelo.forecast(len(datas_prev)))


def _atributos(mes: int, h: int, lag12: float, nivel: float) -> list[float]:
    angulo = 2 * np.pi * mes / 12
    return [np.sin(angulo), np.cos(angulo), h, np.log(max(lag12, 1e-6) / nivel)]


def prever_random_forest(treino: pd.Series, datas_prev: pd.DatetimeIndex) -> np.ndarray:
    """Floresta aleatória que prevê a razão entre o mês-alvo e o nível recente."""
    y = np.maximum(treino.values, 1e-6)
    datas = treino.index
    n = len(y)

    X, alvo = [], []
    for origem in range(11, n - 1):
        nivel = y[origem - 11 : origem + 1].mean()
        for h in range(1, HORIZONTE + 1):
            t = origem + h
            if t >= n:
                break
            X.append(_atributos(datas[t].month, h, y[t - 12], nivel))
            alvo.append(np.log(y[t] / nivel))

    modelo = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    modelo.fit(np.array(X), np.array(alvo))

    nivel = y[-12:].mean()
    X_prev = [
        _atributos(datas_prev[h - 1].month, h, y[n - 12 + (h - 1)], nivel)
        for h in range(1, len(datas_prev) + 1)
    ]
    return nivel * np.exp(modelo.predict(np.array(X_prev)))


def montar_modelos() -> dict:
    modelos = {
        REFERENCIA: prever_referencia,
        "Índice sazonal x nível recente": prever_indice_nivel,
    }
    if ExponentialSmoothing is not None:
        modelos["Suavização exponencial (log)"] = prever_ets
    else:
        print("Aviso: statsmodels não encontrado; suavização exponencial pulada.")
    if RandomForestRegressor is not None:
        modelos["Random Forest"] = prever_random_forest
    else:
        print("Aviso: scikit-learn não encontrado; Random Forest pulado.")
    return modelos


# --------------------------------------------------------------- avaliação
def metricas(real: np.ndarray, prev: np.ndarray) -> dict:
    erro = prev - real
    return {
        "mae": np.abs(erro).mean(),
        "wape_%": np.abs(erro).sum() / real.sum() * 100,
        "erro_total_12m_%": erro.sum() / real.sum() * 100,
    }


def janelas_de_teste(serie: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    fim = serie.index[-1]
    out = []
    for k in range(N_JANELAS):
        fim_k = fim - pd.DateOffset(months=12 * k)
        ini_k = fim_k - pd.DateOffset(months=HORIZONTE - 1)
        out.append((ini_k, fim_k))
    return out


def avaliar_modelos(serie: pd.Series, modelos: dict):
    resultados, previsoes = [], []
    for ini, fim in janelas_de_teste(serie):
        treino = serie[serie.index < ini]
        teste = serie[ini:fim]
        if len(treino) < 36 or len(teste) < HORIZONTE:
            print(f"Janela {ini:%Y-%m} a {fim:%Y-%m} ignorada (dados insuficientes).")
            continue
        rotulo = f"{ini:%Y-%m} a {fim:%Y-%m}"
        bloco = pd.DataFrame({"janela": rotulo, "data": teste.index, "real": teste.values})
        for nome, funcao in modelos.items():
            prev = funcao(treino, teste.index)
            bloco[nome] = prev
            resultados.append({"janela": rotulo, "modelo": nome,
                               **metricas(teste.values, prev)})
        previsoes.append(bloco)
    return pd.DataFrame(resultados), pd.concat(previsoes, ignore_index=True)


# ---------------------------------------------------------------- gráficos
def salvar(fig, nome: str) -> None:
    PASTA_GRAFICOS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / nome, dpi=150)
    plt.close(fig)


def grafico_ultima_janela(previsoes: pd.DataFrame, modelos: dict) -> None:
    janela = previsoes["janela"].iloc[0]
    bloco = previsoes[previsoes["janela"] == janela]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(bloco["data"], bloco["real"], color="black", linewidth=2.5, label="Real")
    for nome in modelos:
        estilo = "--" if nome != REFERENCIA else ":"
        ax.plot(bloco["data"], bloco[nome], estilo, label=nome)
    ax.set_title(f"Previsão x real, janela de teste {janela}")
    ax.set_ylabel(ROTULO_ALVO[ALVO])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    salvar(fig, "06_previsao_ultima_janela.png")


def grafico_wape(resultados: pd.DataFrame) -> None:
    tabela = resultados.pivot(index="modelo", columns="janela", values="wape_%")
    tabela = tabela.loc[resultados["modelo"].unique()]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    tabela.plot(kind="bar", ax=ax, rot=15)
    ax.set_title("Erro de previsão (WAPE, %) por modelo e janela de teste; menor é melhor")
    ax.set_ylabel("WAPE (%)")
    ax.set_xlabel("")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="Janela de teste", fontsize=8)
    salvar(fig, "07_erro_por_modelo.png")


def grafico_futuro(serie: pd.Series, futuro: pd.Series, nome_modelo: str) -> None:
    historico = serie.iloc[-36:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(historico.index, historico.values, color="#1f4e79", label="Histórico")
    ax.plot(futuro.index, futuro.values, "--", color="#c0504d", linewidth=2,
            label=f"Previsão ({nome_modelo})")
    ax.set_title("Previsão dos próximos 12 meses")
    ax.set_ylabel(ROTULO_ALVO[ALVO])
    ax.legend()
    ax.grid(alpha=0.3)
    salvar(fig, "08_previsao_proximos_12m.png")


# --------------------------------------------------------------------- main
def main() -> None:
    serie = carregar()
    modelos = montar_modelos()

    registrar("=== MODELAGEM ===")
    registrar(f"Alvo: {ALVO} ({ROTULO_ALVO[ALVO]})")
    registrar(f"Série: {serie.index[0]:%Y-%m} a {serie.index[-1]:%Y-%m} ({len(serie)} meses)")
    registrar(f"Modelos: {', '.join(modelos)}")
    registrar()

    resultados, previsoes = avaliar_modelos(serie, modelos)

    registrar("=== ERRO POR JANELA (WAPE, %; menor é melhor) ===")
    wape = resultados.pivot(index="modelo", columns="janela", values="wape_%")
    wape = wape.loc[list(modelos)]
    registrar(wape.round(1).to_string())
    registrar()

    media = (
        resultados.groupby("modelo")[["wape_%", "mae", "erro_total_12m_%"]]
        .mean()
        .loc[list(modelos)]
        .sort_values("wape_%")
    )
    registrar("=== MÉDIA DAS JANELAS ===")
    registrar("mae: erro médio absoluto em " + ROTULO_ALVO[ALVO])
    registrar("erro_total_12m_%: viés do total de 12 meses (positivo = superestimou)")
    registrar(media.round(1).to_string())
    registrar()

    melhor = media.index[0]
    wape_ref = media.loc[REFERENCIA, "wape_%"]
    wape_melhor = media.loc[melhor, "wape_%"]
    if melhor == REFERENCIA:
        registrar("Nenhum modelo superou a referência simples nas janelas testadas.")
    else:
        ganho = (1 - wape_melhor / wape_ref) * 100
        registrar(
            f"Melhor modelo: {melhor}. WAPE médio {wape_melhor:.1f}% contra "
            f"{wape_ref:.1f}% da referência: erro {ganho:.0f}% menor."
        )
    registrar()

    # previsão dos próximos 12 meses com o melhor modelo, treinado em toda a série
    datas_futuras = pd.date_range(serie.index[-1] + pd.DateOffset(months=1),
                                  periods=HORIZONTE, freq="MS")
    prev_futura = pd.Series(modelos[melhor](serie, datas_futuras), index=datas_futuras)
    registrar(f"=== PREVISÃO DOS PRÓXIMOS {HORIZONTE} MESES ({melhor}) ===")
    registrar(prev_futura.round(1).to_string())
    registrar(f"Total: {prev_futura.sum():,.0f} ({ROTULO_ALVO[ALVO]})")

    PASTA_TRATADOS.mkdir(exist_ok=True)
    resultados.to_csv(PASTA_TRATADOS / "resultados_modelos.csv", index=False)
    previsoes.to_csv(PASTA_TRATADOS / "previsoes_teste.csv", index=False)
    prev_futura.rename("previsao").to_csv(PASTA_TRATADOS / "previsao_proximos_12m.csv",
                                          index_label="data")
    (PASTA_TRATADOS / "resumo_modelagem.txt").write_text(
        "\n".join(linhas_resumo), encoding="utf-8"
    )

    grafico_ultima_janela(previsoes, modelos)
    grafico_wape(resultados)
    grafico_futuro(serie, prev_futura, melhor)
    print(f"\nGráficos salvos em: {PASTA_GRAFICOS}")
    print("Tabelas salvas em: dados_tratados/ (resultados_modelos.csv, previsoes_teste.csv, "
          "previsao_proximos_12m.csv, resumo_modelagem.txt)")


if __name__ == "__main__":
    main()