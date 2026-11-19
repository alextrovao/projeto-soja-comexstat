"""
Projeto: previsão das exportações mensais de soja em grão (Comex Stat / MDIC)
Etapa 1: coleta e limpeza

Saída: dados_tratados/soja_mensal.csv (uma linha por mês)
"""
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------- parâmetros
ANO_INICIAL = 2018
ANO_FINAL = 2026            # 2026 ainda está em andamento: o último mês pode estar incompleto
PREFIXO_NCM = "1201"        # SH4 1201 = soja, mesmo triturada. Troque para outro produto se quiser
BASE_URL = "https://balanca.mdic.gov.br/balanca/bd/comexstat-bd/ncm/EXP_{ano}.csv"

PASTA_BASE = Path(__file__).resolve().parent   # pasta onde está este script
PASTA_BRUTOS = PASTA_BASE / "dados_brutos"
PASTA_TRATADOS = PASTA_BASE / "dados_tratados"
COLUNAS = ["CO_ANO", "CO_MES", "CO_NCM", "VL_FOB", "KG_LIQUIDO"]


# ------------------------------------------------------------------- coleta
def baixar(ano: int) -> Path:
    """Baixa o CSV de exportações do ano, se ainda não existir."""
    PASTA_BRUTOS.mkdir(exist_ok=True)
    destino = PASTA_BRUTOS / f"EXP_{ano}.csv"
    if destino.exists():
        return destino

    try:
        resposta = requests.get(BASE_URL.format(ano=ano), stream=True, timeout=300)
        resposta.raise_for_status()
        with open(destino, "wb") as arquivo:
            for bloco in resposta.iter_content(chunk_size=1 << 20):
                arquivo.write(bloco)
    except requests.exceptions.RequestException as erro:
        destino.unlink(missing_ok=True)  # apaga arquivo parcial, se houver
        raise SystemExit(
            f"\nNão consegui baixar EXP_{ano}.csv automaticamente.\n"
            f"Motivo: {type(erro).__name__}\n\n"
            "Baixe os arquivos manualmente na página de base de dados bruta do MDIC\n"
            "(item 'Base de dados detalhada por NCM' > Exportação), salve-os como\n"
            f"{PASTA_BRUTOS}/EXP_AAAA.csv (um por ano, de {ANO_INICIAL} a {ANO_FINAL}) "
            "e rode o script de novo.\n"
            "Arquivos já existentes nessa pasta não são baixados outra vez."
        ) from erro
    return destino


# ------------------------------------------------------------------ leitura
def ler_produto(caminho: Path, prefixo_ncm: str) -> pd.DataFrame:
    """Lê o CSV em blocos (os arquivos são grandes) e guarda só o produto."""
    partes = []
    for bloco in pd.read_csv(
        caminho,
        sep=";",
        usecols=COLUNAS,
        dtype={"CO_NCM": str},
        chunksize=500_000,
        encoding="latin-1",
    ):
        partes.append(bloco[bloco["CO_NCM"].str.startswith(prefixo_ncm)])
    if not partes:
        return pd.DataFrame(columns=COLUNAS)
    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------- limpeza
def agregar_mensal(df: pd.DataFrame) -> pd.DataFrame:
    """Soma valor (US$ FOB) e peso (kg) por mês."""
    df = df.copy()
    df["data"] = pd.to_datetime(
        {"year": df["CO_ANO"], "month": df["CO_MES"], "day": 1}
    )
    mensal = (
        df.groupby("data", as_index=False)
        .agg(vl_fob_usd=("VL_FOB", "sum"), kg_liquido=("KG_LIQUIDO", "sum"))
        .sort_values("data")
        .reset_index(drop=True)
    )
    return mensal


def checar_qualidade(mensal: pd.DataFrame) -> pd.DataFrame:
    """Imprime verificações de qualidade e devolve a série completa mês a mês."""
    print("\n--- Verificações de qualidade ---")
    print(f"Período: {mensal['data'].min():%Y-%m} a {mensal['data'].max():%Y-%m}")
    print(f"Linhas: {len(mensal)}")

    # 1) meses faltando
    esperado = pd.date_range(mensal["data"].min(), mensal["data"].max(), freq="MS")
    faltando = esperado.difference(mensal["data"])
    print(f"Meses faltando: {len(faltando)}")
    if len(faltando):
        print(list(faltando.strftime("%Y-%m")))

    # 1b) o último mês publicado pode estar incompleto
    ultimo = mensal.iloc[-1]
    print(
        f"Último mês na base: {ultimo['data']:%Y-%m} "
        f"(US$ {ultimo['vl_fob_usd']:,.0f}). Compare com o mesmo mês de 2025 "
        "antes de usá-lo no modelo."
    )

    # 2) duplicados
    print(f"Meses duplicados: {mensal['data'].duplicated().sum()}")

    # 3) valores zerados ou negativos
    print(f"Meses com valor <= 0: {(mensal['vl_fob_usd'] <= 0).sum()}")

    # 4) preço médio implícito (US$/t) para achar valores estranhos
    mensal = mensal.set_index("data").reindex(esperado)
    mensal.index.name = "data"
    mensal["preco_usd_por_t"] = mensal["vl_fob_usd"] / (mensal["kg_liquido"] / 1000)
    print(mensal["preco_usd_por_t"].describe().round(1).to_string())

    return mensal.reset_index()


# --------------------------------------------------------------------- main
def diagnosticar_arquivos() -> None:
    """Mostra onde o script procura os arquivos e o que falta."""
    PASTA_BRUTOS.mkdir(exist_ok=True)
    print(f"Pasta dos arquivos brutos: {PASTA_BRUTOS}")
    presentes = sorted(p.name for p in PASTA_BRUTOS.iterdir() if p.is_file())
    print(f"Arquivos encontrados nessa pasta: {presentes if presentes else 'nenhum'}")

    faltando = [
        f"EXP_{ano}.csv"
        for ano in range(ANO_INICIAL, ANO_FINAL + 1)
        if not (PASTA_BRUTOS / f"EXP_{ano}.csv").exists()
    ]
    if faltando:
        print(f"Arquivos que faltam (nome exato): {faltando}")
        print(
            "Coloque-os na pasta acima. O nome precisa ser exatamente esse "
            "(no Windows, confira que não ficou 'EXP_2018.csv.csv').\n"
        )
    else:
        print("Todos os arquivos estão no lugar.\n")


def main() -> None:
    diagnosticar_arquivos()
    partes = []
    for ano in range(ANO_INICIAL, ANO_FINAL + 1):
        print(f"Processando {ano}...")
        caminho = baixar(ano)
        partes.append(ler_produto(caminho, PREFIXO_NCM))

    bruto = pd.concat(partes, ignore_index=True)
    mensal = agregar_mensal(bruto)
    mensal = checar_qualidade(mensal)

    PASTA_TRATADOS.mkdir(exist_ok=True)
    mensal.to_csv(PASTA_TRATADOS / "soja_mensal.csv", index=False)
    print("\nArquivo salvo em dados_tratados/soja_mensal.csv")


if __name__ == "__main__":
    main()