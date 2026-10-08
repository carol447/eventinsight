# -*- coding: utf-8 -*-
"""EventInsight - Trabalho Prático 2.

Este arquivo continua o Trabalho 1. As etapas serão adicionadas gradualmente,
com uma justificativa metodológica antes de cada experimento.
"""

from pathlib import Path
import os
import re

# Mantém o cache gráfico dentro do projeto. Esta configuração deve ocorrer antes
# da importação do matplotlib.
PASTA_PROJETO = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PASTA_PROJETO / ".cache_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS
# -----------------------------------------------------------------------------

# O caminho é relativo ao projeto, evitando a dependência do Google Drive ou de
# uma pasta pessoal. No Colab, basta enviar o CSV para a pasta "inputs".
CAMINHO_DADOS = PASTA_PROJETO / "inputs" / "baseinicial.csv"
PASTA_EDA = PASTA_PROJETO / "outputs" / "eda"


def carregar_dados(caminho=CAMINHO_DADOS):
    """Carrega a base original sem modificar o arquivo-fonte."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Base não encontrada em: {caminho}\n"
            "Informe o caminho correto ou coloque baseinicial.csv em inputs/."
        )

    dados = pd.read_csv(caminho)
    print(f"Base carregada: {dados.shape[0]} linhas e {dados.shape[1]} colunas.")
    return dados


# -----------------------------------------------------------------------------
# 2. CORREÇÃO SEMÂNTICA E LIMPEZA
# -----------------------------------------------------------------------------

def extrair_impacto_milhoes(valor):
    """Extrai somente o valor nominal, em milhões, do impacto econômico.

    Atenção: este valor ainda NÃO representa uma moeda comum. DKK 70 milhões e
    USD 70 milhões, por exemplo, não podem ser tratados como equivalentes.
    Guardamos esse atributo para reproduzir o Trabalho 1 e documentar a
    limitação, mas ele não deve ser interpretado como valor monetário comparável.
    """
    encontrado = re.search(r"([\d.,]+)", str(valor))
    if encontrado is None:
        return pd.NA
    return float(encontrado.group(1).replace(",", "."))


def extrair_moeda(valor):
    """Preserva o prefixo monetário para auditar as moedas da base."""
    prefixo = re.match(r"\s*([^\d]+)", str(valor))
    if prefixo is None:
        return "não_identificada"
    moeda = prefixo.group(1).strip()
    # Alguns símbolos foram corrompidos pela codificação do arquivo.
    # Prefixos não ASCII aparecem como caracteres de substituição no arquivo
    # recebido; sem a fonte original não é seguro decidir se eram GBP ou EUR.
    return "simbolo_corrompido" if any(ord(caractere) > 127 for caractere in moeda) else moeda


def limpar_dados(dados_brutos):
    """Corrige o cabeçalho, os tipos e problemas objetivos de qualidade."""
    colunas_obrigatorias = {
        "Festival_Name",
        "Location",
        "Attendance_Numbers",
        "Visitor_Demographics",
        "Economic_Impact",
        "Music_Genre",
    }
    ausentes = colunas_obrigatorias.difference(dados_brutos.columns)
    if ausentes:
        raise ValueError(f"Colunas esperadas não encontradas: {sorted(ausentes)}")

    # O conteúdo mostra que o cabeçalho original está deslocado:
    # Festival_Name contém países; Location contém o público; e
    # Attendance_Numbers contém a faixa etária.
    df = pd.DataFrame(
        {
            "pais": dados_brutos["Festival_Name"].astype("string").str.strip(),
            "audience": pd.to_numeric(dados_brutos["Location"], errors="coerce"),
            "faixa_etaria": dados_brutos["Attendance_Numbers"].astype("string").str.strip(),
            "tipo_publico": dados_brutos["Visitor_Demographics"].astype("string").str.strip(),
            "impacto_original": dados_brutos["Economic_Impact"].astype("string").str.strip(),
            "genero": dados_brutos["Music_Genre"].astype("string").str.strip(),
        }
    )

    df["impacto_milhoes_nominais"] = df["impacto_original"].apply(extrair_impacto_milhoes)
    df["moeda"] = df["impacto_original"].apply(extrair_moeda)

    # Converte a faixa etária em duas colunas numéricas básicas. Idade central e
    # amplitude serão criadas depois, na etapa de engenharia de atributos.
    idades = df["faixa_etaria"].str.extract(r"(?P<idade_min>\d+)\D+(?P<idade_max>\d+)")
    df[["idade_min", "idade_max"]] = idades.astype("float")

    # A duplicata foi verificada manualmente: as duas linhas de Fiji coincidem
    # em todos os campos disponíveis. Mantê-las daria peso duplo ao mesmo padrão.
    colunas_identificacao = [
        "pais", "audience", "faixa_etaria", "tipo_publico",
        "impacto_original", "genero",
    ]
    duplicadas = int(df.duplicated(subset=colunas_identificacao).sum())
    df = df.drop_duplicates(subset=colunas_identificacao, keep="first")

    # Removemos somente registros sem alvo ou sem impacto extraível. Sem o alvo
    # não é possível treinar regressão; sem impacto não é possível reproduzir o
    # modelo do Trabalho 1. Nenhum registro é removido silenciosamente.
    antes_nulos = len(df)
    df = df.dropna(subset=["audience", "impacto_milhoes_nominais"]).reset_index(drop=True)
    removidos_nulos = antes_nulos - len(df)

    # Tipos explícitos facilitam auditoria e evitam conversões implícitas.
    df["audience"] = df["audience"].astype("int64")
    df["impacto_milhoes_nominais"] = df["impacto_milhoes_nominais"].astype("float64")

    print(f"Duplicatas exatas removidas: {duplicadas}")
    print(f"Registros removidos por campos essenciais ausentes: {removidos_nulos}")
    return df


def resumir_dados_limpos(df_limpo):
    """Exibe tipos, valores ausentes e estatísticas após a limpeza.

    Esta verificação ocorre antes da EDA para detectar erros de conversão e
    garantir que as variáveis numéricas estejam realmente prontas para análise.
    """
    print("\nESTRUTURA APÓS A LIMPEZA")
    df_limpo.info()
    print("\nESTATÍSTICAS DAS VARIÁVEIS NUMÉRICAS")
    print(df_limpo.describe().round(2).to_string())


# -----------------------------------------------------------------------------
# 5. ENGENHARIA DE ATRIBUTOS (será explorada após a EDA e o diagnóstico)
# -----------------------------------------------------------------------------

def criar_atributos(df_limpo):
    """Cria variáveis derivadas sem modificar o DataFrame limpo original."""
    df = df_limpo.copy()

    # Idade central resume a faixa; amplitude representa quão abrangente é o
    # público etário informado.
    df["idade_central"] = (df["idade_min"] + df["idade_max"]) / 2
    df["amplitude_etaria"] = df["idade_max"] - df["idade_min"]

    # Agrupar combinações como Rock/Pop pelo primeiro gênero reduz a alta
    # cardinalidade. A versão original permanece disponível em "genero".
    df["genero_primario"] = df["genero"].str.split("/").str[0].str.strip()

    # Classificação proposta originalmente pelo grupo. Ela é preservada para a
    # análise de negócio, mesmo sendo muito desbalanceada na base atual.
    df["porte_proposto"] = pd.cut(
        df["audience"],
        bins=[float("-inf"), 20_000, 80_000, 200_000, float("inf")],
        labels=["Pequeno", "Médio", "Grande", "Mega"],
        include_lowest=True,
    )

    # Para métricas por grupo, Grande e Mega são consolidados. Separadamente,
    # eles possuem apenas 14 e 2 observações, quantidade insuficiente para uma
    # comparação estável. Mesmo com 16 casos, os resultados serão exploratórios.
    df["porte_avaliacao"] = pd.cut(
        df["audience"],
        bins=[float("-inf"), 20_000, 80_000, float("inf")],
        labels=["Até 20 mil", "20 a 80 mil", "Acima de 80 mil"],
        include_lowest=True,
    )

    # Quartis são usados como diagnóstico estatístico complementar. Não recebem
    # nomes comerciais, pois o quarto quartil da amostra não é necessariamente
    # um megaevento no sentido de negócio. O rank resolve empates nos 26 valores
    # discretos de audience e mantém grupos com tamanhos semelhantes.
    df["faixa_analitica"] = pd.qcut(
        df["audience"].rank(method="first"),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    return df


# -----------------------------------------------------------------------------
# 3. AUDITORIA DA QUALIDADE
# -----------------------------------------------------------------------------

def auditar_dados(df):
    """Resume limitações que afetam a validade dos experimentos."""
    auditoria = pd.Series(
        {
            "linhas": len(df),
            "valores_distintos_audience": df["audience"].nunique(),
            "paises_distintos": df["pais"].nunique(),
            "generos_originais_distintos": df["genero"].nunique(),
            "generos_primarios_distintos": df["genero_primario"].nunique(),
            "tipos_publico_distintos": df["tipo_publico"].nunique(),
            "rotulos_monetarios_distintos": df["moeda"].nunique(),
            "valores_ausentes": int(df.isna().sum().sum()),
            "linhas_duplicadas": int(
                df.duplicated(
                    subset=["pais", "audience", "faixa_etaria", "tipo_publico", "impacto_original", "genero"]
                ).sum()
            ),
        },
        name="resultado",
    )

    print("\nAUDITORIA DA BASE")
    print(auditoria.to_string())
    print("\nEVENTOS POR PORTE PROPOSTO")
    print(df["porte_proposto"].value_counts(sort=False).to_string())
    print("\nGRUPOS CONSOLIDADOS PARA AVALIAÇÃO")
    print(df["porte_avaliacao"].value_counts(sort=False).to_string())
    print("\nQUARTIS ANALÍTICOS")
    print(df["faixa_analitica"].value_counts(sort=False).to_string())
    print("\nTIPOS DE MOEDA/PREFIXO")
    print(df["moeda"].value_counts().to_string())
    return auditoria


# -----------------------------------------------------------------------------
# 4. ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
# -----------------------------------------------------------------------------

def executar_eda(df_limpo):
    """Executa uma EDA orientada às decisões das próximas etapas.

    Os gráficos não selecionam atributos por si só. Eles diagnosticam escala,
    assimetria, valores extremos, cardinalidade e relações candidatas que serão
    testadas formalmente com validação cruzada e feature selection.
    """
    PASTA_EDA.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # 1) Distribuição do alvo: justifica métricas absolutas e relativas e avalia
    # se uma transformação logarítmica deve ser testada posteriormente.
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df_limpo["audience"], bins=20, kde=True, ax=eixos[0])
    eixos[0].set(title="Distribuição do público", xlabel="Público", ylabel="Eventos")
    sns.histplot(np.log1p(df_limpo["audience"]), bins=20, kde=True, ax=eixos[1])
    eixos[1].set(title="Distribuição de log(1 + público)", xlabel="log(1 + público)", ylabel="Eventos")
    fig.tight_layout()
    fig.savefig(PASTA_EDA / "01_distribuicao_publico.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) Boxplot: evidencia a distância dos eventos de grande escala. Pontos
    # extremos não são removidos automaticamente, pois podem ser eventos reais.
    fig, ax = plt.subplots(figsize=(10, 3.8))
    sns.boxplot(x=df_limpo["audience"], ax=ax)
    ax.set(title="Dispersão e eventos de maior público", xlabel="Público")
    fig.tight_layout()
    fig.savefig(PASTA_EDA / "02_boxplot_publico.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) Relação reproduzida do Trabalho 1. Escalas log ajudam a visualizar toda
    # a faixa, mas o impacto continua nominal e mistura moedas diferentes.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(data=df_limpo, x="impacto_milhoes_nominais", y="audience", alpha=.75, ax=ax)
    ax.set(xscale="log", yscale="log",
           title="Impacto nominal e público (escalas logarítmicas)",
           xlabel="Impacto em milhões na moeda original", ylabel="Público")
    fig.tight_layout()
    fig.savefig(PASTA_EDA / "03_impacto_vs_publico.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4) Correlação numérica: Pearson mede associação linear; Spearman mede
    # associação monotônica e é menos sensível à escala e a valores extremos.
    numericas = df_limpo[["audience", "impacto_milhoes_nominais", "idade_min", "idade_max"]]
    corr_pearson = numericas.corr(method="pearson")
    corr_spearman = numericas.corr(method="spearman")
    corr_pearson.to_csv(PASTA_EDA / "correlacao_pearson.csv", encoding="utf-8-sig")
    corr_spearman.to_csv(PASTA_EDA / "correlacao_spearman.csv", encoding="utf-8-sig")
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=eixos[0])
    eixos[0].set_title("Correlação de Pearson")
    sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=eixos[1])
    eixos[1].set_title("Correlação de Spearman")
    fig.tight_layout()
    fig.savefig(PASTA_EDA / "04_correlacoes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5) Categorias: exibimos frequência, e não ranking de público médio. Com
    # muitas categorias de ocorrência única, uma média individual seria instável.
    for coluna, nome in [("genero", "generos"), ("pais", "paises")]:
        contagem = df_limpo[coluna].value_counts().head(15).sort_values()
        fig, ax = plt.subplots(figsize=(9, 6))
        contagem.plot(kind="barh", ax=ax)
        ax.set(title=f"15 {nome} mais frequentes", xlabel="Quantidade de eventos", ylabel="")
        fig.tight_layout()
        fig.savefig(PASTA_EDA / f"05_frequencia_{nome}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 6) Impacto: o histograma serve para verificar escala, mas não representa
    # distribuição monetária real porque as moedas não foram convertidas.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df_limpo["impacto_milhoes_nominais"], bins=20, kde=True, ax=ax)
    ax.set(title="Distribuição do impacto nominal", xlabel="Milhões na moeda original", ylabel="Eventos")
    fig.tight_layout()
    fig.savefig(PASTA_EDA / "06_distribuicao_impacto_nominal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    resumo = pd.DataFrame({
        "metrica": ["Assimetria do público", "Assimetria do impacto nominal",
                    "Correlação Pearson: impacto x público", "Correlação Spearman: impacto x público",
                    "Proporção de países com uma observação", "Proporção de gêneros com uma observação"],
        "valor": [df_limpo["audience"].skew(), df_limpo["impacto_milhoes_nominais"].skew(),
                  corr_pearson.loc["impacto_milhoes_nominais", "audience"],
                  corr_spearman.loc["impacto_milhoes_nominais", "audience"],
                  (df_limpo["pais"].value_counts() == 1).mean(),
                  (df_limpo["genero"].value_counts() == 1).mean()],
    })
    resumo.to_csv(PASTA_EDA / "resumo_eda.csv", index=False, encoding="utf-8-sig")
    print("\nRESUMO QUANTITATIVO DA EDA")
    print(resumo.to_string(index=False))
    print(f"\nGráficos e tabelas da EDA salvos em: {PASTA_EDA}")
    return resumo


if __name__ == "__main__":
    dados_brutos = carregar_dados()
    df_limpo = limpar_dados(dados_brutos)
    resumir_dados_limpos(df_limpo)
    resumo_eda = executar_eda(df_limpo)
    df = criar_atributos(df_limpo)
    auditoria = auditar_dados(df)

    print("\nAMOSTRA APÓS A LIMPEZA")
    print(df.head().to_string(index=False))
