import os
import numpy as np
import pandas as pd

# ===========================
# CONFIGURAÇÕES GERAIS
# ===========================

TAMANHO_JANELA = 100
PASSO = 25

PASTA_DADOS = "dados"

ARQUIVOS = {

    "normal": "openBCI_raw_2018-07-02_15-46-30.txt",
    "piscada": "piscada.csv",

    "cima": None,
    "baixo": None,
    "esquerda": None,
    "direita": None

}


# ===========================
# FEATURES
# ===========================

def extrair_features(janela):

    c1 = janela["canal1"].astype(float).values
    c2 = janela["canal2"].astype(float).values

    return [

        np.mean(c1),
        np.std(c1),
        np.max(c1),
        np.min(c1),
        np.ptp(c1),

        np.mean(c2),
        np.std(c2),
        np.max(c2),
        np.min(c2),
        np.ptp(c2),

        np.mean(np.abs(np.diff(c1))),
        np.mean(np.abs(np.diff(c2))),

        np.sqrt(np.mean(c1**2)),
        np.sqrt(np.mean(c2**2))
    ]


# ===========================
# JANELAS
# ===========================

def criar_janelas(df):

    X=[]

    for i in range(
        0,
        len(df)-TAMANHO_JANELA,
        PASSO
    ):

        janela=df.iloc[
            i:i+TAMANHO_JANELA
        ]

        X.append(
            extrair_features(janela)
        )

    return np.array(X)


# ===========================
# NORMAL
# ===========================

def carregar_openbci():

    caminho=os.path.join(
        PASTA_DADOS,
        ARQUIVOS["normal"]
    )

    df=pd.read_csv(
        caminho,
        skiprows=5
    )

    df=df.iloc[:,:3]

    df.columns=[
        "amostra",
        "canal1",
        "canal2"
    ]

    df=df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    df=df.dropna()

    df=df[
        (abs(df.canal1)<500)&
        (abs(df.canal2)<500)
    ]

    return df.reset_index(drop=True)


# ===========================
# CSV GENÉRICO
# ===========================

def carregar_csv(nome):

    caminho=os.path.join(
        PASTA_DADOS,
        ARQUIVOS[nome]
    )

    if not os.path.exists(caminho):
        return None

    raw=pd.read_csv(
        caminho,
        header=None
    )

    df=raw.iloc[4:,:3]

    df.columns=[
        "tempo",
        "canal1",
        "canal2"
    ]

    for c in df.columns:
        df[c]=pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df=df.dropna()

    if abs(df.canal1).max()<10:
        df["canal1"]*=1000
        df["canal2"]*=1000

    df["amostra"]=range(len(df))

    return df[
        [
            "amostra",
            "canal1",
            "canal2"
        ]
    ]



# ===========================
# ALIAS NORMAL
# ===========================

def carregar_movimento(nome):

    if nome == "normal":

        return carregar_openbci()

    return carregar_csv(nome)