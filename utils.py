import os  # usado para trabalhar com arquivos e pastas
import numpy as np  # usado para fazer cálculos matemáticos
import pandas as pd  # usado para trabalhar com tabelas

MODELO = "modelo_svm.pkl"

# define a porta serial do OpenBCI

PORTA = "COM3"

# define a velocidade da comunicação serial

BAUDRATE = 115200

# define a intensidade mínima para movimentar o mouse

LIMIAR_INTENSIDADE = 30

# define a intensidade máxima usada no cálculo
# da velocidade do mouse

INTENSIDADE_MAXIMA = 300

# define a velocidade mínima do mouse

VELOCIDADE_MINIMA = 2

# define a velocidade máxima do mouse

VELOCIDADE_MAXIMA = 15

# define quantas classificações iguais são necessárias
# para confirmar um movimento

CONFIRMACOES_NECESSARIAS = 2


# define os arquivos de cada classe

ARQUIVOS = {

    "normal":
        "openBCI_raw_2018-07-02_15-46-30.txt",

    "piscada":
        "piscada.csv",

    "cima":
        "cimabaixocsr1.csv",

    "baixo":
        "olhar_baixo.csv",

    "esquerda":
        "esquerda01.csv",

    "direita":
        "direita_teste.csv"
}




# configurações gerais

# quantidade de amostras em cada janela
TAMANHO_JANELA = 100

# quantidade de amostras que a janela avança
PASSO = 25

# pasta onde ficam os arquivos de dados
PASTA_DADOS = "dados"

# nome dos arquivos utilizados
ARQUIVOS = {

    # arquivo com o sinal normal
    "normal": "openBCI_raw_2018-07-02_15-46-30.txt",

    # arquivo com o sinal de piscada
    "piscada": "piscada.csv",

    # arquivos dos movimentos
    "cima": None,
    "baixo": None,
    "esquerda": None,
    "direita": None

}



# extrai as características de uma janela
def extrair_features(janela):

    # obtém os valores do canal 1
    c1 = janela["canal1"].astype(float).values

    # obtém os valores do canal 2
    c2 = janela["canal2"].astype(float).values

    # retorna todas as características da janela
    return [

        # média do canal 1
        np.mean(c1),

        # desvio padrão do canal 1
        np.std(c1),

        # maior valor do canal 1
        np.max(c1),

        # menor valor do canal 1
        np.min(c1),

        # amplitude do canal 1
        np.ptp(c1),

        # média do canal 2
        np.mean(c2),

        # desvio padrão do canal 2
        np.std(c2),

        # maior valor do canal 2
        np.max(c2),

        # menor valor do canal 2
        np.min(c2),

        # amplitude do canal 2
        np.ptp(c2),

        # variação média do canal 1
        np.mean(np.abs(np.diff(c1))),

        # variação média do canal 2
        np.mean(np.abs(np.diff(c2))),

        # energia do canal 1
        np.sqrt(np.mean(c1**2)),

        # energia do canal 2
        np.sqrt(np.mean(c2**2))
    ]



# cria janelas do sinal
def criar_janelas(df):

    # lista que armazenará as features
    X = []

    # percorre o sinal usando uma janela deslizante
    for i in range(
        0,
        len(df) - TAMANHO_JANELA,
        PASSO
    ):

        # seleciona uma janela do sinal
        janela = df.iloc[
            i:i + TAMANHO_JANELA
        ]

        # extrai as features da janela
        X.append(
            extrair_features(janela)
        )

    # retorna todas as janelas
    return np.array(X)



# carrega o arquivo do openbci
def carregar_openbci():

    # monta o caminho do arquivo
    caminho = os.path.join(
        PASTA_DADOS,
        ARQUIVOS["normal"]
    )

    # lê o arquivo
    df = pd.read_csv(
        caminho,
        skiprows=5
    )

    # mantém apenas as três primeiras colunas
    df = df.iloc[:, :3]

    # define o nome das colunas
    df.columns = [
        "amostra",
        "canal1",
        "canal2"
    ]

    # converte os dados para números
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # remove linhas inválidas
    df = df.dropna()

    # remove valores muito altos considerados ruído
    df = df[
        (abs(df.canal1) < 500) &
        (abs(df.canal2) < 500)
    ]

    # reorganiza os índices
    return df.reset_index(drop=True)



# carrega qualquer arquivo csv de movimento
def carregar_csv(nome):

    # verifica se o movimento existe no dicionário
    if nome not in ARQUIVOS:
        return None

    # obtém o nome do arquivo
    arquivo = ARQUIVOS[nome]

    # verifica se o arquivo foi definido
    if arquivo is None:
        return None

    # monta o caminho do arquivo
    caminho = os.path.join(
        PASTA_DADOS,
        arquivo
    )

    # verifica se o arquivo existe
    if not os.path.exists(caminho):
        return None

    # lê o arquivo
    raw = pd.read_csv(
        caminho,
        header=None
    )

    # remove as primeiras linhas e mantém três colunas
    df = raw.iloc[4:, :3]

    # define o nome das colunas
    df.columns = [
        "tempo",
        "canal1",
        "canal2"
    ]

    # converte os dados para números
    for c in df.columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    # remove linhas inválidas
    df = df.dropna()

    # converte de volts para milivolts quando necessário
    if abs(df.canal1).max() < 10:
        df["canal1"] *= 1000
        df["canal2"] *= 1000

    # cria o índice das amostras
    df["amostra"] = range(len(df))

    # retorna apenas as colunas necessárias
    return df[
        [
            "amostra",
            "canal1",
            "canal2"
        ]
    ]



def carregar_movimento(nome):

    # carrega o sinal normal
    if nome == "normal":
        return carregar_openbci()

    # carrega a piscada
    if nome == "piscada":
        return carregar_piscada()

    # verifica se existe arquivo para o movimento
    if ARQUIVOS.get(nome) is None:
        return None

    # carrega qualquer outro movimento
    return carregar_csv(nome)

# carrega o arquivo de piscada
def carregar_piscada():

    # monta o caminho do arquivo
    caminho = os.path.join(
        PASTA_DADOS,
        ARQUIVOS["piscada"]
    )

    # lê o arquivo
    raw = pd.read_csv(
        caminho,
        header=None
    )

    # remove o cabeçalho
    df = raw.iloc[4:, :3]

    # define os nomes das colunas
    df.columns = [
        "tempo",
        "canal1",
        "canal2"
    ]

    # converte para números
    for c in df.columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    # remove linhas inválidas
    df = df.dropna()

    # converte para milivolts
    df["canal1"] *= 1000
    df["canal2"] *= 1000

    # cria a coluna amostra
    df["amostra"] = range(len(df))

    return df[
        [
            "amostra",
            "canal1",
            "canal2"
        ]
    ]