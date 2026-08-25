import os
import tkinter as tk

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÕES DOS DADOS
# ============================================================

TAMANHO_JANELA = 100
PASSO = 25

PASTA_DADOS = "dados"

MODELO_PATH = "modelo_svm.pkl"

ARQUIVO_TESTE = "dados/olhr_esq04.csv"


# ============================================================
# CONFIGURAÇÕES DO MOUSE
# ============================================================

LARGURA_JANELA = 800
ALTURA_JANELA = 600

VELOCIDADE_MINIMA = 3
VELOCIDADE_MAXIMA = 30

TEMPO_MOUSE = 30

# ------------------------------------------------------------
# IMPORTANTE
#
# Essa configuração define aproximadamente qual amplitude
# será considerada como 100% de intensidade.
#
# Se a intensidade ficar sempre muito alta, aumente.
# Se ficar sempre muito baixa, diminua.
# ------------------------------------------------------------

INTENSIDADE_MAXIMA = 50.0


# ============================================================
# ARQUIVOS
# ============================================================

ARQUIVOS = {

    "piscada":
        "piscada.csv",

    "cima":
        None,

    "baixo":
        None,

    "esquerda": [
        "olhr_esq05.csv",
        "olhr_esq04.csv",
        "olhr_esq_ampli.csv",
        "olhr_esq09.csv",
    ],

    "direita": [
        "olhr_drt05.csv",
        "olhr_drt04.csv",
        "olhr_drt03.csv",
        "olhr_drt.csv",
        "olhr_drt02.csv"
    ],

    "normal":
        None
}


# ============================================================
# EXTRAIR FEATURES
# ============================================================

def extrair_features(janela):

    c1 = janela["canal1"].astype(float).values
    c2 = janela["canal2"].astype(float).values

    amostras = np.arange(len(c1))

    media_c1 = np.mean(c1)
    media_c2 = np.mean(c2)

    desvio_c1 = np.std(c1)
    desvio_c2 = np.std(c2)

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------

    if desvio_c1 > 0:
        c1_norm = (c1 - media_c1) / desvio_c1
    else:
        c1_norm = np.zeros_like(c1)

    if desvio_c2 > 0:
        c2_norm = (c2 - media_c2) / desvio_c2
    else:
        c2_norm = np.zeros_like(c2)

    # --------------------------------------------------------
    # DIFERENÇA ENTRE OS CANAIS
    # --------------------------------------------------------

    diferenca_norm = c1_norm - c2_norm

    # --------------------------------------------------------
    # INCLINAÇÕES
    # --------------------------------------------------------

    inclinacao_c1 = np.polyfit(
        amostras,
        c1_norm,
        1
    )[0]

    inclinacao_c2 = np.polyfit(
        amostras,
        c2_norm,
        1
    )[0]

    inclinacao_diferenca = np.polyfit(
        amostras,
        diferenca_norm,
        1
    )[0]

    # --------------------------------------------------------
    # CORRELAÇÃO
    # --------------------------------------------------------

    correlacao = np.corrcoef(
        c1_norm,
        c2_norm
    )[0, 1]

    if np.isnan(correlacao):
        correlacao = 0.0

    # --------------------------------------------------------
    # VARIAÇÕES
    # --------------------------------------------------------

    variacao_c1 = (
        c1_norm[-1] -
        c1_norm[0]
    )

    variacao_c2 = (
        c2_norm[-1] -
        c2_norm[0]
    )

    variacao_diferenca = (
        diferenca_norm[-1] -
        diferenca_norm[0]
    )

    # --------------------------------------------------------
    # AMPLITUDES
    # --------------------------------------------------------

    amplitude_c1 = np.ptp(c1_norm)
    amplitude_c2 = np.ptp(c2_norm)

    amplitude_diferenca = np.ptp(
        diferenca_norm
    )

    # --------------------------------------------------------
    # VARIAÇÃO MÉDIA
    # --------------------------------------------------------

    variacao_media_c1 = np.mean(
        np.abs(
            np.diff(c1_norm)
        )
    )

    variacao_media_c2 = np.mean(
        np.abs(
            np.diff(c2_norm)
        )
    )

    # --------------------------------------------------------
    # DESVIO DAS VARIAÇÕES
    # --------------------------------------------------------

    desvio_variacao_c1 = np.std(
        np.diff(c1_norm)
    )

    desvio_variacao_c2 = np.std(
        np.diff(c2_norm)
    )

    # --------------------------------------------------------
    # ENERGIA DA DIFERENÇA
    # --------------------------------------------------------

    energia_diferenca = np.sqrt(
        np.mean(
            diferenca_norm ** 2
        )
    )

    # --------------------------------------------------------
    # RETORNO
    # --------------------------------------------------------

    return [

        inclinacao_c1,
        inclinacao_c2,

        variacao_c1,
        variacao_c2,

        amplitude_c1,
        amplitude_c2,

        variacao_media_c1,
        variacao_media_c2,

        desvio_variacao_c1,
        desvio_variacao_c2,

        correlacao,

        inclinacao_diferenca,

        variacao_diferenca,

        amplitude_diferenca,

        energia_diferenca
    ]


# ============================================================
# CRIAR JANELAS DE FEATURES
# ============================================================

def criar_janelas(df):

    X = []

    if df is None:
        return np.array([])

    if len(df) < TAMANHO_JANELA:
        return np.array([])

    # +1 para não perder a última janela
    for i in range(
        0,
        len(df) - TAMANHO_JANELA + 1,
        PASSO
    ):

        janela = df.iloc[
            i:i + TAMANHO_JANELA
        ]

        features = extrair_features(
            janela
        )

        X.append(features)

    return np.array(X)


# ============================================================
# CRIAR JANELAS ORIGINAIS
#
# ESSA FUNÇÃO É A QUE ESTAVA FALTANDO.
#
# Ela mantém os sinais originais para calcular a intensidade.
# ============================================================

def criar_janelas_originais(df):

    janelas = []

    if df is None:
        return janelas

    if len(df) < TAMANHO_JANELA:
        return janelas

    # +1 para incluir a última janela
    for i in range(
        0,
        len(df) - TAMANHO_JANELA + 1,
        PASSO
    ):

        janela = df.iloc[
            i:i + TAMANHO_JANELA
        ].copy()

        janela = janela.reset_index(
            drop=True
        )

        janelas.append(
            janela
        )

    return janelas


# ============================================================
# CARREGAR ARQUIVO
# ============================================================

def carregar_arquivo(caminho):

    print("\n==")
    print("CARREGANDO ARQUIVO")
    print("==")

    print(
        f"arquivo: {os.path.basename(caminho)}"
    )

    try:

        raw = pd.read_csv(
            caminho,
            header=None
        )

    except Exception as erro:

        print(
            f"\nerro ao ler o arquivo: {erro}"
        )

        return None

    if raw.shape[1] < 2:

        print(
            "\no arquivo não possui canais suficientes."
        )

        return None

    # Pega no máximo 3 colunas
    df = raw.iloc[:, :3].copy()

    # --------------------------------------------------------
    # CONVERTER PARA NÚMERO
    # --------------------------------------------------------

    for coluna in df.columns:

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )

    df = df.dropna(
        how="all"
    )

    # --------------------------------------------------------
    # 3 COLUNAS
    # --------------------------------------------------------

    if df.shape[1] >= 3:

        df = df.dropna(
            subset=[
                df.columns[1],
                df.columns[2]
            ]
        )

        df = df.iloc[:, :3].copy()

        df.columns = [
            "tempo",
            "canal1",
            "canal2"
        ]

    # --------------------------------------------------------
    # 2 COLUNAS
    # --------------------------------------------------------

    elif df.shape[1] == 2:

        df = df.dropna(
            subset=[
                df.columns[0],
                df.columns[1]
            ]
        )

        df.columns = [
            "canal1",
            "canal2"
        ]

        df["tempo"] = range(
            len(df)
        )

    else:

        print(
            "\no arquivo não possui dados suficientes."
        )

        return None

    # --------------------------------------------------------
    # GARANTIR CANAIS NUMÉRICOS
    # --------------------------------------------------------

    df["canal1"] = pd.to_numeric(
        df["canal1"],
        errors="coerce"
    )

    df["canal2"] = pd.to_numeric(
        df["canal2"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "canal1",
            "canal2"
        ]
    )

    if len(df) == 0:

        print(
            "\no arquivo não possui dados válidos."
        )

        return None

    # --------------------------------------------------------
    # CONVERSÃO PARA MILIVOLTS
    # --------------------------------------------------------

    maior_valor = max(

        df["canal1"].abs().max(),

        df["canal2"].abs().max()
    )

    if maior_valor < 10:

        df["canal1"] *= 1000

        df["canal2"] *= 1000

        print(
            "Sinais convertidos para mV."
        )

    # --------------------------------------------------------
    # AMOSTRAS
    # --------------------------------------------------------

    df["amostra"] = range(
        len(df)
    )

    # --------------------------------------------------------
    # MANTER COLUNAS
    # --------------------------------------------------------

    df = df[
        [
            "amostra",
            "canal1",
            "canal2"
        ]
    ]

    df = df.reset_index(
        drop=True
    )

    print(
        f"Amostras carregadas: {len(df)}"
    )

    print(
        f"Canal 1: "
        f"{df['canal1'].min():.2f} "
        f"até "
        f"{df['canal1'].max():.2f} mV"
    )

    print(
        f"Canal 2: "
        f"{df['canal2'].min():.2f} "
        f"até "
        f"{df['canal2'].max():.2f} mV"
    )

    return df


# ============================================================
# LISTAR ARQUIVOS
# ============================================================

def listar_arquivos_dados():

    if not os.path.exists(
        PASTA_DADOS
    ):

        print(
            f"\na pasta '{PASTA_DADOS}' não existe."
        )

        return []

    arquivos = []

    for nome in os.listdir(
        PASTA_DADOS
    ):

        caminho = os.path.join(
            PASTA_DADOS,
            nome
        )

        if not os.path.isfile(
            caminho
        ):
            continue

        extensao = os.path.splitext(
            nome
        )[1].lower()

        if extensao in [
            ".csv",
            ".txt"
        ]:

            arquivos.append(
                nome
            )

    arquivos.sort()

    return arquivos


# ============================================================
# CARREGAR MOVIMENTO
# ============================================================

def carregar_movimento(nome):

    arquivos = ARQUIVOS.get(
        nome
    )

    if arquivos is None:

        print(
            f"\nNenhum arquivo configurado para '{nome}'."
        )

        return []

    if isinstance(
        arquivos,
        str
    ):

        arquivos = [
            arquivos
        ]

    dados = []

    for arquivo in arquivos:

        caminho = os.path.join(
            PASTA_DADOS,
            arquivo
        )

        if not os.path.exists(
            caminho
        ):

            print(
                f"\nArquivo não encontrado: {caminho}"
            )

            continue

        print(
            f"\nCarregando arquivo: {arquivo}"
        )

        df = carregar_arquivo(
            caminho
        )

        if df is None:

            print(
                f"\nNão foi possível carregar: {arquivo}"
            )

            continue

        if len(df) < TAMANHO_JANELA:

            print(
                f"\nArquivo ignorado por possuir "
                f"menos de {TAMANHO_JANELA} amostras:"
            )

            print(
                arquivo
            )

            continue

        dados.append(
            df
        )

    if len(dados) == 0:

        print(
            f"\nNenhum arquivo válido encontrado para '{nome}'."
        )

    return dados


# ============================================================
# CALCULAR INTENSIDADE
# ============================================================

def calcular_intensidade(janela):

    """
    Calcula a intensidade usando a amplitude ORIGINAL
    dos sinais.

    NÃO usa as features normalizadas da SVM.
    """

    try:

        if not isinstance(
            janela,
            pd.DataFrame
        ):

            return 0.0

        if (
            "canal1" not in janela.columns
            or
            "canal2" not in janela.columns
        ):

            return 0.0

        canal1 = pd.to_numeric(
            janela["canal1"],
            errors="coerce"
        ).dropna().values.astype(float)

        canal2 = pd.to_numeric(
            janela["canal2"],
            errors="coerce"
        ).dropna().values.astype(float)

        if (
            len(canal1) == 0
            or
            len(canal2) == 0
        ):

            return 0.0

        # ----------------------------------------------------
        # REMOVE O OFFSET
        # ----------------------------------------------------

        canal1 = canal1 - np.median(canal1)
        canal2 = canal2 - np.median(canal2)

        # ----------------------------------------------------
        # RMS DOS DOIS CANAIS
        # ----------------------------------------------------

        rms_c1 = np.sqrt(
            np.mean(
                canal1 ** 2
            )
        )

        rms_c2 = np.sqrt(
            np.mean(
                canal2 ** 2
            )
        )

        rms = np.sqrt(
            (
                rms_c1 ** 2
                +
                rms_c2 ** 2
            ) / 2
        )

        # ----------------------------------------------------
        # AMPLITUDE PICO A PICO
        # ----------------------------------------------------

        amplitude_c1 = np.ptp(
            canal1
        )

        amplitude_c2 = np.ptp(
            canal2
        )

        amplitude = (
            amplitude_c1
            +
            amplitude_c2
        ) / 2.0

        # ----------------------------------------------------
        # COMBINA RMS + AMPLITUDE
        #
        # RMS evita depender de apenas dois pontos.
        # Amplitude ajuda a representar a força do movimento.
        # ----------------------------------------------------

        medida = (
            0.5 * rms
            +
            0.5 * (amplitude / 2.0)
        )

        # ----------------------------------------------------
        # NORMALIZAÇÃO
        # ----------------------------------------------------

        intensidade = (
            medida /
            INTENSIDADE_MAXIMA
        )

        intensidade = max(
            0.0,
            min(
                1.0,
                intensidade
            )
        )

        return float(
            intensidade
        )

    except Exception as erro:

        print(
            f"Erro ao calcular intensidade: {erro}"
        )

        return 0.0


# ============================================================
# CALCULAR VELOCIDADE
# ============================================================

def calcular_velocidade(intensidade):

    intensidade = max(
        0.0,
        min(
            1.0,
            float(intensidade)
        )
    )

    velocidade = (
        VELOCIDADE_MINIMA
        +
        (
            intensidade ** 1.5
        )
        *
        (
            VELOCIDADE_MAXIMA
            -
            VELOCIDADE_MINIMA
        )
    )

    return int(
        velocidade
    )


# ============================================================
# MOUSE VIRTUAL
# ============================================================

class MouseVirtual:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "Mouse Virtual - SVM"
        )

        self.root.geometry(
            f"{LARGURA_JANELA}x{ALTURA_JANELA}"
        )

        self.canvas = tk.Canvas(
            self.root,
            width=LARGURA_JANELA,
            height=ALTURA_JANELA,
            bg="white"
        )

        self.canvas.pack()

        # ----------------------------------------------------
        # CENTRO
        # ----------------------------------------------------

        self.centro_x = (
            LARGURA_JANELA // 2
        )

        self.centro_y = (
            ALTURA_JANELA // 2
        )

        self.x = self.centro_x
        self.y = self.centro_y

        self.raio = 10

        # ----------------------------------------------------
        # CURSOR
        # ----------------------------------------------------

        self.cursor = self.canvas.create_oval(

            self.x - self.raio,
            self.y - self.raio,

            self.x + self.raio,
            self.y + self.raio,

            fill="black"
        )

        # ----------------------------------------------------
        # TÍTULO
        # ----------------------------------------------------

        self.texto = self.canvas.create_text(

            self.centro_x,
            30,

            text="Mouse Virtual - SVM",

            font=("Arial", 18)
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status = self.canvas.create_text(

            self.centro_x,

            ALTURA_JANELA - 50,

            text="Normal",

            font=("Arial", 14)
        )

        # ----------------------------------------------------
        # INFORMAÇÕES
        # ----------------------------------------------------

        self.info = self.canvas.create_text(

            self.centro_x,

            ALTURA_JANELA - 25,

            text="Velocidade: 0 | Intensidade: 0%",

            font=("Arial", 11)
        )

        self.root.update()

    # ========================================================
    # ATUALIZAR CURSOR
    # ========================================================

    def atualizar_cursor(self):

        self.canvas.coords(

            self.cursor,

            self.x - self.raio,
            self.y - self.raio,

            self.x + self.raio,
            self.y + self.raio
        )

        self.root.update()

    # ========================================================
    # MOVER MOUSE
    # ========================================================

    def mover(
        self,
        classe,
        intensidade
    ):

        velocidade = calcular_velocidade(
            intensidade
        )

        # ----------------------------------------------------
        # MOVIMENTO
        # ----------------------------------------------------

        if classe == "cima":

            self.y -= velocidade

        elif classe == "baixo":

            self.y += velocidade

        elif classe == "esquerda":

            self.x -= velocidade

        elif classe == "direita":

            self.x += velocidade

        # ----------------------------------------------------
        # LIMITES
        # ----------------------------------------------------

        self.x = max(
            self.raio,
            min(
                LARGURA_JANELA - self.raio,
                self.x
            )
        )

        self.y = max(
            self.raio,
            min(
                ALTURA_JANELA - self.raio,
                self.y
            )
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.canvas.itemconfig(

            self.status,

            text=f"Movimento: {classe}"
        )

        self.canvas.itemconfig(

            self.info,

            text=(
                f"Velocidade: {velocidade} px"
                f" | Intensidade: "
                f"{intensidade * 100:.1f}%"
            )
        )

        self.atualizar_cursor()

    # ========================================================
    # CENTRO
    # ========================================================

    def voltar_centro(self):

        self.x = self.centro_x
        self.y = self.centro_y

        self.canvas.itemconfig(

            self.status,

            text="Centro"
        )

        self.canvas.itemconfig(

            self.info,

            text="Velocidade: 0 | Intensidade: 0%"
        )

        self.atualizar_cursor()

    # ========================================================
    # FECHAR
    # ========================================================

    def fechar(self):

        self.root.destroy()