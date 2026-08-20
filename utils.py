import os  # usado para trabalhar com arquivos e pastas
import numpy as np  # usado para cálculos matemáticos
import pandas as pd  # usado para trabalhar com tabelas

# quantidade de amostras em cada janela

TAMANHO_JANELA = 100


# quantidade de amostras que a janela avança

PASSO = 25


# pasta onde ficam os arquivos

PASTA_DADOS = "dados"


# ARQUIVOS

# arquivos conhecidos pelo projeto

ARQUIVOS = {

    # arquivo do movimento de piscada
    "piscada": 
        "piscada.csv"
    ,

    # arquivos do movimento para cima
    "cima": 
        None
    ,

    # arquivos do movimento para baixo
    "baixo": 
        None
    ,

    # arquivos do movimento para esquerda
    "esquerda": [
        "olhr_esq05.csv",
        "olhr_esq04.csv",
        "olhr_esq_ampli.csv",
        "olhr_esq09.csv",
        
    ],

    # arquivos do movimento para direita
    "direita": [
        "olhr_drt05.csv",
        "olhr_drt04.csv",
        "olhr_drt03.csv",
        "olhr_drt.csv",
        "olhr_drt02.csv"
    ],

    # arquivos do sinal normal
    "normal": 
       None
    
}


def extrair_features(janela):

    
    # OBTÉM OS SINAIS
    

    c1 = janela["canal1"].astype(float).values
    c2 = janela["canal2"].astype(float).values


    
    # DIFERENÇA ENTRE OS CANAIS
    

    diferenca = c1 - c2


    
    # EIXO DAS AMOSTRAS
    

    amostras = np.arange(len(c1))


    
    # INCLINAÇÃO DOS SINAIS
    

    inclinacao_c1 = np.polyfit(
        amostras,
        c1,
        1
    )[0]

    inclinacao_c2 = np.polyfit(
        amostras,
        c2,
        1
    )[0]


    
    # CORRELAÇÃO ENTRE OS CANAIS
    

    correlacao = np.corrcoef(
        c1,
        c2
    )[0, 1]

    if np.isnan(correlacao):
        correlacao = 0.0


    
    # NORMALIZAÇÃO DENTRO DA JANELA
    

    media_c1 = np.mean(c1)
    desvio_c1 = np.std(c1)

    media_c2 = np.mean(c2)
    desvio_c2 = np.std(c2)


    if desvio_c1 > 0:
        c1_norm = (
            c1 - media_c1
        ) / desvio_c1

    else:
        c1_norm = np.zeros_like(c1)


    if desvio_c2 > 0:
        c2_norm = (
            c2 - media_c2
        ) / desvio_c2

    else:
        c2_norm = np.zeros_like(c2)


    
    # NOVAS FEATURES NORMALIZADAS
    

    
    # inclinação normalizada do canal 1
    

    inclinacao_c1_norm = np.polyfit(
        amostras,
        c1_norm,
        1
    )[0]


    
    # inclinação normalizada do canal 2
    

    inclinacao_c2_norm = np.polyfit(
        amostras,
        c2_norm,
        1
    )[0]


    
    # variação normalizada canal 1
    

    variacao_c1_norm = (
        c1_norm[-1]
        -
        c1_norm[0]
    )


    
    # variação normalizada canal 2
    

    variacao_c2_norm = (
        c2_norm[-1]
        -
        c2_norm[0]
    )


    
    # amplitude normalizada canal 1
    

    amplitude_c1_norm = np.ptp(
        c1_norm
    )


    
    # amplitude normalizada canal 2
    

    amplitude_c2_norm = np.ptp(
        c2_norm
    )


    
    # RETORNA AS FEATURES
    

    return [

        
        # CANAL 1
        

        np.mean(c1),
        np.std(c1),
        np.max(c1),
        np.min(c1),
        np.ptp(c1),


        
        # CANAL 2
        

        np.mean(c2),
        np.std(c2),
        np.max(c2),
        np.min(c2),
        np.ptp(c2),


        
        # VARIAÇÃO
        

        np.mean(
            np.abs(
                np.diff(c1)
            )
        ),

        np.mean(
            np.abs(
                np.diff(c2)
            )
        ),


        
        # ENERGIA
        

        np.sqrt(
            np.mean(
                c1 ** 2
            )
        ),

        np.sqrt(
            np.mean(
                c2 ** 2
            )
        ),


        
        # DIFERENÇA ENTRE OS CANAIS
        

        np.mean(
            diferenca
        ),

        np.std(
            diferenca
        ),

        np.max(
            diferenca
        ),

        np.min(
            diferenca
        ),

        np.ptp(
            diferenca
        ),


        
        # RELAÇÃO ENTRE OS CANAIS
        

        correlacao,

        np.sqrt(
            np.mean(
                diferenca ** 2
            )
        ),


        
        # DIREÇÃO DO SINAL
        

        inclinacao_c1,

        inclinacao_c2,

        c1[-1] - c1[0],

        c2[-1] - c2[0],


        
        # POSIÇÃO DOS PICOS
        

        np.argmax(c1),

        np.argmin(c1),

        np.argmax(c2),

        np.argmin(c2),


        
        # NOVAS FEATURES NORMALIZADAS
        

        # inclinação relativa C1

        inclinacao_c1_norm,

        # inclinação relativa C2

        inclinacao_c2_norm,

        # variação relativa C1

        variacao_c1_norm,

        # variação relativa C2

        variacao_c2_norm,

        # amplitude relativa C1

        amplitude_c1_norm,

        # amplitude relativa C2

        amplitude_c2_norm
    ]


# CRIAR JANELAS

def criar_janelas(df):

    # cria lista para guardar as features

    X = []


    # percorre o sinal

    for i in range(
        0,
        len(df) - TAMANHO_JANELA,
        PASSO
    ):

        # seleciona uma janela

        janela = df.iloc[
            i:i + TAMANHO_JANELA
        ]


        # extrai as features

        X.append(
            extrair_features(
                janela
            )
        )


    # transforma em matriz numpy

    return np.array(X)


def carregar_arquivo(caminho):

    # informa qual arquivo está sendo carregado

    print("\n==")
    print("CARREGANDO ARQUIVO")
    print("==")


    print(
        f"arquivo: {os.path.basename(caminho)}"
    )


    # tenta ler o arquivo

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


    # verifica se existem pelo menos duas colunas

    if raw.shape[1] < 2:

        print(
            "\no arquivo não possui canais suficientes."
        )

        return None


    # pega no máximo três colunas

    df = raw.iloc[
        :,
        :3
    ].copy()


    # converte as colunas para números

    for coluna in df.columns:

        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce"
        )


    # remove linhas completamente vazias

    df = df.dropna(
        how="all"
    )


    
    # ARQUIVO COM TRÊS COLUNAS
    

    if df.shape[1] >= 3:

        # remove linhas sem os canais

        df = df.dropna(
            subset=[
                df.columns[1],
                df.columns[2]
            ]
        )


        # mantém as três colunas

        df = df.iloc[
            :,
            :3
        ].copy()


        # define os nomes

        df.columns = [
            "tempo",
            "canal1",
            "canal2"
        ]


    
    # ARQUIVO COM DUAS COLUNAS
    

    elif df.shape[1] == 2:

        # remove linhas inválidas

        df = df.dropna(
            subset=[
                df.columns[0],
                df.columns[1]
            ]
        )


        # define os nomes dos canais

        df.columns = [
            "canal1",
            "canal2"
        ]


        # cria tempo artificial

        df["tempo"] = range(
            len(df)
        )


    else:

        print(
            "\no arquivo não possui dados suficientes."
        )

        return None


    
    # GARANTIR CANAIS NUMÉRICOS
    

    # converte canal 1

    df["canal1"] = pd.to_numeric(
        df["canal1"],
        errors="coerce"
    )


    # converte canal 2

    df["canal2"] = pd.to_numeric(
        df["canal2"],
        errors="coerce"
    )


    # remove linhas inválidas

    df = df.dropna(
        subset=[
            "canal1",
            "canal2"
        ]
    )


    # verifica se sobraram dados

    if len(df) == 0:

        print(
            "\no arquivo não possui dados válidos."
        )

        return None


    
    # CONVERSÃO PARA MILIVOLTS
    

    # encontra o maior valor absoluto

    maior_valor = max(

        df["canal1"].abs().max(),

        df["canal2"].abs().max()
    )


    # se os valores parecem estar em volts

    if maior_valor < 10:

        # converte canal 1

        df["canal1"] *= 1000


        # converte canal 2

        df["canal2"] *= 1000


    
    # AMOSTRAS
    

    # cria o número da amostra

    df["amostra"] = range(
        len(df)
    )


    # mantém somente as colunas utilizadas

    df = df[
        [
            "amostra",
            "canal1",
            "canal2"
        ]
    ]


    # reorganiza os índices

    df = df.reset_index(
        drop=True
    )


    # retorna os dados

    return df


# LISTAR ARQUIVOS

def listar_arquivos_dados():

    # verifica se a pasta existe

    if not os.path.exists(
        PASTA_DADOS
    ):

        print(
            f"\na pasta '{PASTA_DADOS}' não existe."
        )

        return []


    # cria lista

    arquivos = []


    # percorre a pasta dados

    for nome in os.listdir(
        PASTA_DADOS
    ):

        # cria caminho

        caminho = os.path.join(
            PASTA_DADOS,
            nome
        )


        # ignora pastas

        if not os.path.isfile(
            caminho
        ):

            continue


        # pega extensão

        extensao = os.path.splitext(
            nome
        )[1].lower()


        # aceita CSV e TXT

        if extensao in [
            ".csv",
            ".txt"
        ]:

            arquivos.append(
                nome
            )


    # organiza os arquivos

    arquivos.sort()


    # retorna

    return arquivos


# CARREGAR MOVIMENTO

def carregar_movimento(nome):

    # pega os arquivos configurados para o movimento

    arquivos = ARQUIVOS.get(
        nome
    )


    # verifica se existem arquivos configurados

    if arquivos is None:

        print(
            f"\nNenhum arquivo configurado para '{nome}'."
        )

        return []


    # se foi configurado apenas um arquivo,
    # transforma em uma lista

    if isinstance(
        arquivos,
        str
    ):

        arquivos = [
            arquivos
        ]


    # cria uma lista para guardar
    # cada dataframe individualmente

    dados = []


    # percorre todos os arquivos
    # configurados para esse movimento

    for arquivo in arquivos:


        # monta o caminho completo

        caminho = os.path.join(
            PASTA_DADOS,
            arquivo
        )


        # verifica se o arquivo existe

        if not os.path.exists(
            caminho
        ):

            print(
                f"\nArquivo não encontrado: {caminho}"
            )

            continue


        # mostra qual arquivo está sendo carregado

        print(
            f"\nCarregando arquivo: {arquivo}"
        )


        # carrega o arquivo

        df = carregar_arquivo(
            caminho
        )


        # verifica se o arquivo foi carregado

        if df is None:

            print(
                f"\nNão foi possível carregar: {arquivo}"
            )

            continue


        # verifica se o arquivo possui
        # dados suficientes para criar uma janela

        if len(df) < TAMANHO_JANELA:

            print(
                f"\nArquivo ignorado por possuir "
                f"menos de {TAMANHO_JANELA} amostras:"
            )

            print(
                arquivo
            )

            continue


        # adiciona o dataframe individual
        # à lista de arquivos do movimento

        dados.append(
            df
        )


    # verifica se algum arquivo válido
    # foi encontrado

    if len(dados) == 0:

        print(
            f"\nNenhum arquivo válido encontrado para '{nome}'."
        )


    # retorna uma lista

    return dados