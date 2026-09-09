import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS = r".\data\data"

JANELA = 1000
PASSO = 200


# ============================================================
# ENCONTRAR TODOS OS CSVs DA PASTA
# ============================================================

arquivos = glob.glob(
    os.path.join(PASTA_DADOS, "*.csv")
)

print("\nArquivos encontrados:")

for arquivo in arquivos:
    print("-", os.path.basename(arquivo))

print(f"\nTotal de arquivos: {len(arquivos)}")


# ============================================================
# CARREGAR UM ARQUIVO
# ============================================================

def carregar_arquivo(arquivo):

    # O arquivo possui:
    #
    # x-axis,1
    # second,Volt
    # -12.0000E-03,+2.0683417E+00
    #
    # Então ignoramos a segunda linha.

    dados = pd.read_csv(
        arquivo,
        skiprows=[1],
        header=0
    )

    # Renomeia as colunas
    dados.columns = ["tempo", "sinal"]

    # Converte para número
    dados["tempo"] = pd.to_numeric(
        dados["tempo"],
        errors="coerce"
    )

    dados["sinal"] = pd.to_numeric(
        dados["sinal"],
        errors="coerce"
    )

    # Remove linhas inválidas
    dados = dados.dropna()

    return dados


# ============================================================
# CARREGAR TODOS OS ARQUIVOS
# ============================================================

dados_arquivos = {}

print("\nCarregando arquivos...\n")

for arquivo in arquivos:

    nome = os.path.splitext(
        os.path.basename(arquivo)
    )[0]

    try:

        dados = carregar_arquivo(arquivo)

        dados_arquivos[nome] = dados

        print(
            f"{nome}: "
            f"{len(dados)} amostras"
        )

    except Exception as erro:

        print(
            f"ERRO ao carregar {nome}: {erro}"
        )


# ============================================================
# GRÁFICO
# ============================================================

indice = 0
rodando = False

fig, ax = plt.subplots(
    figsize=(16, 8)
)


# ============================================================
# DESENHAR OS SINAIS
# ============================================================

def desenhar(i):

    ax.clear()

    for nome, dados in dados_arquivos.items():

        # Pega uma janela do sinal
        janela_dados = dados.iloc[
            i:i + JANELA
        ]

        # Se não houver mais dados, ignora
        if janela_dados.empty:
            continue

        tempo = janela_dados["tempo"]
        sinal = janela_dados["sinal"]

        # Desenha o sinal
        ax.plot(
            tempo,
            sinal,
            label=nome,
            alpha=0.8
        )


    # Configuração do gráfico
    ax.set_title(
        f"Comparação dos sinais | "
        f"Amostras {i} até {i + JANELA}"
    )

    ax.set_xlabel("Tempo (s)")

    ax.set_ylabel("Amplitude (V)")

    ax.legend()

    ax.grid(True)

    plt.draw()


# ============================================================
# CONTROLE DO TECLADO
# ============================================================

def on_key(event):

    global indice
    global rodando


    # Próxima janela
    if event.key == "right":

        indice += PASSO

        desenhar(indice)


    # Janela anterior
    elif event.key == "left":

        indice = max(
            0,
            indice - PASSO
        )

        desenhar(indice)


    # Espaço inicia/para
    elif event.key == " ":

        rodando = not rodando


    # Voltar ao início
    elif event.key == "home":

        indice = 0

        desenhar(indice)


# ============================================================
# CONECTAR TECLADO
# ============================================================

fig.canvas.mpl_connect(
    "key_press_event",
    on_key
)


# ============================================================
# DESENHA A PRIMEIRA JANELA
# ============================================================

desenhar(indice)


# ============================================================
# LOOP
# ============================================================

while plt.fignum_exists(fig.number):

    if rodando:

        desenhar(indice)

        indice += PASSO

        plt.pause(0.3)

    else:

        plt.pause(0.1)


plt.show()