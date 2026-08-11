import serial  # usado para receber os dados pela porta serial
import time  # usado para esperar a conexão serial estabilizar
import numpy as np  # usado para cálculos matemáticos
import pandas as pd  # usado para organizar os dados
import joblib  # usado para carregar o modelo salvo
import pyautogui  # usado para controlar o mouse

from collections import deque  # usado para armazenar as últimas amostras

from utils import (
    extrair_features,             # extrai as features do sinal
    TAMANHO_JANELA,               # define o tamanho da janela
    PASSO,                         # define o intervalo entre classificações
    MODELO,                        # define o arquivo do modelo
    PORTA,                         # define a porta do OpenBCI
    BAUDRATE,                      # define a velocidade da conexão
    LIMIAR_INTENSIDADE,            # define a intensidade mínima
    INTENSIDADE_MAXIMA,            # define a intensidade máxima
    VELOCIDADE_MINIMA,             # define a velocidade mínima do mouse
    VELOCIDADE_MAXIMA,             # define a velocidade máxima do mouse
    CONFIRMACOES_NECESSARIAS       # define quantas confirmações são necessárias
)


# permite parar o programa
# levando o mouse para o canto superior esquerdo

pyautogui.FAILSAFE = True

# define uma pequena pausa entre os comandos do mouse

pyautogui.PAUSE = 0.01


# informa que o modelo está sendo carregado

print("\ncarregando modelo...")


# carrega o modelo svm salvo

try:

    modelo = joblib.load(
        MODELO
    )

except Exception as erro:

    print("\nerro ao carregar o modelo:")
    print(erro)

    exit()


# informa que o modelo foi carregado

print("modelo carregado!")


# mostra as classes que o modelo conhece

try:

    print(
        "classes:",
        modelo.classes_
    )

except Exception:

    pass


# informa que a conexão com o OpenBCI será iniciada

print("\nconectando ao OpenBCI...")


# tenta abrir a porta serial

try:

    ser = serial.Serial(
        PORTA,
        BAUDRATE,
        timeout=1
    )

except Exception as erro:

    print("\nerro ao conectar ao OpenBCI:")
    print(erro)

    exit()


# espera a conexão estabilizar

time.sleep(2)


# informa que a conexão foi estabelecida

print("OpenBCI conectado!")


# cria o buffer que armazenará as últimas amostras

buffer = deque(
    maxlen=TAMANHO_JANELA
)


# cria o histórico das últimas classificações

historico = deque(
    maxlen=CONFIRMACOES_NECESSARIAS
)


# controla quantas amostras chegaram
# desde a última classificação

contador = 0


# calcula a intensidade do sinal

def calcular_intensidade(df):

    # pega os valores do canal 1

    canal1 = df[
        "canal1"
    ].astype(float).values

    # pega os valores do canal 2

    canal2 = df[
        "canal2"
    ].astype(float).values

    # encontra a maior intensidade do canal 1

    intensidade1 = np.max(
        np.abs(canal1)
    )

    # encontra a maior intensidade do canal 2

    intensidade2 = np.max(
        np.abs(canal2)
    )

    # usa a maior intensidade entre os dois canais

    intensidade = max(
        intensidade1,
        intensidade2
    )

    # retorna a intensidade

    return float(
        intensidade
    )


# transforma a intensidade em velocidade do mouse

def calcular_velocidade(intensidade):

    # verifica se o sinal está abaixo do limite

    if intensidade < LIMIAR_INTENSIDADE:

        # mantém o mouse parado

        return 0


    # limita a intensidade máxima

    intensidade = min(
        intensidade,
        INTENSIDADE_MAXIMA
    )


    # calcula a proporção da intensidade

    proporcao = (
        intensidade - LIMIAR_INTENSIDADE
    ) / (
        INTENSIDADE_MAXIMA - LIMIAR_INTENSIDADE
    )


    # garante que a proporção fique entre zero e um

    proporcao = np.clip(
        proporcao,
        0,
        1
    )


    # transforma a proporção em velocidade

    velocidade = (
        VELOCIDADE_MINIMA
        +
        proporcao
        *
        (
            VELOCIDADE_MAXIMA
            -
            VELOCIDADE_MINIMA
        )
    )


    # retorna a velocidade calculada

    return int(
        velocidade
    )


# classifica as amostras armazenadas no buffer

def classificar(buffer):

    # transforma o buffer em um dataframe

    df = pd.DataFrame(
        list(buffer),
        columns=[
            "canal1",
            "canal2"
        ]
    )


    # extrai as mesmas features usadas no treinamento

    features = extrair_features(
        df
    )


    # transforma as features em uma matriz

    X = np.array(
        features,
        dtype=float
    ).reshape(
        1,
        -1
    )


    # faz a previsão usando o modelo

    pred = modelo.predict(
        X
    )


    # pega a classe prevista

    classe = str(
        pred[0]
    )


    # retorna a classe e a janela

    return classe, df


# verifica se o movimento foi confirmado

def confirmar_movimento(classe):

    # adiciona a classe ao histórico

    historico.append(
        classe
    )


    # verifica se já existem classificações suficientes

    if len(historico) < CONFIRMACOES_NECESSARIAS:

        return None


    # conta quantas vezes cada classe apareceu

    valores, quantidades = np.unique(
        list(historico),
        return_counts=True
    )


    # encontra a classe mais frequente

    indice = np.argmax(
        quantidades
    )


    # pega a classe vencedora

    classe_final = valores[
        indice
    ]


    # pega quantas vezes ela apareceu

    quantidade = quantidades[
        indice
    ]


    # verifica se houve confirmações suficientes

    if quantidade >= CONFIRMACOES_NECESSARIAS:

        return classe_final


    # retorna nada caso ainda não esteja confirmado

    return None


# movimenta o mouse de acordo com o movimento

def mover_mouse(
    classe,
    velocidade
):

    # não faz nada se a velocidade for zero

    if velocidade <= 0:

        return


    # movimenta o mouse para cima

    if classe == "cima":

        pyautogui.moveRel(
            0,
            -velocidade
        )


    # movimenta o mouse para baixo

    elif classe == "baixo":

        pyautogui.moveRel(
            0,
            velocidade
        )


    # movimenta o mouse para a esquerda

    elif classe == "esquerda":

        pyautogui.moveRel(
            -velocidade,
            0
        )


    # movimenta o mouse para a direita

    elif classe == "direita":

        pyautogui.moveRel(
            velocidade,
            0
        )


# mostra as instruções do programa

print("\n==============================")
print("modo tempo real")
print("==============================")

print("\nmovimentos:")

print("cima -> mouse para cima")
print("baixo -> mouse para baixo")
print("esquerda -> mouse para esquerda")
print("direita -> mouse para direita")
print("normal -> mouse parado")
print("piscada -> reservada para clique")

print("\npressione CTRL+C para encerrar")
print()


# inicia o loop de leitura

try:

    while True:

        # lê uma linha enviada pela porta serial

        linha = ser.readline()


        # ignora quando nenhuma informação chegou

        if not linha:

            continue


        try:

            # transforma os bytes recebidos em texto

            linha = linha.decode(
                errors="ignore"
            ).strip()


            # ignora linhas vazias

            if not linha:

                continue


            # separa os valores recebidos pela vírgula

            valores = linha.split(",")


            # verifica se existem os três valores esperados

            if len(valores) < 3:

                continue


            # pega o valor do canal 1

            canal1 = float(
                valores[1]
            )


            # pega o valor do canal 2

            canal2 = float(
                valores[2]
            )


            # adiciona os canais ao buffer

            buffer.append(
                [
                    canal1,
                    canal2
                ]
            )


            # aumenta o contador

            contador += 1


            # espera preencher a primeira janela

            if len(buffer) < TAMANHO_JANELA:

                continue


            # espera chegar o número de amostras definido pelo passo

            if contador < PASSO:

                continue


            # reinicia o contador

            contador = 0


            # classifica a janela atual

            classe, df = classificar(
                buffer
            )


            # calcula a intensidade do sinal

            intensidade = calcular_intensidade(
                df
            )


            # transforma a intensidade em velocidade

            velocidade = calcular_velocidade(
                intensidade
            )


            # verifica se o movimento foi confirmado

            movimento = confirmar_movimento(
                classe
            )


            # mostra as informações no terminal

            print(
                f"\r"
                f"classe: {classe:<10} | "
                f"intensidade: {intensidade:7.2f} | "
                f"velocidade: {velocidade:2d} | "
                f"confirmado: {str(movimento):<10}",
                end="",
                flush=True
            )


            # verifica se foi confirmado um movimento do mouse

            if movimento in [
                "cima",
                "baixo",
                "esquerda",
                "direita"
            ]:

                # movimenta o mouse

                mover_mouse(
                    movimento,
                    velocidade
                )


            # verifica se foi detectada uma piscada

            elif movimento == "piscada":

                # por enquanto a piscada não executa nenhuma ação

                pass


        except ValueError:

            # ignora linhas que não possuem números válidos

            continue


        except Exception:

            # ignora outros erros de uma linha
            # e continua recebendo os dados

            continue


# permite encerrar o programa com CTRL+C

except KeyboardInterrupt:

    print(
        "\n\nprograma encerrado pelo usuário."
    )


# interrompe caso o mouse seja levado para o canto

except pyautogui.FailSafeException:

    print(
        "\n\nfailsafe do mouse ativado."
    )

    print(
        "programa interrompido."
    )


# mostra outros erros inesperados

except Exception as erro:

    print(
        "\n\nerro durante a execução:"
    )

    print(
        erro
    )


# fecha a conexão serial ao finalizar

finally:

    try:

        ser.close()

    except Exception:

        pass


    print(
        "\nconexão encerrada."
    )

