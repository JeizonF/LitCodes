import numpy as np
import time


# =====================================================
# MATRIZ DE LETRAS
# =====================================================


LETRAS = [

    ["A","B","C","D","E"],

    ["F","G","H","I","J"],

    ["K","L","M","N","O"],

    ["P","Q","R","S","T"],

    ["U","V","W","X","Y"],

    ["Z","_","<","OK"," "]

]



linha = 0
coluna = 0


texto = ""



# =====================================================
# MOSTRAR TECLADO
# =====================================================


def mostrar_teclado():


    print("\n====================")

    for i,linha_letras in enumerate(LETRAS):

        saida=""


        for j,letra in enumerate(linha_letras):


            if i == linha and j == coluna:

                saida += f"[{letra}] "

            else:

                saida += f" {letra}  "


        print(saida)


    print("====================")

    print(
        "Texto:",
        texto
    )



# =====================================================
# CALCULAR INTENSIDADE
# =====================================================


def calcular_intensidade(janela):


    canal1 = janela["canal1"].values

    canal2 = janela["canal2"].values



    energia = np.sqrt(
        np.mean(
            canal1**2 +
            canal2**2
        )
    )


    return energia



# =====================================================
# CONVERTER INTENSIDADE EM DISTÂNCIA
# =====================================================


def calcular_pulo(intensidade):


    if intensidade < 50:

        return 1


    elif intensidade < 100:

        return 2


    elif intensidade < 200:

        return 3


    elif intensidade < 400:

        return 5


    else:

        return 8



# =====================================================
# MOVIMENTAÇÃO
# =====================================================


def mover(direcao, intensidade):

    global linha,coluna


    pulo = calcular_pulo(
        intensidade
    )


    if direcao=="cima":

        linha -= pulo


    elif direcao=="baixo":

        linha += pulo


    elif direcao=="esquerda":

        coluna -= pulo


    elif direcao=="direita":

        coluna += pulo



    corrigir_limites()



# =====================================================
# LIMITES
# =====================================================


def corrigir_limites():

    global linha,coluna


    linha=max(
        0,
        min(
            linha,
            len(LETRAS)-1
        )
    )


    coluna=max(
        0,
        min(
            coluna,
            len(LETRAS[0])-1
        )
    )



# =====================================================
# SELECIONAR LETRA
# =====================================================


def selecionar():


    global texto


    letra = LETRAS[
        linha
    ][
        coluna
    ]



    if letra=="<":

        texto = texto[:-1]



    elif letra=="OK":

        print(
            "\nTexto final:"
        )

        print(
            texto
        )



    else:

        texto += letra



    return letra



# =====================================================
# PROCESSAR RESULTADO DO SVM
# =====================================================


def processar_movimento(
        classe,
        janela
):


    intensidade = calcular_intensidade(
        janela
    )



    print(
        "Classe:",
        classe,
        " Intensidade:",
        round(intensidade,2)
    )



    if classe=="piscada":

        letra=selecionar()

        print(
            "Selecionado:",
            letra
        )



    elif classe in [

        "cima",
        "baixo",
        "esquerda",
        "direita"

    ]:

        mover(
            classe,
            intensidade
        )



    mostrar_teclado()