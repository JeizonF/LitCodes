import serial
import time
import numpy as np
import pandas as pd
import joblib
from collections import deque


from utils import (
    extrair_features
)



# =====================================================
# CONFIGURAÇÃO
# =====================================================


PORTA = "COM3"

BAUDRATE = 115200


TAMANHO_JANELA = 100


MODELO = "modelo_svm.pkl"



# =====================================================
# CARREGAR MODELO
# =====================================================


print("\nCarregando modelo...")

modelo = joblib.load(
    MODELO
)


print("Modelo carregado!")



# =====================================================
# CONEXÃO OPENBCI
# =====================================================


print("\nConectando OpenBCI...")


ser = serial.Serial(
    PORTA,
    BAUDRATE,
    timeout=1
)


time.sleep(2)


print("Conectado!")



# =====================================================
# BUFFER DE DADOS
# =====================================================


buffer = deque(
    maxlen=TAMANHO_JANELA
)



# =====================================================
# CLASSIFICAÇÃO
# =====================================================


def classificar(buffer):


    df = pd.DataFrame(
        buffer,
        columns=[
            "canal1",
            "canal2"
        ]
    )


    X = extrair_features(
        df
    )


    X = np.array(
        X
    ).reshape(
        1,-1
    )


    pred = modelo.predict(
        X
    )


    classe = pred[0]


    return classe, df




# =====================================================
# INTENSIDADE
# =====================================================


def calcular_intensidade(df):


    canal1 = df["canal1"].values

    canal2 = df["canal2"].values



    intensidade = max(
        np.max(abs(canal1)),
        np.max(abs(canal2))
    )


    return intensidade




# =====================================================
# LOOP PRINCIPAL
# =====================================================


print("\nIniciando leitura em tempo real...")
print("CTRL+C para parar\n")



try:


    while True:


        linha = ser.readline()


        if not linha:
            continue



        try:

            linha = linha.decode(
                errors="ignore"
            )


            valores = linha.split(",")



            if len(valores) < 3:
                continue



            canal1 = float(
                valores[1]
            )


            canal2 = float(
                valores[2]
            )



            buffer.append(
                [
                    canal1,
                    canal2
                ]
            )



            if len(buffer) == TAMANHO_JANELA:



                classe, df = classificar(
                    buffer
                )


                intensidade = calcular_intensidade(
                    df
                )


                print(
                    f"{classe.upper()} | intensidade: {intensidade:.2f}"
                )



        except Exception as erro:

            pass



except KeyboardInterrupt:


    print("\nEncerrando...")


    ser.close()