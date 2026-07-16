import numpy as np


def extrair_features(janela):

    c1 = janela["canal1"].astype(float).values
    c2 = janela["canal2"].astype(float).values


    features = [

        # canal 1
        np.mean(c1),
        np.std(c1),
        np.max(c1),
        np.min(c1),
        np.ptp(c1),


        # canal 2
        np.mean(c2),
        np.std(c2),
        np.max(c2),
        np.min(c2),
        np.ptp(c2),


        # relação entre canais
        np.mean(c1-c2),
        np.std(c1-c2),


        # energia
        np.sqrt(np.mean(c1**2)),
        np.sqrt(np.mean(c2**2))

    ]


    return features



def criar_janelas(
        dados,
        tamanho=200,
        passo=100
):

    janelas=[]


    for inicio in range(
        0,
        len(dados)-tamanho,
        passo
    ):

        janela=dados.iloc[
            inicio:
            inicio+tamanho
        ]


        janelas.append(
            extrair_features(janela)
        )


    return np.array(janelas)