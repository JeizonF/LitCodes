import pandas as pd
import numpy as np
import joblib


# =====================================================
# CONFIGURAÇÃO
# =====================================================

TAMANHO_JANELA = 100
PASSO = 25



# =====================================================
# FEATURES (MESMAS DO TREINO)
# =====================================================

def extrair_features(janela):

    c1 = janela["canal1"].astype(float).values
    c2 = janela["canal2"].astype(float).astype(float).values


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


        # diferença
        np.mean(c1-c2),
        np.std(c1-c2),


        # energia
        np.sqrt(np.mean(c1**2)),
        np.sqrt(np.mean(c2**2))

    ]


    return features



# =====================================================
# JANELAS
# =====================================================

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




# =====================================================
# OPENBCI
# =====================================================

def carregar_openbci():

    print("\nCarregando OpenBCI...")


    df=pd.read_csv(
        "dados/openBCI_raw_2018-07-02_15-46-30.txt",
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
        (abs(df.canal1)<500) &
        (abs(df.canal2)<500)
    ]



    df=df.reset_index(drop=True)


    print(df.describe())


    return df




# =====================================================
# PISCADA
# =====================================================

def carregar_piscada():

    print("\nCarregando piscadas...")


    raw=pd.read_csv(
        "dados/piscada.csv",
        header=None
    )


    print("\nFormato original:")
    print(raw.head())



    df=raw.iloc[4:,:3]


    df.columns=[
        "tempo",
        "canal1",
        "canal2"
    ]



    df["tempo"]=pd.to_numeric(
        df["tempo"],
        errors="coerce"
    )


    df["canal1"]=pd.to_numeric(
        df["canal1"],
        errors="coerce"
    )


    df["canal2"]=pd.to_numeric(
        df["canal2"],
        errors="coerce"
    )



    df=df.dropna()



    # mesma conversão do treino
    df["canal1"]*=1000
    df["canal2"]*=1000



    df["amostra"]=range(len(df))



    df=df[
        [
            "amostra",
            "canal1",
            "canal2"
        ]
    ]



    print("\nPiscada limpa:")
    print(df.describe())


    return df





# =====================================================
# CARREGAR MODELO
# =====================================================

print("\nCarregando modelo...")


modelo=joblib.load(
    "modelo_svm.pkl"
)


scaler=joblib.load(
    "scaler.pkl"
)



# =====================================================
# MENU
# =====================================================


print(
"""
Digite:

1 - testar OpenBCI normal
2 - testar piscada
"""
)



opcao=input("> ")



if opcao=="1":

    dados=carregar_openbci()


elif opcao=="2":

    dados=carregar_piscada()


else:

    print("Opção inválida")
    exit()



# =====================================================
# CLASSIFICAÇÃO
# =====================================================


print("\nCriando janelas...")


X=criar_janelas(
    dados
)



print(
    "Quantidade de janelas:",
    len(X)
)



# conferir features

print(
    "Quantidade de features:",
    X.shape[1]
)



X=scaler.transform(
    X
)



pred=modelo.predict(
    X
)



prob=modelo.predict_proba(
    X
)



print("\n====================")
print("RESULTADO")
print("====================")



resultados=[]



for i,p in enumerate(pred):


    confianca=max(prob[i])*100


    print(
        f"{i} -> {p} "
        f"confiança: {confianca:.2f}%"
    )


    resultados.append(p)



# resumo


valores,quantidades=np.unique(
    resultados,
    return_counts=True
)



print("\nResumo:")


for v,q in zip(valores,quantidades):

    print(
        v,
        q
    )



# decisão final


final=max(
    zip(valores,quantidades),
    key=lambda x:x[1]
)



print("\n====================")
print("DECISÃO FINAL")
print("====================")


print(
    final[0],
    "->",
    round(
        final[1]/len(resultados)*100,
        2
    ),
    "%"
)