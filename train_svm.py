import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# =====================================================
# CONFIGURAÇÃO
# =====================================================

TAMANHO_JANELA = 100
PASSO = 25



# =====================================================
# EXTRAÇÃO DE FEATURES
# =====================================================

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


        # diferença entre canais
        np.mean(c1-c2),
        np.std(c1-c2),


        # energia do sinal
        np.sqrt(np.mean(c1**2)),
        np.sqrt(np.mean(c2**2))

    ]


    return features



# =====================================================
# CRIAR JANELAS
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


    # remove ruídos extremos
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


    # converte volts para escala parecida com OpenBCI
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


    print(df.describe())


    return df




# =====================================================
# EXECUÇÃO
# =====================================================


normal=carregar_openbci()

piscada=carregar_piscada()



print("\nCriando janelas normal...")


X_normal=criar_janelas(normal)


print(
    "Normal:",
    len(X_normal)
)



print("\nCriando janelas piscada...")


X_piscada=criar_janelas(piscada)


print(
    "Piscada:",
    len(X_piscada)
)



# juntar dados


X=np.vstack(
    (
        X_normal,
        X_piscada
    )
)



y=np.array(
    ["normal"]*len(X_normal)
    +
    ["piscada"]*len(X_piscada)
)



print("\nTOTAL:")
print(len(X))



# =====================================================
# ESCALONAMENTO
# =====================================================


scaler=StandardScaler()


X=scaler.fit_transform(X)



# =====================================================
# TREINO
# =====================================================


X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)



modelo=SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    probability=True
)



modelo.fit(
    X_train,
    y_train
)



# =====================================================
# AVALIAÇÃO
# =====================================================


pred=modelo.predict(
    X_test
)



print("\n===================")
print("RESULTADOS")
print("===================")


print(
    "ACURÁCIA:",
    accuracy_score(
        y_test,
        pred
    )
)



print("\nMATRIZ:")

print(
    confusion_matrix(
        y_test,
        pred
    )
)



print(
    classification_report(
        y_test,
        pred
    )
)



# =====================================================
# VALIDAÇÃO
# =====================================================


scores=cross_val_score(
    modelo,
    X,
    y,
    cv=5
)


print("\nVALIDAÇÃO:")
print(scores)


print(
    "MÉDIA:",
    scores.mean()
)



# =====================================================
# SALVAR
# =====================================================


joblib.dump(
    modelo,
    "modelo_svm.pkl"
)


joblib.dump(
    scaler,
    "scaler.pkl"
)



print("\nMODELO SALVO COM SUCESSO!")