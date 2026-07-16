import pandas as pd
import numpy as np
import joblib
import os


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# CONFIG
# =====================================================

TAMANHO_JANELA = 100
PASSO = 25


PASTA_DADOS = "dados"


# =====================================================
# FEATURES
# =====================================================

def extrair_features(janela):

    c1 = janela["canal1"].values
    c2 = janela["canal2"].values


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


        # variação
        np.mean(np.abs(np.diff(c1))),
        np.mean(np.abs(np.diff(c2))),


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
# OPENBCI NORMAL
# =====================================================

def carregar_normal():

    print("\n================")
    print("Carregando normal")
    print("================")


    df=pd.read_csv(
        f"{PASTA_DADOS}/openBCI_raw_2018-07-02_15-46-30.txt",
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
        (abs(df.canal1)<500)&
        (abs(df.canal2)<500)
    ]


    print(df.describe())


    return df.reset_index(drop=True)





# =====================================================
# CSV DOS MOVIMENTOS
# =====================================================

def carregar_csv(nome):


    caminho=f"{PASTA_DADOS}/{nome}.csv"


    if not os.path.exists(caminho):

        print(
            "Arquivo não existe:",
            caminho
        )

        return None



    print("\n================")
    print("Carregando", nome)
    print("================")



    raw=pd.read_csv(
        caminho,
        header=None
    )



    # formato igual piscada
    df=raw.iloc[4:,:3]


    df.columns=[
        "tempo",
        "canal1",
        "canal2"
    ]



    for c in [
        "tempo",
        "canal1",
        "canal2"
    ]:

        df[c]=pd.to_numeric(
            df[c],
            errors="coerce"
        )


    df=df.dropna()



    # volts -> mV
    if abs(df.canal1).max()<10:

        df["canal1"]*=1000
        df["canal2"]*=1000



    df["amostra"]=range(
        len(df)
    )



    return df[
        [
            "amostra",
            "canal1",
            "canal2"
        ]
    ]





# =====================================================
# PISCADA
# =====================================================

def carregar_piscada():

    print("\n================")
    print("Carregando piscada")
    print("================")


    raw=pd.read_csv(
        f"{PASTA_DADOS}/piscada.csv",
        header=None
    )


    df=raw.iloc[4:,:3]


    df.columns=[
        "tempo",
        "canal1",
        "canal2"
    ]


    for c in [
        "tempo",
        "canal1",
        "canal2"
    ]:

        df[c]=pd.to_numeric(
            df[c],
            errors="coerce"
        )


    df=df.dropna()


    df["canal1"]*=1000
    df["canal2"]*=1000


    df["amostra"]=range(
        len(df)
    )


    print(
        df.describe()
    )


    return df[
        [
            "amostra",
            "canal1",
            "canal2"
        ]
    ]





# =====================================================
# CRIAR DATASET
# =====================================================


dados=[]
classes=[]



movimentos={

    "normal": carregar_normal(),

    "piscada": carregar_piscada(),

    "cima": None,

    "baixo": None,

    "esquerda": None,

    "direita": None

}



for nome,df in movimentos.items():


    if df is None:

        if nome not in [
            "normal",
            "piscada"
        ]:

            df=carregar_csv(nome)



    if df is None:

        print(
            "Ignorando",
            nome
        )

        continue



    print(
        "Criando janelas:",
        nome
    )


    X=criar_janelas(df)


    print(
        "Quantidade:",
        len(X)
    )


    dados.extend(X)


    classes.extend(
        [nome]*len(X)
    )





X=np.array(dados)

y=np.array(classes)



print("\n================")
print("DATASET FINAL")
print("================")

print(
    "Amostras:",
    len(X)
)

print(
    "Features:",
    X.shape[1]
)


print(
    pd.Series(y).value_counts()
)



# =====================================================
# SEGURANÇA
# =====================================================

if len(np.unique(y)) < 2:

    raise Exception(
        "Necessário pelo menos duas classes diferentes!"
    )





# =====================================================
# MODELO
# =====================================================


pipeline=Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        )
    )

])





X_train,X_test,y_train,y_test=train_test_split(

    X,
    y,

    test_size=0.25,

    random_state=42,

    stratify=y

)




pipeline.fit(
    X_train,
    y_train
)




# =====================================================
# RESULTADO
# =====================================================


pred=pipeline.predict(
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



scores=cross_val_score(
    pipeline,
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
    pipeline,
    "modelo_svm.pkl"
)



print(
"\nMODELO SALVO!"
)