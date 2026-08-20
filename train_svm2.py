from utils import *  # importa configurações e funções do projeto

import os  # usado para trabalhar com arquivos e pastas
import joblib  # usado para salvar o modelo treinado
import pandas as pd  # usado para trabalhar com tabelas
import numpy as np  # usado para fazer cálculos matemáticos
from sklearn.calibration import CalibratedClassifierCV

from sklearn.pipeline import Pipeline  # junta as etapas do treinamento
from sklearn.preprocessing import StandardScaler  # normaliza os dados
from sklearn.svm import SVC  # cria o classificador svm

from sklearn.model_selection import (
    train_test_split,  # separa treino e teste
    cross_val_score  # faz validação cruzada
)

from sklearn.metrics import (
    accuracy_score,  # calcula a acurácia
    confusion_matrix,  # cria a matriz de confusão
    classification_report  # mostra as métricas
)


# =====================================================
# FUNÇÃO PARA CARREGAR UMA CLASSE
# =====================================================

# carrega todos os arquivos de uma classe
# criando as janelas separadamente para cada arquivo

def carregar_classe(nome):

    # lista que guardará todas as features da classe

    dados_classe = []


    # pega os arquivos configurados no utils

    arquivos = ARQUIVOS.get(
        nome
    )


    # verifica se existem arquivos configurados

    if arquivos is None:

        print(
            f"\nnenhum arquivo configurado para '{nome}'"
        )

        return np.array([])


    # se existir apenas um arquivo,
    # transforma em lista

    if isinstance(
        arquivos,
        str
    ):

        arquivos = [
            arquivos
        ]


    # percorre todos os arquivos da classe

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
                f"\narquivo não encontrado: {arquivo}"
            )

            continue


        print("\n==============================")
        print(
            f"classe: {nome}"
        )
        print(
            f"arquivo: {arquivo}"
        )
        print("==============================")


        # carrega somente este arquivo

        df = carregar_arquivo(
            caminho
        )


        # verifica se o arquivo foi carregado

        if df is None:

            print(
                "não foi possível carregar"
            )

            continue


        # verifica se existem dados suficientes

        if len(df) < TAMANHO_JANELA:

            print(
                "arquivo pequeno demais "
                "para criar uma janela"
            )

            continue


        # cria as janelas SOMENTE deste arquivo

        X_arquivo = criar_janelas(
            df
        )


        # verifica se foram criadas janelas

        if len(X_arquivo) == 0:

            print(
                "nenhuma janela criada"
            )

            continue


        # adiciona as features deste arquivo

        dados_classe.extend(
            X_arquivo
        )


        # mostra a quantidade de janelas

        print(
            "janelas criadas:",
            len(X_arquivo)
        )


    # transforma em array numpy

    return np.array(
        dados_classe
    )


# =====================================================
# CRIAÇÃO DO DATASET
# =====================================================

# lista que armazenará todas as features

dados = []


# lista que armazenará as classes

classes = []


# movimentos que o sistema tentará treinar

movimentos = [
    "normal",
    "piscada",
    "cima",
    "baixo",
    "esquerda",
    "direita"
]


# percorre todos os movimentos

for nome in movimentos:

    print("\n================================")
    print(
        "carregando classe:",
        nome
    )
    print("================================")


    # carrega todos os arquivos daquela classe

    X_classe = carregar_classe(
        nome
    )


    # verifica se existem janelas

    if len(X_classe) == 0:

        print(
            f"\nclasse '{nome}' ignorada"
        )

        continue


    # adiciona as features ao dataset

    dados.extend(
        X_classe
    )


    # adiciona o nome da classe
    # para cada janela criada

    classes.extend(
        [nome] * len(X_classe)
    )


    # mostra o total da classe

    print(
        f"\ntotal de janelas "
        f"da classe '{nome}':",
        len(X_classe)
    )


# =====================================================
# TRANSFORMA OS DADOS EM ARRAYS
# =====================================================

X = np.array(
    dados
)


y = np.array(
    classes
)


# =====================================================
# MOSTRA O DATASET FINAL
# =====================================================

print("\n================================")
print("dataset final")
print("================================")


# mostra o total de janelas

print(
    "total de amostras:",
    len(X)
)


# verifica se existem dados

if len(X) == 0:

    raise Exception(
        "nenhuma amostra foi encontrada"
    )


# mostra a quantidade de features

print(
    "quantidade de features:",
    X.shape[1]
)


# mostra quantas janelas existem
# para cada classe

print("\namostras por classe:")

print(
    pd.Series(
        y
    ).value_counts()
)


# verifica quantas classes foram encontradas

if len(
    np.unique(y)
) < 2:

    raise Exception(
        "é necessário possuir "
        "pelo menos duas classes"
    )


# =====================================================
# DIVISÃO ENTRE TREINO E TESTE
# =====================================================

# separa os dados

X_train, X_test, y_train, y_test = train_test_split(

    # features

    X,


    # classes

    y,


    # porcentagem usada para teste

    test_size=0.25,


    # mantém o mesmo resultado
    # quando os dados são iguais

    random_state=42,


    # mantém a proporção das classes

    stratify=y
)


# =====================================================
# CRIAÇÃO DO MODELO
# =====================================================

# cria um pipeline

pipeline = Pipeline([


    # normaliza as features

    (
        "scaler",

        StandardScaler()
    ),


    # cria o svm

    (
        "svm",
        CalibratedClassifierCV(

            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced"
            ),

            ensemble=False
        )
    )
])


# =====================================================
# TREINAMENTO
# =====================================================

print("\n================================")
print("treinando modelo")
print("================================")


# treina usando apenas os dados de treino

pipeline.fit(
    X_train,
    y_train
)


# =====================================================
# TESTE
# =====================================================

# faz previsões nos dados separados para teste

pred = pipeline.predict(
    X_test
)


# calcula a acurácia

acuracia = accuracy_score(
    y_test,
    pred
)


print("\n================================")
print("resultado")
print("================================")


print(
    f"acurácia: {acuracia * 100:.2f}%"
)


# =====================================================
# MATRIZ DE CONFUSÃO
# =====================================================

print("\nmatriz de confusão:")


# mostra a ordem das classes

print(
    "classes:",
    pipeline.classes_
)


# mostra a matriz

print(
    confusion_matrix(
        y_test,
        pred,
        labels=pipeline.classes_
    )
)


# =====================================================
# RELATÓRIO DETALHADO
# =====================================================

print("\nrelatório:")


print(
    classification_report(
        y_test,
        pred
    )
)


# =====================================================
# VALIDAÇÃO CRUZADA
# =====================================================

print("\n================================")
print("validação cruzada")
print("================================")


# faz validação cruzada

scores = cross_val_score(

    # modelo

    pipeline,


    # dados

    X,


    # classes

    y,


    # quantidade de divisões

    cv=5
)


# mostra os resultados individuais

print(
    "resultados:"
)


print(
    scores
)


# mostra a média

print(
    f"média: "
    f"{scores.mean() * 100:.2f}%"
)


# mostra o desvio

print(
    f"desvio: "
    f"{scores.std() * 100:.2f}%"
)


# =====================================================
# TREINA O MODELO FINAL
# =====================================================

print("\n================================")
print("treinando modelo final")
print("================================")


# agora treina usando TODOS os dados

pipeline.fit(
    X,
    y
)


# =====================================================
# SALVA O MODELO
# =====================================================

joblib.dump(

    # modelo

    pipeline,


    # arquivo onde será salvo

    "modelo_svm.pkl"
)


print(
    "\nmodelo salvo com sucesso!"
)