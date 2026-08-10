from utils import *  # importa configurações e funções do projeto

import joblib  # usado para salvar o modelo treinado
import pandas as pd  # trabalha com tabelas
import numpy as np  # faz cálculos matemáticos

from sklearn.pipeline import Pipeline  # junta todas as etapas do treinamento
from sklearn.preprocessing import StandardScaler  # normaliza os dados
from sklearn.svm import SVC  # cria o classificador svm

from sklearn.model_selection import (
    train_test_split,  # separa treino e teste
    cross_val_score    # faz validação cruzada
)

from sklearn.metrics import (
    accuracy_score,        # calcula a acurácia
    confusion_matrix,      # cria a matriz de confusão
    classification_report  # mostra as métricas de cada classe
)


# cria o dataset usado no treinamento

# lista que armazenará todas as features
dados = []

# lista que armazenará o nome de cada classe
classes = []

# lista com todos os movimentos
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

    print("\n================")
    print("carregando", nome)
    print("================")

    # carrega o movimento atual
    df = carregar_movimento(nome)

    # verifica se o arquivo foi encontrado
    if df is None:

        print("arquivo não encontrado")
        continue

    # cria as janelas do movimento
    X = criar_janelas(df)

    # adiciona as features ao dataset
    dados.extend(X)

    # adiciona o nome da classe para cada janela
    classes.extend(
        [nome] * len(X)
    )

    # mostra quantas janelas foram criadas
    print("janelas:", len(X))

# transforma as listas em arrays do numpy
X = np.array(dados)
y = np.array(classes)

print("\n================")
print("dataset final")
print("================")

# mostra o total de amostras
print(
    "amostras:",
    len(X)
)

# mostra a quantidade de features
print(
    "features:",
    X.shape[1]
)

# mostra quantas amostras existem em cada classe
print(
    pd.Series(y).value_counts()
)

# verifica se existem pelo menos duas classes
if len(np.unique(y)) < 2:

    raise Exception(
        "necessário pelo menos duas classes diferentes"
    )


# cria o pipeline de treinamento
pipeline = Pipeline([

    # primeira etapa do pipeline
    (
        "scaler",

        # normaliza os dados antes do treinamento
        StandardScaler()
    ),

    # segunda etapa do pipeline
    (
        "svm",

        # cria o classificador svm
        SVC(

            # usa um kernel não linear
            kernel="rbf",

            # controla o quanto o modelo tenta evitar erros
            C=10,

            # calcula automaticamente o valor de gamma
            gamma="scale",

            # equilibra classes com quantidades diferentes
            class_weight="balanced"
        )
    )

])


# divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(

    # features
    X,

    # classes
    y,

    # separa 25 por cento para teste
    test_size=0.25,

    # mantém o mesmo resultado em cada execução
    random_state=42,

    # mantém a proporção das classes
    stratify=y

)


# treina o modelo usando os dados de treino
pipeline.fit(
    X_train,
    y_train
)


# faz previsões usando os dados de teste
pred = pipeline.predict(
    X_test
)

# mostra os resultados do modelo
print("\n===================")
print("resultados")
print("===================")

# mostra a porcentagem de acertos
print(
    "acurácia:",
    accuracy_score(
        y_test,
        pred
    )
)

# mostra a matriz de confusão
print("\nmatriz:")

print(
    confusion_matrix(
        y_test,
        pred
    )
)

# mostra as métricas de cada classe
print(
    classification_report(
        y_test,
        pred
    )
)


# faz uma validação cruzada com cinco divisões
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5
)

# mostra os resultados da validação
print("\nvalidação:")
print(scores)

# mostra a média da validação
print(
    "média:",
    scores.mean()
)



# salva o modelo treinado em um arquivo
joblib.dump(
    pipeline,
    "modelo_svm.pkl"
)

# informa que o modelo foi salvo
print(
    "\nmodelo salvo"
)