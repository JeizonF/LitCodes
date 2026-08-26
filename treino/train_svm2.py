import os
import random
import json
import numpy as np
import joblib
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from config.utils import (
    ARQUIVOS,
    PASTA_DADOS,
    TAMANHO_JANELA,
    MODELO_PATH,
    carregar_arquivo,
    criar_janelas
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


SEED = 42
ARQUIVO_INFO_TREINO = "arquivos_treinamento.json"


def obter_arquivos():
    classes = {}

    for classe, arquivos in ARQUIVOS.items():

        if arquivos is None:
            continue

        if isinstance(arquivos, str):
            arquivos = [arquivos]

        validos = []

        for arquivo in arquivos:

            caminho = os.path.join(
                PASTA_DADOS,
                arquivo
            )

            if os.path.isfile(caminho):
                validos.append(arquivo)

        if validos:
            classes[classe] = validos

    return classes


def mostrar_arquivos(classes):

    print("\n================================")
    print("ARQUIVOS DISPONÍVEIS")
    print("================================")

    for classe, arquivos in classes.items():

        print(
            f"\n{classe}: {len(arquivos)} arquivo(s)"
        )

        for arquivo in arquivos:
            print(f"  - {arquivo}")


def selecionar_modo_treino():

    print("\n================================")
    print("MODO DE TREINAMENTO")
    print("================================")

    print("\n1 - Treinar com TODOS os arquivos")
    print("2 - Escolher arquivos para o treino")

    while True:

        opcao = input("\nEscolha uma opção: ").strip()

        if opcao in ["1", "2"]:
            return opcao

        print("Opção inválida.")


def selecionar_treino(classes):

    treino = {}

    print("\n================================")
    print("SELEÇÃO DOS ARQUIVOS DE TREINO")
    print("================================")

    print(
        "\nDigite a quantidade de arquivos "
        "que deseja usar no TREINO para cada classe."
    )

    print(
        "\nCada arquivo será usado inteiro."
    )

    print(
        "Nenhum arquivo será dividido entre treino e teste."
    )

    for classe, arquivos in classes.items():

        quantidade = len(arquivos)

        print("\n--------------------------------")
        print(f"Classe: {classe}")
        print(f"Arquivos disponíveis: {quantidade}")

        if quantidade == 1:

            print(
                "\nEsta classe possui apenas 1 arquivo."
            )

            print(
                "Esse arquivo será usado no TREINO."
            )

            treino[classe] = arquivos.copy()

            continue

        while True:

            try:

                qtd = int(
                    input(
                        "\nQuantidade para TREINO: "
                    )
                )

            except ValueError:

                print(
                    "Digite um número válido."
                )

                continue

            if qtd < 1:

                print(
                    "Escolha pelo menos 1 arquivo."
                )

                continue

            if qtd >= quantidade:

                print(
                    f"Você possui {quantidade} arquivos."
                )

                print(
                    "É necessário deixar pelo menos "
                    "1 arquivo fora do treino."
                )

                continue

            break

        arquivos_embaralhados = arquivos.copy()

        random.shuffle(
            arquivos_embaralhados
        )

        treino[classe] = (
            arquivos_embaralhados[:qtd]
        )

    return treino


def selecionar_todos(classes):

    treino = {}

    for classe, arquivos in classes.items():

        treino[classe] = arquivos.copy()

    return treino


def mostrar_selecao(treino):

    print("\n================================")
    print("ARQUIVOS SELECIONADOS PARA TREINO")
    print("================================")

    classes_treino = []

    for classe, arquivos in treino.items():

        if not arquivos:
            continue

        classes_treino.append(classe)

        print(f"\n{classe}:")

        for arquivo in arquivos:
            print(f"  - {arquivo}")

    print("\n================================")
    print("CLASSES DO TREINO")
    print("================================")

    for classe in classes_treino:
        print(f"- {classe}")


def criar_dataset(treino):

    X = []
    y = []

    print("\n================================")
    print("CARREGANDO DADOS DE TREINO")
    print("================================")

    for classe, arquivos in treino.items():

        for arquivo in arquivos:

            caminho = os.path.join(
                PASTA_DADOS,
                arquivo
            )

            print("\n--------------------------------")
            print(f"Classe: {classe}")
            print(f"Arquivo: {arquivo}")

            df = carregar_arquivo(
                caminho
            )

            if df is None:

                print(
                    "Arquivo ignorado."
                )

                continue

            if len(df) < TAMANHO_JANELA:

                print(
                    f"Arquivo possui menos de "
                    f"{TAMANHO_JANELA} amostras."
                )

                continue

            janelas = criar_janelas(
                df
            )

            if len(janelas) == 0:

                print(
                    "Nenhuma janela criada."
                )

                continue

            print(
                f"Janelas criadas: {len(janelas)}"
            )

            for janela in janelas:

                X.append(janela)
                y.append(classe)

    return (
        np.array(X),
        np.array(y)
    )


def salvar_informacoes_treino(treino, modo):

    arquivos_treinados = []

    for classe, arquivos in treino.items():

        for arquivo in arquivos:

            arquivos_treinados.append({
                "arquivo": arquivo,
                "classe": classe
            })

    informacoes = {
        "modo_treinamento": modo,
        "arquivos_treinamento": arquivos_treinados
    }

    with open(
        ARQUIVO_INFO_TREINO,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            informacoes,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nInformações salvas em: "
        f"{ARQUIVO_INFO_TREINO}"
    )


def criar_modelo():

    modelo = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
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

    return modelo


def main():

    print("\n================================")
    print("TREINAMENTO DO SVM")
    print("================================")

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    classes = obter_arquivos()

    if not classes:

        raise Exception(
            "Nenhum arquivo encontrado."
        )

    mostrar_arquivos(
        classes
    )

    modo = selecionar_modo_treino()

    if modo == "1":

        print("\n================================")
        print("TREINAMENTO COM TODOS OS ARQUIVOS")
        print("================================")

        treino = selecionar_todos(
            classes
        )

        nome_modo = "todos"

    else:

        print("\n================================")
        print("TREINAMENTO COM SELEÇÃO DE ARQUIVOS")
        print("================================")

        treino = selecionar_treino(
            classes
        )

        nome_modo = "selecao"

    mostrar_selecao(
        treino
    )

    X, y = criar_dataset(
        treino
    )

    if len(X) == 0:

        raise Exception(
            "Nenhuma amostra foi criada."
        )

    classes_encontradas = np.unique(
        y
    )

    print("\n================================")
    print("DATASET")
    print("================================")

    print(
        f"Total de janelas: {len(X)}"
    )

    print(
        f"Quantidade de features: {X.shape[1]}"
    )

    print("\nClasses:")

    for classe in classes_encontradas:

        quantidade = np.sum(
            y == classe
        )

        print(
            f"- {classe}: {quantidade} janelas"
        )

    if len(classes_encontradas) < 2:

        raise Exception(
            "\nO treinamento precisa possuir "
            "pelo menos duas classes."
        )

    print("\n================================")
    print("CRIANDO MODELO")
    print("================================")

    modelo = criar_modelo()

    print("\n================================")
    print("TREINANDO")
    print("================================")

    modelo.fit(
        X,
        y
    )

    print(
        "\nTreinamento concluído."
    )

    print("\n================================")
    print("SALVANDO MODELO")
    print("================================")

    joblib.dump(
        modelo,
        MODELO_PATH
    )

    print(
        f"\nModelo salvo em:"
    )

    print(
        MODELO_PATH
    )

    salvar_informacoes_treino(
        treino,
        nome_modo
    )

    print("\n================================")
    print("TREINAMENTO FINALIZADO")
    print("================================")


if __name__ == "__main__":
    main()