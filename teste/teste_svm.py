import os  # usado para trabalhar com caminhos
import numpy as np  # usado para cálculos matemáticos
import joblib  # usado para carregar o modelo salvo
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from config.utils import (
    PASTA_DADOS,           # pasta onde estão os arquivos
    listar_arquivos_dados, # procura os arquivos disponíveis
    carregar_arquivo,      # carrega e prepara o arquivo
    criar_janelas    # cria as janelas do sinal
)


# =====================================================
# CARREGAR MODELO
# =====================================================

# informa que o modelo está sendo carregado

print("\ncarregando modelo...")

# carrega o modelo SVM

try:

    modelo = joblib.load(
        "modelo_svm.pkl"
    )

except Exception as erro:

    print(
        f"\nerro ao carregar o modelo: {erro}"
    )

    exit()


# informa que o modelo foi carregado

print("modelo carregado!")


# =====================================================
# PROCURAR ARQUIVOS
# =====================================================

# procura os arquivos dentro da pasta configurada

arquivos = listar_arquivos_dados()


# verifica se encontrou arquivos

if len(arquivos) == 0:

    print(
        "\nnenhum arquivo .csv ou .txt foi encontrado."
    )

    exit()


# =====================================================
# MENU DE ARQUIVOS
# =====================================================

# mostra o título

print("\n==============================")
print("ARQUIVOS DISPONÍVEIS")
print("==============================")


# mostra todos os arquivos encontrados

for i, nome in enumerate(
    arquivos,
    start=1
):

    # mostra número e nome do arquivo

    print(
        f"{i} - {nome}"
    )


# =====================================================
# ESCOLHER ARQUIVO
# =====================================================

# pede ao usuário para escolher

print(
    "\ndigite o número do arquivo que deseja testar:"
)

opcao = input(
    "> "
).strip()


# verifica se foi digitado um número

if not opcao.isdigit():

    print(
        "\nopção inválida."
    )

    exit()


# transforma o número em índice

indice = int(opcao) - 1


# verifica se o índice existe

if indice < 0 or indice >= len(arquivos):

    print(
        "\nopção inválida."
    )

    exit()


# pega o nome do arquivo

nome_arquivo = arquivos[indice]


# monta o caminho completo

caminho = os.path.join(
    PASTA_DADOS,
    nome_arquivo
)


# =====================================================
# CARREGAR ARQUIVO
# =====================================================

# informa qual arquivo foi escolhido

print(
    f"\narquivo escolhido: {nome_arquivo}"
)


# carrega e prepara os dados

dados = carregar_arquivo(
    caminho
)


# verifica se os dados foram carregados

if dados is None:

    print(
        "\nnão foi possível carregar o arquivo."
    )

    exit()


# =====================================================
# CRIAR JANELAS
# =====================================================

# informa que as janelas serão criadas

print(
    "\ncriando janelas..."
)


# transforma o sinal em janelas

X = criar_janelas(
    dados
)


# verifica se existem janelas

if len(X) == 0:

    print(
        "\no arquivo não possui dados suficientes para criar janelas."
    )

    exit()


# mostra a quantidade de janelas

print(
    "quantidade de janelas:",
    len(X)
)


# verifica se X possui features

if X.ndim != 2:

    print(
        "\nerro: formato das features inválido."
    )

    exit()


# mostra a quantidade de features

print(
    "quantidade de features:",
    X.shape[1]
)


# =====================================================
# VERIFICAR MODELO
# =====================================================

# verifica quantas features o modelo espera

if hasattr(
    modelo,
    "n_features_in_"
):

    features_modelo = modelo.n_features_in_

    # compara com as features produzidas

    if X.shape[1] != features_modelo:

        print(
            "\nerro: quantidade de features diferente."
        )

        print(
            f"features do teste: {X.shape[1]}"
        )

        print(
            f"features esperadas pelo modelo: {features_modelo}"
        )

        exit()


# =====================================================
# CLASSIFICAÇÃO
# =====================================================

# informa que a classificação começou

print(
    "\nclassificando..."
)


# faz a previsão

try:

    pred = modelo.predict(
        X
    )

except Exception as erro:

    print(
        f"\nerro durante a classificação: {erro}"
    )

    exit()


# =====================================================
# RESULTADO
# =====================================================

# mostra o título

print("\n====================")
print("RESULTADO")
print("====================")


# percorre todas as previsões

for i, classe in enumerate(
    pred
):

    # mostra a classe prevista para a janela

    print(
        f"{i} -> {classe}"
    )


# =====================================================
# RESUMO
# =====================================================

# conta quantas vezes cada classe apareceu

valores, quantidades = np.unique(
    pred,
    return_counts=True
)


# mostra o resumo

print("\nresumo:")


# percorre as classes

for valor, quantidade in zip(
    valores,
    quantidades
):

    # mostra a quantidade

    print(
        f"{valor}: {quantidade}"
    )


# =====================================================
# DECISÃO FINAL
# =====================================================

# encontra a classe mais frequente

indice_vencedor = np.argmax(
    quantidades
)


# pega o nome da classe vencedora

classe_final = valores[
    indice_vencedor
]


# pega a quantidade da classe vencedora

quantidade_final = quantidades[
    indice_vencedor
]


# calcula a porcentagem

porcentagem = (
    quantidade_final
    /
    len(pred)
) * 100


# mostra a decisão final

print("\n====================")
print("DECISÃO FINAL")
print("====================")


print(
    f"{classe_final} -> {porcentagem:.2f}%"
)


# =====================================================
# INFORMAÇÃO SOBRE O MODELO
# =====================================================

# verifica se o modelo possui classes conhecidas

if hasattr(
    modelo,
    "classes_"
):

    # mostra as classes utilizadas pelo SVM

    print("\nclasses conhecidas pelo modelo:")

    for classe in modelo.classes_:

        print(
            f"- {classe}"
        )


# =====================================================
# FINAL
# =====================================================

print("\n====================")
print("TESTE FINALIZADO")
print("====================")