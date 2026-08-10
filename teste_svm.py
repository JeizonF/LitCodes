import numpy as np  # usado para fazer cálculos matemáticos
import joblib  # usado para carregar o modelo salvo

from utils import (
    criar_janelas,      # cria as janelas do sinal
    carregar_movimento  # carrega o arquivo do movimento
)

# informa que o modelo está sendo carregado
print("\ncarregando modelo...")

# carrega o modelo svm salvo
modelo = joblib.load(
    "modelo_svm.pkl"
)

# mostra o menu de opções
print("""
digite:

1 - normal
2 - piscada
3 - cima
4 - baixo
5 - esquerda
6 - direita
""")

# lê a opção escolhida pelo usuário
opcao = input("> ").strip()

# relaciona cada número com um movimento
mapa = {
    "1": "normal",
    "2": "piscada",
    "3": "cima",
    "4": "baixo",
    "5": "esquerda",
    "6": "direita"
}

# verifica se a opção é válida
if opcao not in mapa:

    print("opção inválida")
    exit()

# obtém o nome do movimento escolhido
classe = mapa[opcao]

# carrega os dados do movimento escolhido
dados = carregar_movimento(
    classe
)

# verifica se o arquivo existe
if dados is None:

    print(
        "\narquivo desse movimento ainda não existe"
    )

    exit()

# cria as janelas do sinal
print("\ncriando janelas...")

# transforma o sinal em várias janelas
X = criar_janelas(
    dados
)

# mostra quantas janelas foram criadas
print(
    "quantidade de janelas:",
    len(X)
)

# mostra quantas features cada janela possui
print(
    "quantidade de features:",
    X.shape[1]
)

# faz a previsão para todas as janelas
pred = modelo.predict(
    X
)

# mostra o resultado das previsões
print("\n====================")
print("resultado")
print("====================")

# cria uma lista para guardar as previsões
resultados = []

# percorre todas as previsões
for i, p in enumerate(pred):

    # adiciona a previsão na lista
    resultados.append(p)

    # mostra a previsão da janela
    print(
        f"{i} -> {p}"
    )

# cria um resumo das previsões

# conta quantas vezes cada classe apareceu
valores, quantidades = np.unique(
    pred,
    return_counts=True
)

print("\nresumo:")

# mostra a quantidade de cada classe
for v, q in zip(
    valores,
    quantidades
):

    print(
        f"{v}: {q}"
    )

# encontra a classe com maior quantidade
indice = np.argmax(
    quantidades
)

# define a classe final
classe_final = valores[indice]

# calcula a porcentagem da classe vencedora
porcentagem = (
    quantidades[indice]
    /
    len(pred)
) * 100

# mostra a decisão final
print("\n====================")
print("decisão final")
print("====================")

# mostra a classe prevista e sua porcentagem
print(
    f"{classe_final} -> {porcentagem:.2f}%"
)