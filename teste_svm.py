import numpy as np
import joblib


from utils import (
    criar_janelas,
    carregar_movimento
)



print("\nCarregando modelo...")


modelo = joblib.load(
    "modelo_svm.pkl"
)



print("""
Digite:

1 - normal
2 - piscada
3 - cima
4 - baixo
5 - esquerda
6 - direita
""")


opcao=input("> ").strip()



mapa={

    "1":"normal",
    "2":"piscada",
    "3":"cima",
    "4":"baixo",
    "5":"esquerda",
    "6":"direita"

}



if opcao not in mapa:

    print("Opção inválida")
    exit()



classe=mapa[opcao]



dados=carregar_movimento(
    classe
)


if dados is None:

    print(
        "\nArquivo desse movimento ainda não existe!"
    )

    exit()



# =====================================
# CRIAR JANELAS
# =====================================


print("\nCriando janelas...")


X=criar_janelas(
    dados
)


print(
    "Quantidade de janelas:",
    len(X)
)


print(
    "Quantidade de features:",
    X.shape[1]
)



pred=modelo.predict(
    X
)



print("\n====================")
print("RESULTADO")
print("====================")



for i,p in enumerate(pred):

    print(
        f"{i} -> {p}"
    )


# =====================================
# PREDIÇÃO
# =====================================


pred = modelo.predict(
    X
)



print("\n====================")
print("RESULTADO")
print("====================")



resultados=[]


for i, p in enumerate(pred):

    resultados.append(p)

    print(
        f"{i} -> {p}"
    )



# =====================================
# RESUMO
# =====================================
valores,quantidades=np.unique(
    pred,
    return_counts=True
)



print("\nResumo:")


for v,q in zip(
    valores,
    quantidades
):

    print(
        f"{v}: {q}"
    )



indice=np.argmax(
    quantidades
)


classe_final=valores[indice]


porcentagem=(

    quantidades[indice]
    /
    len(pred)

)*100



print("\n====================")
print("DECISÃO FINAL")
print("====================")


print(
    f"{classe_final} -> {porcentagem:.2f}%"
)