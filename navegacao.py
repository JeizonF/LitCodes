from navegador_letras import *


mostrar_teclado()



while True:


    comando=input(
        "\nComando: "
    )


    if comando=="sair":

        break



    if comando=="piscada":

        selecionar()



    elif comando in [

        "cima",
        "baixo",
        "esquerda",
        "direita"

    ]:


        intensidade=float(
            input(
                "Intensidade: "
            )
        )


        mover(
            comando,
            intensidade
        )



    mostrar_teclado()