import tkinter as tk
import joblib
import pyautogui

from config.utils import (
    MOVIMENTO_MOUSE,
    TAMANHO_JANELA,
    PASSO,
    MODELO_PATH,
    ARQUIVO_TESTE,
    carregar_arquivo,
    extrair_features
)


modelo = joblib.load(
    MODELO_PATH
)

rodando = False
indice = 0
dados = None


def mover_mouse(movimento):

    if movimento == "esquerda":

        pyautogui.moveRel(
            -MOVIMENTO_MOUSE,
            0
        )

    elif movimento == "direita":

        pyautogui.moveRel(
            MOVIMENTO_MOUSE,
            0
        )

    elif movimento == "cima":

        pyautogui.moveRel(
            0,
            -MOVIMENTO_MOUSE
        )

    elif movimento == "baixo":

        pyautogui.moveRel(
            0,
            MOVIMENTO_MOUSE
        )


def carregar_dados():

    global dados

    dados = carregar_arquivo(
        ARQUIVO_TESTE
    )

    if dados is None:

        label_status.config(
            text="Erro ao carregar arquivo"
        )

        return False

    return True

def processar_janela():

    global indice
    global rodando

    if not rodando:
        return

    if indice + TAMANHO_JANELA > len(dados):

        rodando = False

        label_status.config(
            text="Teste finalizado"
        )

        return

    janela = dados.iloc[
        indice:
        indice + TAMANHO_JANELA
    ]

    features = extrair_features(
        janela
    )

    previsao = modelo.predict(
        [features]
    )[0]

    print(
        f"{indice} -> {previsao}"
    )

    label_movimento.config(
        text=f"Movimento: {previsao.upper()}"
    )

    if previsao in [
        "esquerda",
        "direita",
        "cima",
        "baixo"
    ]:

        mover_mouse(
            previsao
        )

    indice += PASSO

    janela_gui.after(
        100,
        processar_janela
    )

def iniciar():

    global rodando
    global indice

    if dados is None:

        if not carregar_dados():

            return

    indice = 0
    rodando = True

    label_status.config(
        text="Teste em execução"
    )

    processar_janela()


def parar():

    global rodando

    rodando = False

    label_status.config(
        text="Teste parado"
    )


janela_gui = tk.Tk()

janela_gui.title(
    "Teste do Modelo + Mouse"
)

janela_gui.geometry(
    "450x350"
)

janela_gui.resizable(
    False,
    False
)


titulo = tk.Label(
    janela_gui,
    text="TESTE DO MODELO",
    font=("Arial", 20, "bold")
)

titulo.pack(
    pady=20
)


label_status = tk.Label(
    janela_gui,
    text="Aguardando",
    font=("Arial", 13)
)

label_status.pack(
    pady=5
)


label_movimento = tk.Label(
    janela_gui,
    text="Movimento: nenhum",
    font=("Arial", 16)
)

label_movimento.pack(
    pady=10
)


label_janela = tk.Label(
    janela_gui,
    text="Janela: 0",
    font=("Arial", 11)
)

label_janela.pack(
    pady=5
)


frame = tk.Frame(
    janela_gui
)

frame.pack(
    pady=20
)


botao_iniciar = tk.Button(
    frame,
    text="INICIAR",
    width=15,
    height=2,
    command=iniciar
)

botao_iniciar.grid(
    row=0,
    column=0,
    padx=10
)


botao_parar = tk.Button(
    frame,
    text="PARAR",
    width=15,
    height=2,
    command=parar
)

botao_parar.grid(
    row=0,
    column=1,
    padx=10
)


label_distancia = tk.Label(
    janela_gui,
    text=f"Movimento: {MOVIMENTO_MOUSE} pixels"
)

label_distancia.pack(
    pady=10
)


janela_gui.mainloop()