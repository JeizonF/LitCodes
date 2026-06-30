import pandas as pd
import matplotlib.pyplot as plt

# Caminho do arquivo
arquivo = r".\data\data\openBCI_raw_2018-07-02_15-46-30.txt"

dados = pd.read_csv(
    arquivo,
    comment='%',
    header=None,
    skipinitialspace=True
)

dados = dados.dropna()

x = dados[0]
canal1 = dados[1]
canal2 = dados[2]

# Configurações
janela = 1000
passo = 200

indice = 0
rodando = True

fig, ax = plt.subplots(figsize=(14, 6))


def desenhar(i):
    ax.clear()

    x_janela = x.iloc[i:i+janela]
    c1 = canal1.iloc[i:i+janela]
    c2 = canal2.iloc[i:i+janela]

    ax.plot(x_janela, c1, label="Canal 1")
    ax.plot(x_janela, c2, label="Canal 2")

    ax.set_title(f"EEG OpenBCI | Amostras {i} até {i+janela}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend()
    ax.grid(True)

    plt.draw()


def on_key(event):
    global indice, rodando

    if event.key == "right":
        indice += passo
        desenhar(indice)

    elif event.key == "left":
        indice = max(0, indice - passo)
        desenhar(indice)

    elif event.key == " ":
        rodando = not rodando


fig.canvas.mpl_connect("key_press_event", on_key)

# LOOP PRINCIPAL (AUTOMÁTICO)
while indice < len(dados):

    if rodando:
        desenhar(indice)
        indice += passo
        plt.pause(0.3)
    else:
        plt.pause(0.1)

plt.show()