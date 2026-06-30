import pandas as pd
import matplotlib.pyplot as plt

# Caminho do arquivo
arquivo = r".\data\data\openBCI_raw_2018-07-02_15-46-30.txt"

# Lê o arquivo OpenBCI
dados = pd.read_csv(
    arquivo,
    comment='%',
    header=None,
    skipinitialspace=True
)

dados = dados.dropna()

# Colunas
x = dados[0]
canal1 = dados[1]
canal2 = dados[2]

# Tamanho da janela (1000 amostras)
janela = 100

plt.figure(figsize=(14, 6))

# loop para "animar"
for i in range(0, len(dados), janela):

    plt.clf()  # limpa o gráfico

    x_janela = x.iloc[i:i+janela]
    c1_janela = canal1.iloc[i:i+janela]
    c2_janela = canal2.iloc[i:i+janela]

    plt.plot(x_janela, c1_janela, label="Canal 1")
    plt.plot(x_janela, c2_janela, label="Canal 2")

    plt.title(f"EEG OpenBCI - Amostras {i} até {i+janela}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.grid(True)

    plt.pause(1.00)  # velocidade da animação

plt.show()