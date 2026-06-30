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

# Remove linhas inválidas
dados = dados.dropna()

# Colunas
x = dados[0]
canal1 = dados[1]
canal2 = dados[2]

# Cria UMA figura só
plt.figure(figsize=(18, 6))  # câmera maior

# Plota os dados
plt.plot(x, canal1, label="Canal 1")
plt.plot(x, canal2, label="Canal 2")

# ZOOM (IMPORTANTE: aqui sim funciona)
plt.xlim(0, 1000)      # aproxima no eixo X
# plt.ylim(-2000, 2000)  # opcional: zoom no eixo Y

plt.title("Sinal EEG OpenBCI (Zoom)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid(True)

plt.show()
