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

# ----------------------------
# FILTRO DE FAIXA (AQUI)
# ----------------------------

inicio = 100   # começo da faixa
fim = 25094      # fim da faixa

dados = dados.iloc[inicio:fim]

# ----------------------------

# Colunas
x = dados[0]
canal1 = dados[1]
canal2 = dados[2]

# Gráfico
plt.figure(figsize=(12, 6))

plt.plot(x, canal1, label="Canal 1")
plt.plot(x, canal2, label="Canal 2")

plt.title(f"Sinal EEG OpenBCI ({inicio} até {fim})")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid(True)

plt.show()