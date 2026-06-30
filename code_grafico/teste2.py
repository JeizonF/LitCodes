import pandas as pd
import matplotlib.pyplot as plt

# Caminho do arquivo
arquivo = r".\data\data\openBCI_raw_2018-07-02_15-46-30.txt"

# Lê o arquivo OpenBCI
dados = pd.read_csv(
    arquivo,
    comment='%',      # Ignora linhas iniciadas por %
    header=None,
    skipinitialspace=True
)

# Remove linhas vazias
dados = dados.dropna()

# Mostra algumas informações
print("Quantidade de amostras:", len(dados))
print("\nPrimeiras 5 linhas:")
print(dados.head())

# Colunas
x = dados[0]       # SampleIndex
canal1 = dados[1]  # EEG Canal 1
canal2 = dados[2]  # EEG Canal 2

# Cria o gráfico
plt.figure(figsize=(12, 6))

plt.plot(x, canal1, label="Canal 1")
plt.plot(x, canal2, label="Canal 2")

plt.title("Sinal EEG OpenBCI")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid(True)


plt.show()