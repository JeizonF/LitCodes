import matplotlib.pyplot as plt

# Dados nos eixos X e Y
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Cria o gráfico
plt.plot(x, y, marker='o')

# Adiciona títulos e rótulos
plt.title("Meu Primeiro Gráfico")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")

# Exibe o gráfico
plt.show()
