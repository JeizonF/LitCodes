
import pandas as pd
import matplotlib.pyplot as plt


class ClassificadorSinais:

    def __init__(self, limite=50):
        self.limite = limite
        self.num = 0

    def proximo(self):
        self.num += 1
        return self.num

    def classificar_vertical(self, sinal):
        valor = sinal.mean()
        
        if valor >= self.limite:
            return "Cima"
        elif valor <= -self.limite:
            return "Baixo"
        return "Centro"

    def classificar_horizontal(self, sinal):
        valor = sinal.mean()

  
        if valor >= self.limite:
            return "Direita"
        elif valor <= -self.limite:
            return "Esquerda"
        return "Centro"


arquivo = r".\data\data\openBCI_raw_2018-07-02_15-46-30.txt"

dados = pd.read_csv(
    arquivo,
    comment="%",
    header=None,
    skipinitialspace=True
)

dados = dados.dropna()

x = dados[0]
canal1 = dados[1]
canal2 = dados[2]

janela = 1000
passo = 200


indice = 0
rodando = True

fig, ax = plt.subplots(figsize=(14, 6))

classificador = ClassificadorSinais()


def desenhar(i):
    
    ax.clear()

    x_janela = x.iloc[i:i + janela]
    c1 = canal1.iloc[i:i + janela]
    c2 = canal2.iloc[i:i + janela]

    vertical = classificador.classificar_vertical(c1)
    horizontal = classificador.classificar_horizontal(c2)
    proximo = classificador.proximo()
    print('')
    print(f"{proximo} Vertical: {vertical}")
    print(f"{proximo} Horizontal: {horizontal}")
    print('')

    ax.plot(x_janela, c1, label=f"Canal 1 ({vertical})")
    ax.plot(x_janela, c2, label=f"Canal 2 ({horizontal})")

    ax.set_title(
        f"EEG OpenBCI |{proximo} Vertical: {vertical} |{proximo} Horizontal: {horizontal}"
    )

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

while indice < len(dados):

    if rodando:
        desenhar(indice)
        indice += passo
        plt.pause(0.3)
    else:
        plt.pause(0.1)

plt.show()

