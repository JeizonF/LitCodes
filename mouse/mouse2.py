import os
import csv
import json
import tkinter as tk
import numpy as np
import joblib
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from config.utils import (
    ARQUIVOS,
    PASTA_DADOS,
    MODELO_PATH,
    carregar_arquivo,
    criar_janelas
)

from sklearn.metrics import confusion_matrix


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_RESULTADOS = "resultados"
ARQUIVO_INFO_TREINO = "arquivos_treinamento.json"

LARGURA_JANELA = 800
ALTURA_JANELA = 600

VELOCIDADE_MOUSE = 15
TEMPO_JANELA_MS = 120


# ============================================================
# MOUSE VIRTUAL
# ============================================================

class MouseVirtual:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title("Mouse Virtual - SVM")

        self.root.geometry(
            f"{LARGURA_JANELA}x{ALTURA_JANELA}"
        )

        self.root.resizable(False, False)

        # ----------------------------------------------------
        # CANVAS
        # ----------------------------------------------------

        self.canvas = tk.Canvas(
            self.root,
            width=LARGURA_JANELA,
            height=ALTURA_JANELA,
            bg="white"
        )

        self.canvas.pack()

        # ----------------------------------------------------
        # CENTRO
        # ----------------------------------------------------

        self.centro_x = LARGURA_JANELA // 2
        self.centro_y = ALTURA_JANELA // 2

        self.x = self.centro_x
        self.y = self.centro_y

        self.raio = 10

        # ----------------------------------------------------
        # CURSOR
        # ----------------------------------------------------

        self.cursor = self.canvas.create_oval(
            self.x - self.raio,
            self.y - self.raio,
            self.x + self.raio,
            self.y + self.raio,
            fill="black"
        )

        # ----------------------------------------------------
        # TÍTULO
        # ----------------------------------------------------

        self.texto_titulo = self.canvas.create_text(
            self.centro_x,
            30,
            text="Mouse Virtual - SVM",
            font=("Arial", 18, "bold")
        )

        # ----------------------------------------------------
        # INFORMAÇÕES
        # ----------------------------------------------------

        self.texto_arquivo = self.canvas.create_text(
            20,
            70,
            anchor="w",
            text="Arquivo: nenhum",
            font=("Arial", 12)
        )

        self.texto_classe_real = self.canvas.create_text(
            20,
            95,
            anchor="w",
            text="Classe real: -",
            font=("Arial", 12)
        )

        self.texto_janela = self.canvas.create_text(
            20,
            120,
            anchor="w",
            text="Janela: -",
            font=("Arial", 12)
        )

        self.texto_previsao = self.canvas.create_text(
            20,
            145,
            anchor="w",
            text="Previsão: -",
            font=("Arial", 12)
        )

        self.texto_confianca = self.canvas.create_text(
            20,
            170,
            anchor="w",
            text="Confiança: -",
            font=("Arial", 12)
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status = self.canvas.create_text(
            self.centro_x,
            ALTURA_JANELA - 80,
            text="Aguardando teste...",
            font=("Arial", 15, "bold")
        )

        # ----------------------------------------------------
        # BOTÕES
        # ----------------------------------------------------

        self.botao_iniciar = tk.Button(
            self.root,
            text="INICIAR",
            width=12,
            command=self.iniciar_teste
        )

        self.botao_iniciar.place(
            x=220,
            y=ALTURA_JANELA - 55
        )

        self.botao_pausar = tk.Button(
            self.root,
            text="PAUSAR",
            width=12,
            command=self.pausar_teste
        )

        self.botao_pausar.place(
            x=350,
            y=ALTURA_JANELA - 55
        )

        self.botao_centro = tk.Button(
            self.root,
            text="CENTRO",
            width=12,
            command=self.voltar_centro
        )

        self.botao_centro.place(
            x=480,
            y=ALTURA_JANELA - 55
        )

        # ----------------------------------------------------
        # CONTROLE
        # ----------------------------------------------------

        self.teste_ativo = False
        self.teste_pausado = False

        self.janelas = []
        self.indice_janela = 0

        self.resultado_atual = None

        self.root.update()

    # ========================================================
    # CURSOR
    # ========================================================

    def atualizar_cursor(self):

        self.canvas.coords(
            self.cursor,
            self.x - self.raio,
            self.y - self.raio,
            self.x + self.raio,
            self.y + self.raio
        )

        self.root.update_idletasks()

    # ========================================================
    # MOVER
    # ========================================================

    def mover(self, classe):

        if classe == "cima":

            self.y -= VELOCIDADE_MOUSE

        elif classe == "baixo":

            self.y += VELOCIDADE_MOUSE

        elif classe == "esquerda":

            self.x -= VELOCIDADE_MOUSE

        elif classe == "direita":

            self.x += VELOCIDADE_MOUSE

        # ----------------------------------------------------
        # LIMITES
        # ----------------------------------------------------

        self.x = max(
            self.raio,
            min(
                LARGURA_JANELA - self.raio,
                self.x
            )
        )

        self.y = max(
            self.raio,
            min(
                ALTURA_JANELA - self.raio,
                self.y
            )
        )

        self.canvas.itemconfig(
            self.status,
            text=f"Movimento: {classe}"
        )

        self.atualizar_cursor()

    # ========================================================
    # CENTRO
    # ========================================================

    def voltar_centro(self):

        self.x = self.centro_x
        self.y = self.centro_y

        self.canvas.itemconfig(
            self.status,
            text="Cursor no centro"
        )

        self.atualizar_cursor()

    # ========================================================
    # ATUALIZAR INFORMAÇÕES
    # ========================================================

    def atualizar_informacoes(
        self,
        arquivo=None,
        classe_real=None,
        janela=None,
        total=None,
        previsao=None,
        confianca=None
    ):

        if arquivo is not None:

            self.canvas.itemconfig(
                self.texto_arquivo,
                text=f"Arquivo: {arquivo}"
            )

        if classe_real is not None:

            self.canvas.itemconfig(
                self.texto_classe_real,
                text=f"Classe real: {classe_real}"
            )

        if janela is not None and total is not None:

            self.canvas.itemconfig(
                self.texto_janela,
                text=f"Janela: {janela}/{total}"
            )

        if previsao is not None:

            self.canvas.itemconfig(
                self.texto_previsao,
                text=f"Previsão: {previsao}"
            )

        if confianca is not None:

            self.canvas.itemconfig(
                self.texto_confianca,
                text=f"Confiança: {confianca:.2f}%"
            )

        self.root.update()

    # ========================================================
    # INICIAR TESTE
    # ========================================================

    def iniciar_teste(self):

        if self.teste_ativo:

            self.teste_pausado = False

            self.canvas.itemconfig(
                self.status,
                text="Teste em execução..."
            )

            return

        print("\nMouse virtual pronto.")

    # ========================================================
    # PAUSAR
    # ========================================================

    def pausar_teste(self):

        if not self.teste_ativo:

            return

        self.teste_pausado = not self.teste_pausado

        if self.teste_pausado:

            self.canvas.itemconfig(
                self.status,
                text="TESTE PAUSADO"
            )

        else:

            self.canvas.itemconfig(
                self.status,
                text="Teste em execução..."
            )

        self.root.update()

    # ========================================================
    # FECHAR
    # ========================================================

    def fechar(self):

        try:

            self.root.destroy()

        except:

            pass


# ============================================================
# ARQUIVOS
# ============================================================

def obter_arquivos():

    arquivos = []

    for classe, nomes in ARQUIVOS.items():

        if nomes is None:
            continue

        if isinstance(nomes, str):

            nomes = [nomes]

        for nome in nomes:

            caminho = os.path.join(
                PASTA_DADOS,
                nome
            )

            if os.path.isfile(caminho):

                arquivos.append({
                    "arquivo": nome,
                    "classe": classe
                })

    return arquivos


# ============================================================
# ARQUIVOS TREINADOS
# ============================================================

def carregar_arquivos_treinados():

    if not os.path.exists(
        ARQUIVO_INFO_TREINO
    ):

        print(
            "\nAviso: arquivo de informações "
            "do treinamento não encontrado."
        )

        return set()

    try:

        with open(
            ARQUIVO_INFO_TREINO,
            "r",
            encoding="utf-8"
        ) as arquivo:

            dados = json.load(arquivo)

        return {
            item["arquivo"]
            for item in dados.get(
                "arquivos_treinamento",
                []
            )
        }

    except Exception as erro:

        print(
            f"\nErro ao carregar informações "
            f"do treinamento: {erro}"
        )

        return set()


# ============================================================
# MOSTRAR ARQUIVOS
# ============================================================

def mostrar_arquivos(arquivos):

    print("\n================================")
    print("ARQUIVOS DISPONÍVEIS")
    print("================================")

    for i, item in enumerate(
        arquivos,
        start=1
    ):

        print(
            f"{i} - "
            f"{item['arquivo']} "
            f"-> {item['classe']}"
        )


# ============================================================
# SELECIONAR ARQUIVOS
# ============================================================

def selecionar_arquivos(arquivos):

    print("\n================================")
    print("MODO DE TESTE")
    print("================================")

    print("\n1 - Testar 1 arquivo")
    print("2 - Escolher vários arquivos")
    print("3 - Testar todos os arquivos")

    while True:

        opcao = input("\n> ").strip()

        if opcao in ["1", "2", "3"]:
            break

        print("Opção inválida.")

    # --------------------------------------------------------
    # UM ARQUIVO
    # --------------------------------------------------------

    if opcao == "1":

        while True:

            try:

                numero = int(
                    input(
                        "\nNúmero do arquivo: "
                    )
                )

                if (
                    numero < 1
                    or numero > len(arquivos)
                ):

                    raise ValueError

                return [
                    arquivos[numero - 1]
                ]

            except ValueError:

                print(
                    "Número inválido."
                )

    # --------------------------------------------------------
    # VÁRIOS
    # --------------------------------------------------------

    if opcao == "2":

        while True:

            entrada = input(
                "\nDigite os números separados por vírgula: "
            ).strip()

            try:

                numeros = []

                for valor in entrada.split(","):

                    numero = int(
                        valor.strip()
                    )

                    if (
                        numero < 1
                        or numero > len(arquivos)
                    ):

                        raise ValueError

                    numeros.append(numero)

                numeros = list(
                    dict.fromkeys(numeros)
                )

                return [
                    arquivos[i - 1]
                    for i in numeros
                ]

            except ValueError:

                print(
                    "Entrada inválida."
                )

    # --------------------------------------------------------
    # TODOS
    # --------------------------------------------------------

    return arquivos.copy()


# ============================================================
# RESULTADOS
# ============================================================

def preparar_resultados():

    os.makedirs(
        PASTA_RESULTADOS,
        exist_ok=True
    )


# ============================================================
# TESTAR ARQUIVO
# ============================================================

def testar_arquivo(
    modelo,
    item,
    arquivos_treinados,
    mouse
):

    arquivo = item["arquivo"]
    classe_real = item["classe"]

    foi_treinado = (
        arquivo in arquivos_treinados
    )

    caminho = os.path.join(
        PASTA_DADOS,
        arquivo
    )

    print("\n--------------------------------")

    print(
        f"Arquivo: {arquivo}"
    )

    if foi_treinado:

        print(
            "Status: TREINADO"
        )

    else:

        print(
            "Status: NÃO TREINADO"
        )

    dados = carregar_arquivo(
        caminho
    )

    if dados is None:
        return None

    janelas = criar_janelas(
        dados
    )

    if len(janelas) == 0:

        print(
            "Nenhuma janela criada."
        )

        return None

    print(
        f"Janelas: {len(janelas)}"
    )

    # --------------------------------------------------------
    # CONFIGURAÇÃO DO MOUSE
    # --------------------------------------------------------

    mouse.teste_ativo = True
    mouse.teste_pausado = False

    mouse.atualizar_informacoes(
        arquivo=arquivo,
        classe_real=classe_real,
        janela=0,
        total=len(janelas),
        previsao="-",
        confianca=None
    )

    predicoes = []

    print(
        "\nExecutando mouse virtual..."
    )

    # --------------------------------------------------------
    # TESTE DAS JANELAS
    # --------------------------------------------------------

    for i, janela in enumerate(
        janelas,
        start=1
    ):

        # ----------------------------------------------------
        # PAUSA
        # ----------------------------------------------------

        while mouse.teste_pausado:

            mouse.root.update()

        # ----------------------------------------------------
        # PREVISÃO
        # ----------------------------------------------------

        previsao = modelo.predict(
            np.array([janela])
        )[0]

        previsao = str(
            previsao
        )

        predicoes.append(
            previsao
        )

        # ----------------------------------------------------
        # ATUALIZA INTERFACE
        # ----------------------------------------------------

        mouse.atualizar_informacoes(
            janela=i,
            total=len(janelas),
            previsao=previsao
        )

        # ----------------------------------------------------
        # MOVE O CURSOR
        # ----------------------------------------------------

        mouse.mover(
            previsao
        )

        print(
            f"Janela {i}/{len(janelas)} "
            f"-> {previsao}"
        )

        # ----------------------------------------------------
        # ATUALIZA TKINTER
        # ----------------------------------------------------

        mouse.root.update()

        # ----------------------------------------------------
        # PEQUENA PAUSA
        # ----------------------------------------------------

        mouse.root.after(
            TEMPO_JANELA_MS
        )

        mouse.root.update()

    # --------------------------------------------------------
    # FINAL DO TESTE
    # --------------------------------------------------------

    pred = np.array(
        predicoes
    )

    valores, quantidades = np.unique(
        pred,
        return_counts=True
    )

    total = len(pred)

    distribuicao = {}

    for valor, quantidade in zip(
        valores,
        quantidades
    ):

        distribuicao[str(valor)] = (
            quantidade / total * 100
        )

    indice = np.argmax(
        quantidades
    )

    classe_prevista = str(
        valores[indice]
    )

    confianca = (
        quantidades[indice]
        /
        total
        *
        100
    )

    acertou = (
        classe_prevista
        ==
        classe_real
    )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print(
        f"\nClasse real: {classe_real}"
    )

    print(
        f"Classe prevista: {classe_prevista}"
    )

    print(
        f"Confiança: {confianca:.2f}%"
    )

    if acertou:

        print(
            "Resultado: ACERTO"
        )

    else:

        print(
            "Resultado: ERRO"
        )

    print(
        "\nDistribuição:"
    )

    for classe, porcentagem in (
        distribuicao.items()
    ):

        print(
            f"  {classe}: "
            f"{porcentagem:.2f}%"
        )

    # --------------------------------------------------------
    # MOSTRA RESULTADO NA JANELA
    # --------------------------------------------------------

    mouse.atualizar_informacoes(
        janela=len(janelas),
        total=len(janelas),
        previsao=classe_prevista,
        confianca=confianca
    )

    mouse.canvas.itemconfig(
        mouse.status,
        text=(
            "ACERTO"
            if acertou
            else "ERRO"
        )
    )

    mouse.root.update()

    # --------------------------------------------------------
    # VOLTA PARA O CENTRO
    # --------------------------------------------------------

    print(
        "\nRetornando mouse virtual para o centro..."
    )

    mouse.voltar_centro()

    mouse.teste_ativo = False
    mouse.teste_pausado = False

    return {

        "arquivo":
            arquivo,

        "classe_real":
            classe_real,

        "classe_prevista":
            classe_prevista,

        "janelas":
            total,

        "confianca":
            confianca,

        "acerto":
            acertou,

        "foi_treinado":
            foi_treinado,

        "distribuicao":
            distribuicao
    }


# ============================================================
# CSV
# ============================================================

def salvar_csv(resultados):

    caminho = os.path.join(
        PASTA_RESULTADOS,
        "resultados.csv"
    )

    classes = set()

    for resultado in resultados:

        classes.update(
            resultado["distribuicao"].keys()
        )

    classes = sorted(
        classes
    )

    campos = [
        "arquivo",
        "classe_real",
        "classe_prevista",
        "foi_treinado",
        "janelas",
        "confianca",
        "resultado"
    ]

    for classe in classes:

        campos.append(
            f"porcentagem_{classe}"
        )

    with open(
        caminho,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )

        escritor.writeheader()

        for resultado in resultados:

            linha = {

                "arquivo":
                    resultado["arquivo"],

                "classe_real":
                    resultado["classe_real"],

                "classe_prevista":
                    resultado["classe_prevista"],

                "foi_treinado":
                    "SIM"
                    if resultado["foi_treinado"]
                    else "NÃO",

                "janelas":
                    resultado["janelas"],

                "confianca":
                    f"{resultado['confianca']:.2f}",

                "resultado":
                    "ACERTO"
                    if resultado["acerto"]
                    else "ERRO"
            }

            for classe in classes:

                linha[
                    f"porcentagem_{classe}"
                ] = (
                    f"{resultado['distribuicao'].get(classe, 0):.2f}"
                )

            escritor.writerow(
                linha
            )

    return caminho


# ============================================================
# RELATÓRIO
# ============================================================

def salvar_relatorio(resultados):

    caminho = os.path.join(
        PASTA_RESULTADOS,
        "relatorio.txt"
    )

    total = len(resultados)

    acertos = sum(
        r["acerto"]
        for r in resultados
    )

    erros = total - acertos

    acuracia = (
        acertos / total * 100
        if total > 0
        else 0
    )

    treinados = sum(
        r["foi_treinado"]
        for r in resultados
    )

    nao_treinados = (
        total - treinados
    )

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "RELATÓRIO DE TESTE DO SVM\n"
        )

        arquivo.write(
            "=" * 60 + "\n\n"
        )

        arquivo.write(
            f"Arquivos testados: {total}\n"
        )

        arquivo.write(
            f"Arquivos treinados: {treinados}\n"
        )

        arquivo.write(
            f"Arquivos não treinados: "
            f"{nao_treinados}\n"
        )

        arquivo.write(
            f"Acertos: {acertos}\n"
        )

        arquivo.write(
            f"Erros: {erros}\n"
        )

        arquivo.write(
            f"Acurácia por arquivo: "
            f"{acuracia:.2f}%\n"
        )

        arquivo.write("\n")

        arquivo.write(
            "=" * 60 + "\n"
        )

        arquivo.write(
            "RESULTADOS INDIVIDUAIS\n"
        )

        arquivo.write(
            "=" * 60 + "\n\n"
        )

        for resultado in resultados:

            arquivo.write(
                f"Arquivo: "
                f"{resultado['arquivo']}\n"
            )

            arquivo.write(
                f"Classe real: "
                f"{resultado['classe_real']}\n"
            )

            arquivo.write(
                f"Classe prevista: "
                f"{resultado['classe_prevista']}\n"
            )

            arquivo.write(
                "Participou do treinamento: "
                +
                (
                    "SIM"
                    if resultado["foi_treinado"]
                    else "NÃO"
                )
                +
                "\n"
            )

            arquivo.write(
                f"Janelas: "
                f"{resultado['janelas']}\n"
            )

            arquivo.write(
                f"Confiança por votação: "
                f"{resultado['confianca']:.2f}%\n"
            )

            arquivo.write(
                "Resultado: "
                +
                (
                    "ACERTO"
                    if resultado["acerto"]
                    else "ERRO"
                )
                +
                "\n"
            )

            arquivo.write(
                "\nDistribuição:\n"
            )

            for classe, porcentagem in (
                resultado["distribuicao"].items()
            ):

                arquivo.write(
                    f"  {classe}: "
                    f"{porcentagem:.2f}%\n"
                )

            arquivo.write("\n")

    return caminho


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def salvar_matriz_confusao(resultados):

    if not resultados:
        return None

    reais = [
        r["classe_real"]
        for r in resultados
    ]

    previstos = [
        r["classe_prevista"]
        for r in resultados
    ]

    classes = sorted(
        set(reais + previstos)
    )

    matriz = confusion_matrix(
        reais,
        previstos,
        labels=classes
    )

    caminho = os.path.join(
        PASTA_RESULTADOS,
        "matriz_confusao.csv"
    )

    with open(
        caminho,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.writer(
            arquivo
        )

        escritor.writerow(
            ["real/previsto"] + classes
        )

        for classe, linha in zip(
            classes,
            matriz
        ):

            escritor.writerow(
                [classe] + linha.tolist()
            )

    return caminho


# ============================================================
# RESULTADO FINAL
# ============================================================

def mostrar_resultado_final(resultados):

    total = len(resultados)

    acertos = sum(
        r["acerto"]
        for r in resultados
    )

    erros = total - acertos

    treinados = sum(
        r["foi_treinado"]
        for r in resultados
    )

    nao_treinados = (
        total - treinados
    )

    acuracia = (
        acertos / total * 100
        if total > 0
        else 0
    )

    print("\n================================")
    print("RESULTADO FINAL")
    print("================================")

    print(
        f"Arquivos testados: {total}"
    )

    print(
        f"Arquivos treinados: {treinados}"
    )

    print(
        f"Arquivos não treinados: "
        f"{nao_treinados}"
    )

    print(
        f"Acertos: {acertos}"
    )

    print(
        f"Erros: {erros}"
    )

    print(
        f"Acurácia por arquivo: "
        f"{acuracia:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main(mouse):

    print("\n================================")
    print("TESTE DO MODELO SVM + MOUSE VIRTUAL")
    print("================================")

    if not os.path.exists(
        MODELO_PATH
    ):

        raise Exception(
            f"Modelo não encontrado: "
            f"{MODELO_PATH}"
        )

    print(
        f"\nCarregando modelo: "
        f"{MODELO_PATH}"
    )

    modelo = joblib.load(
        MODELO_PATH
    )

    print(
        "Modelo carregado."
    )

    # --------------------------------------------------------
    # ARQUIVOS TREINADOS
    # --------------------------------------------------------

    arquivos_treinados = (
        carregar_arquivos_treinados()
    )

    print("\n================================")
    print("ARQUIVOS DO TREINAMENTO")
    print("================================")

    if arquivos_treinados:

        for arquivo in sorted(
            arquivos_treinados
        ):

            print(
                f"- {arquivo}"
            )

    else:

        print(
            "Nenhum arquivo identificado."
        )

    # --------------------------------------------------------
    # ARQUIVOS
    # --------------------------------------------------------

    arquivos = obter_arquivos()

    if not arquivos:

        raise Exception(
            "Nenhum arquivo encontrado."
        )

    mostrar_arquivos(
        arquivos
    )

    selecionados = selecionar_arquivos(
        arquivos
    )

    print("\n================================")
    print("ARQUIVOS SELECIONADOS")
    print("================================")

    for item in selecionados:

        treinado = (
            "TREINADO"
            if item["arquivo"]
            in arquivos_treinados
            else "NÃO TREINADO"
        )

        print(
            f"- {item['arquivo']} "
            f"-> {item['classe']} "
            f"[{treinado}]"
        )

    preparar_resultados()

    resultados = []

    print("\n================================")
    print("TESTANDO")
    print("================================")

    try:

        for i, item in enumerate(
            selecionados,
            start=1
        ):

            print(
                f"\n[{i}/{len(selecionados)}]"
            )

            resultado = testar_arquivo(
                modelo,
                item,
                arquivos_treinados,
                mouse
            )

            if resultado is not None:

                resultados.append(
                    resultado
                )

    except KeyboardInterrupt:

        print(
            "\nTeste interrompido."
        )

        mouse.voltar_centro()

    if not resultados:

        raise Exception(
            "Nenhum arquivo pôde ser testado."
        )

    mostrar_resultado_final(
        resultados
    )

    caminho_txt = salvar_relatorio(
        resultados
    )

    caminho_csv = salvar_csv(
        resultados
    )

    caminho_matriz = salvar_matriz_confusao(
        resultados
    )

    print("\n================================")
    print("RESULTADOS SALVOS")
    print("================================")

    print(
        f"TXT: {caminho_txt}"
    )

    print(
        f"CSV: {caminho_csv}"
    )

    print(
        f"Matriz: {caminho_matriz}"
    )

    print("\n================================")
    print("TESTE FINALIZADO")
    print("================================")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    mouse = MouseVirtual()

    try:

        while True:

            main(mouse)

            opcao = input(
                "\nDeseja realizar outro teste? (s/n): "
            ).strip().lower()

            if opcao != "s":

                break

            mouse.voltar_centro()

    except KeyboardInterrupt:

        print(
            "\nPrograma interrompido."
        )

    finally:

        mouse.voltar_centro()

        mouse.fechar()

    input(
        "\nPressione ENTER para encerrar..."
    )