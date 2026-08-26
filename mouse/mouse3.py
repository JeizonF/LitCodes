import os
import csv
import json
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
    criar_janelas,
    calcular_intensidade,
    calcular_velocidade,
    TEMPO_MOUSE
)

from mouse_teclado import MouseTeclado as MouseVirtual

from sklearn.metrics import confusion_matrix


PASTA_RESULTADOS = "resultados"
ARQUIVO_INFO_TREINO = "arquivos_treinamento.json"


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


def mostrar_arquivos(arquivos):

    print(
        "\n================================"
    )

    print(
        "ARQUIVOS DISPONÍVEIS"
    )

    print(
        "================================"
    )

    for i, item in enumerate(
        arquivos,
        start=1
    ):

        print(
            f"{i} - "
            f"{item['arquivo']} "
            f"-> {item['classe']}"
        )


def selecionar_arquivos(arquivos):

    print(
        "\n================================"
    )

    print(
        "MODO DE TESTE"
    )

    print(
        "================================"
    )

    print(
        "\n1 - Testar 1 arquivo"
    )

    print(
        "2 - Escolher vários arquivos"
    )

    print(
        "3 - Testar todos os arquivos"
    )

    while True:

        opcao = input(
            "\n> "
        ).strip()

        if opcao in ["1", "2", "3"]:
            break

        print(
            "Opção inválida."
        )

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
                    or
                    numero > len(arquivos)
                ):
                    raise ValueError

                return [
                    arquivos[numero - 1]
                ]

            except ValueError:

                print(
                    "Número inválido."
                )

    if opcao == "2":

        while True:

            entrada = input(
                "\nDigite os números separados "
                "por vírgula: "
            ).strip()

            try:

                numeros = []

                for valor in entrada.split(","):

                    numero = int(
                        valor.strip()
                    )

                    if (
                        numero < 1
                        or
                        numero > len(arquivos)
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

    return arquivos.copy()


def preparar_resultados():

    os.makedirs(
        PASTA_RESULTADOS,
        exist_ok=True
    )


def testar_arquivo(
    modelo,
    item,
    arquivos_treinados,
    mouse,
    indice_arquivo,
    total_arquivos_teste,
    total_arquivos,
    total_treinados
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

    print(
        "\n--------------------------------"
    )

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

    janelas_features = criar_janelas(
        dados
    )

    if len(janelas_features) == 0:

        print(
            "Nenhuma janela criada."
        )

        return None

    total_janelas = len(
        janelas_features
    )

    print(
        f"Janelas: {total_janelas}"
    )

    mouse.atualizar_arquivo(
        arquivo,
        classe_real,
        indice_arquivo,
        total_arquivos_teste,
        foi_treinado,
        total_arquivos,
        total_treinados
    )

    mouse.atualizar_status(
        "EXECUTANDO"
    )

    #mouse.voltar_centro()

    mouse.iniciar()

    predicoes = []

    for i in range(
        total_janelas
    ):

        while not mouse.executando:

            mouse.root.update()

            if not mouse.root.winfo_exists():
                return None

        features = janelas_features[i]

        previsao = modelo.predict(
            np.array([
                features
            ])
        )[0]

        previsao = str(
            previsao
        )

        predicoes.append(
            previsao
        )

        inicio = i * 25

        fim = inicio + 100

        janela_original = dados.iloc[
            inicio:fim
        ]

        intensidade = calcular_intensidade(
            janela_original
        )

        velocidade = calcular_velocidade(
            intensidade
        )

        print(
            f"Janela {i + 1}/"
            f"{total_janelas} "
            f"-> {previsao} "
            f"| intensidade: "
            f"{intensidade * 100:.1f}%"
        )

        mouse.atualizar_janela(
            i + 1,
            total_janelas,
            previsao,
            intensidade,
            velocidade
        )

        if previsao == "piscada":

            mouse.clicar()

        elif previsao in [
            "cima",
            "baixo",
            "esquerda",
            "direita"
        ]:

            mouse.mover(
                previsao,
                intensidade
            )

        else:

            mouse.mostrar_parado(
                previsao,
                intensidade
            )

        mouse.root.after(
            TEMPO_MOUSE
        )

        mouse.root.update()

    if len(predicoes) == 0:
        return None

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
            quantidade /
            total *
            100
        )

    indice = np.argmax(
        quantidades
    )

    classe_prevista = str(
        valores[indice]
    )

    confianca = (
        quantidades[indice] /
        total *
        100
    )

    acertou = (
        classe_prevista ==
        classe_real
    )

    print(
        f"\nClasse real: "
        f"{classe_real}"
    )

    print(
        f"Classe prevista: "
        f"{classe_prevista}"
    )

    print(
        f"Confiança: "
        f"{confianca:.2f}%"
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

    #mouse.voltar_centro()

    return {
        "arquivo": arquivo,
        "classe_real": classe_real,
        "classe_prevista": classe_prevista,
        "janelas": total,
        "confianca": confianca,
        "acerto": acertou,
        "foi_treinado": foi_treinado,
        "distribuicao": distribuicao
    }


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

    classes = sorted(classes)

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
                    (
                        "SIM"
                        if resultado["foi_treinado"]
                        else "NÃO"
                    ),

                "janelas":
                    resultado["janelas"],

                "confianca":
                    f"{resultado['confianca']:.2f}",

                "resultado":
                    (
                        "ACERTO"
                        if resultado["acerto"]
                        else "ERRO"
                    )
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


def salvar_relatorio(resultados):

    caminho = os.path.join(
        PASTA_RESULTADOS,
        "relatorio.txt"
    )

    total = len(
        resultados
    )

    acertos = sum(
        r["acerto"]
        for r in resultados
    )

    erros = (
        total -
        acertos
    )

    acuracia = (
        acertos /
        total *
        100
        if total > 0
        else 0
    )

    treinados = sum(
        r["foi_treinado"]
        for r in resultados
    )

    nao_treinados = (
        total -
        treinados
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
            "=" * 60 +
            "\n\n"
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

        arquivo.write(
            "\n"
        )

        arquivo.write(
            "=" * 60 +
            "\n"
        )

        arquivo.write(
            "RESULTADOS INDIVIDUAIS\n"
        )

        arquivo.write(
            "=" * 60 +
            "\n\n"
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

            arquivo.write(
                "\n"
            )

    return caminho


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
        set(
            reais +
            previstos
        )
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
            ["real/previsto"] +
            classes
        )

        for classe, linha in zip(
            classes,
            matriz
        ):

            escritor.writerow(
                [classe] +
                linha.tolist()
            )

    return caminho


def mostrar_resultado_final(resultados):

    total = len(
        resultados
    )

    acertos = sum(
        r["acerto"]
        for r in resultados
    )

    erros = (
        total -
        acertos
    )

    treinados = sum(
        r["foi_treinado"]
        for r in resultados
    )

    nao_treinados = (
        total -
        treinados
    )

    acuracia = (
        acertos /
        total *
        100
        if total > 0
        else 0
    )

    print(
        "\n================================"
    )

    print(
        "RESULTADO FINAL"
    )

    print(
        "================================"
    )

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


def main():

    print(
        "\n================================"
    )

    print(
        "TESTE DO MODELO SVM + MOUSE VIRTUAL"
    )

    print(
        "================================"
    )

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

    arquivos_treinados = (
        carregar_arquivos_treinados()
    )

    arquivos = obter_arquivos()

    if not arquivos:

        raise Exception(
            "Nenhum arquivo encontrado."
        )

    total_arquivos = len(
        arquivos
    )

    total_treinados = sum(
        1
        for item in arquivos
        if item["arquivo"] in arquivos_treinados
    )

    total_nao_treinados = (
        total_arquivos -
        total_treinados
    )

    print(
        "\n================================"
    )

    print(
        "INFORMAÇÕES DOS ARQUIVOS"
    )

    print(
        "================================"
    )

    print(
        f"Total de arquivos: "
        f"{total_arquivos}"
    )

    print(
        f"Arquivos treinados: "
        f"{total_treinados}"
    )

    print(
        f"Arquivos não treinados: "
        f"{total_nao_treinados}"
    )

    print(
        "\n================================"
    )

    print(
        "ARQUIVOS DO TREINAMENTO"
    )

    print(
        "================================"
    )

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

    mostrar_arquivos(
        arquivos
    )

    selecionados = selecionar_arquivos(
        arquivos
    )

    print(
        "\n================================"
    )

    print(
        "ARQUIVOS SELECIONADOS"
    )

    print(
        "================================"
    )

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

    mouse = MouseVirtual()

    resultados = []

    print(
        "\n================================"
    )

    print(
        "TESTANDO"
    )

    print(
        "================================"
    )

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
                mouse,
                i,
                len(selecionados),
                total_arquivos,
                total_treinados
            )

            if resultado is not None:

                resultados.append(
                    resultado
                )

    except KeyboardInterrupt:

        print(
            "\nTeste interrompido."
        )

        mouse.parar()

    finally:

        #mouse.voltar_centro()

        print(
            "\nMouse virtual finalizado."
        )

    if not resultados:

        mouse.fechar()

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

    caminho_matriz = (
        salvar_matriz_confusao(
            resultados
        )
    )

    print(
        "\n================================"
    )

    print(
        "RESULTADOS SALVOS"
    )

    print(
        "================================"
    )

    print(
        f"TXT: {caminho_txt}"
    )

    print(
        f"CSV: {caminho_csv}"
    )

    print(
        f"Matriz: {caminho_matriz}"
    )

    print(
        "\n================================"
    )

    print(
        "TESTE FINALIZADO"
    )

    print(
        "================================"
    )

    mouse.root.mainloop()


if __name__ == "__main__":
    main()