import tkinter as tk


class MouseVirtual:

    def __init__(self):
        self.root = tk.Tk()

        self.root.title("Mouse Virtual - SVM")
        self.root.geometry("800x600")
        self.root.resizable(False, False)


        self.executando = False
        self.fechado = False


        self.centro_x = 400
        self.centro_y = 300

        self.x = self.centro_x
        self.y = self.centro_y

        self.raio = 10


        self.canvas = tk.Canvas(
            self.root,
            width=800,
            height=600,
            bg="white"
        )

        self.canvas.pack()


        self.cursor = self.canvas.create_oval(
            self.x - self.raio,
            self.y - self.raio,
            self.x + self.raio,
            self.y + self.raio,
            fill="black"
        )


        self.arquivo_texto = self.canvas.create_text(
            20,
            25,
            anchor="w",
            text="Arquivo: -",
            font=("Arial", 12)
        )

        self.status_treinamento_texto = self.canvas.create_text(
            20,
            50,
            anchor="w",
            text="Treinamento: -",
            font=("Arial", 12)
        )

        self.classe_texto = self.canvas.create_text(
            20,
            75,
            anchor="w",
            text="Classe real: -",
            font=("Arial", 12)
        )

        self.janela_texto = self.canvas.create_text(
            20,
            100,
            anchor="w",
            text="Janela: 0/0",
            font=("Arial", 12)
        )

        self.previsao_texto = self.canvas.create_text(
            20,
            125,
            anchor="w",
            text="Previsão: -",
            font=("Arial", 12)
        )

        self.intensidade_texto = self.canvas.create_text(
            20,
            150,
            anchor="w",
            text="Intensidade: 0.0%",
            font=("Arial", 12)
        )

        self.velocidade_texto = self.canvas.create_text(
            20,
            175,
            anchor="w",
            text="Velocidade: 0 px",
            font=("Arial", 12)
        )

        self.status_texto = self.canvas.create_text(
            20,
            200,
            anchor="w",
            text="Status: PARADO",
            font=("Arial", 12)
        )

        self.arquivo_contador = self.canvas.create_text(
            20,
            225,
            anchor="w",
            text="Arquivo: 0/0",
            font=("Arial", 12)
        )

        self.arquivos_treinados_texto = self.canvas.create_text(
            20,
            250,
            anchor="w",
            text="Arquivos treinados: 0",
            font=("Arial", 12)
        )


        self.botao_iniciar = tk.Button(
            self.root,
            text="INICIAR",
            width=12,
            command=self.iniciar
        )

        self.botao_iniciar.place(
            x=120,
            y=550
        )

        self.botao_parar = tk.Button(
            self.root,
            text="PARAR",
            width=12,
            command=self.parar
        )

        self.botao_parar.place(
            x=270,
            y=550
        )

        self.botao_centro = tk.Button(
            self.root,
            text="CENTRO",
            width=12,
            command=self.voltar_centro
        )

        self.botao_centro.place(
            x=420,
            y=550
        )

        self.botao_fechar = tk.Button(
            self.root,
            text="FECHAR",
            width=12,
            command=self.fechar
        )

        self.botao_fechar.place(
            x=570,
            y=550
        )


        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.fechar
        )

        self.root.update()


    def iniciar(self):
        if self.fechado:
            return

        self.executando = True

        self.atualizar_status(
            "EXECUTANDO"
        )


    def parar(self):
        if self.fechado:
            return

        self.executando = False

        self.atualizar_status(
            "PARADO"
        )


    def atualizar_arquivo(
        self,
        arquivo,
        classe_real,
        indice,
        total,
        foi_treinado,
        total_arquivos,
        total_treinados
    ):

        if self.fechado:
            return

        status_treinamento = (
            "TREINADO"
            if foi_treinado
            else "NÃO TREINADO"
        )

        self.canvas.itemconfig(
            self.arquivo_texto,
            text=f"Arquivo: {arquivo}"
        )

        self.canvas.itemconfig(
            self.status_treinamento_texto,
            text=f"Treinamento: {status_treinamento}"
        )

        self.canvas.itemconfig(
            self.classe_texto,
            text=f"Classe real: {classe_real}"
        )

        self.canvas.itemconfig(
            self.arquivo_contador,
            text=f"Arquivo: {indice}/{total}"
        )

        self.canvas.itemconfig(
            self.arquivos_treinados_texto,
            text=(
                f"Arquivos treinados: {total_treinados}"
                f" de {total_arquivos}"
            )
        )

        self.root.update_idletasks()
        self.root.update()


    def atualizar_janela(
        self,
        janela,
        total,
        previsao,
        intensidade,
        velocidade
    ):

        if self.fechado:
            return

        self.canvas.itemconfig(
            self.janela_texto,
            text=f"Janela: {janela}/{total}"
        )

        self.canvas.itemconfig(
            self.previsao_texto,
            text=f"Previsão: {previsao}"
        )

        self.canvas.itemconfig(
            self.intensidade_texto,
            text=(
                f"Intensidade: "
                f"{intensidade * 100:.1f}%"
            )
        )

        self.canvas.itemconfig(
            self.velocidade_texto,
            text=f"Velocidade: {velocidade} px"
        )

        self.root.update_idletasks()
        self.root.update()


    def atualizar_status(self, status):

        if self.fechado:
            return

        self.canvas.itemconfig(
            self.status_texto,
            text=f"Status: {status}"
        )

        self.root.update_idletasks()
        self.root.update()


    def mover(
        self,
        classe,
        intensidade
    ):

        if self.fechado:
            return

        intensidade = max(
            0.0,
            min(
                1.0,
                float(intensidade)
            )
        )


        velocidade = int(
            3
            +
            (
                intensidade ** 1.5
            )
            * 27
        )


        if classe == "cima":

            self.y -= velocidade

        elif classe == "baixo":

            self.y += velocidade

        elif classe == "esquerda":

            self.x -= velocidade

        elif classe == "direita":

            self.x += velocidade


        self.x = max(
            self.raio,
            min(
                800 - self.raio,
                self.x
            )
        )

        self.y = max(
            self.raio,
            min(
                600 - self.raio,
                self.y
            )
        )


        self.canvas.coords(
            self.cursor,
            self.x - self.raio,
            self.y - self.raio,
            self.x + self.raio,
            self.y + self.raio
        )

        self.canvas.itemconfig(
            self.previsao_texto,
            text=f"Previsão: {classe}"
        )

        self.canvas.itemconfig(
            self.intensidade_texto,
            text=(
                f"Intensidade: "
                f"{intensidade * 100:.1f}%"
            )
        )

        self.canvas.itemconfig(
            self.velocidade_texto,
            text=f"Velocidade: {velocidade} px"
        )

        self.root.update_idletasks()
        self.root.update()


    def mostrar_parado(
        self,
        previsao,
        intensidade
    ):

        if self.fechado:
            return

        self.canvas.itemconfig(
            self.previsao_texto,
            text=f"Previsão: {previsao}"
        )

        self.canvas.itemconfig(
            self.intensidade_texto,
            text=(
                f"Intensidade: "
                f"{intensidade * 100:.1f}%"
            )
        )

        self.canvas.itemconfig(
            self.velocidade_texto,
            text="Velocidade: 0 px"
        )

        self.root.update_idletasks()
        self.root.update()


    def voltar_centro(self):

        if self.fechado:
            return

        self.x = self.centro_x
        self.y = self.centro_y

        self.canvas.coords(
            self.cursor,
            self.x - self.raio,
            self.y - self.raio,
            self.x + self.raio,
            self.y + self.raio
        )

        self.root.update_idletasks()
        self.root.update()


    def existe(self):

        if self.fechado:
            return False

        try:
            return bool(
                self.root.winfo_exists()
            )
        except tk.TclError:
            return False


    def fechar(self):

        if self.fechado:
            return

        self.fechado = True
        self.executando = False

        try:
            self.root.destroy()
        except tk.TclError:
            pass