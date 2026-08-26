import threading
import tkinter as tk

import pyttsx3

from mouse_gui import MouseVirtual


COR_TECLA_NORMAL = "#f0f0f0"
COR_TECLA_HOVER = "#add8e6"
COR_TECLA_CLIQUE = "#90ee90"
COR_BORDA = "#888888"

TEMPO_DEBOUNCE_CLIQUE = 700
TEMPO_FLASH_CLIQUE = 250


class MouseTeclado(MouseVirtual):

    def __init__(self):
        self.root = tk.Tk()

        self.root.title("Mouse Virtual + Teclado - SVM")

        self.largura = 900
        self.altura = 760

        self.root.geometry(f"{self.largura}x{self.altura}")
        self.root.resizable(False, False)

        self.executando = False
        self.fechado = False

        self.centro_x = 450
        self.centro_y = 270

        self.x = self.centro_x
        self.y = self.centro_y

        self.raio = 10

        self.teclas = {}
        self.tecla_hover = None
        self.pode_clicar = True
        self.palavra_digitada = ""

        self.canvas = tk.Canvas(
            self.root,
            width=self.largura,
            height=self.altura,
            bg="white"
        )

        self.canvas.pack()

        self.arquivo_texto = self.canvas.create_text(
            20, 25, anchor="w", text="Arquivo: -", font=("Arial", 11)
        )

        self.status_treinamento_texto = self.canvas.create_text(
            20, 45, anchor="w", text="Treinamento: -", font=("Arial", 11)
        )

        self.classe_texto = self.canvas.create_text(
            20, 65, anchor="w", text="Classe real: -", font=("Arial", 11)
        )

        self.janela_texto = self.canvas.create_text(
            20, 85, anchor="w", text="Janela: 0/0", font=("Arial", 11)
        )

        self.previsao_texto = self.canvas.create_text(
            20, 105, anchor="w", text="Previsão: -", font=("Arial", 11)
        )

        self.intensidade_texto = self.canvas.create_text(
            20, 125, anchor="w", text="Intensidade: 0.0%", font=("Arial", 11)
        )

        self.velocidade_texto = self.canvas.create_text(
            20, 145, anchor="w", text="Velocidade: 0 px", font=("Arial", 11)
        )

        self.status_texto = self.canvas.create_text(
            20, 165, anchor="w", text="Status: PARADO", font=("Arial", 11)
        )

        self.arquivo_contador = self.canvas.create_text(
            20, 185, anchor="w", text="Arquivo: 0/0", font=("Arial", 11)
        )

        self.arquivos_treinados_texto = self.canvas.create_text(
            20, 205, anchor="w", text="Arquivos treinados: 0", font=("Arial", 11)
        )

        self.clique_texto = self.canvas.create_text(
            self.largura - 20, 25, anchor="e", text="",
            font=("Arial", 14, "bold"), fill="#2e8b57"
        )

        self._criar_teclado()

        self.palavra_texto = self.canvas.create_text(
            self.largura // 2, 640, text="Palavra: ",
            font=("Arial", 16, "bold")
        )

        self.cursor = self.canvas.create_oval(
            self.x - self.raio, self.y - self.raio,
            self.x + self.raio, self.y + self.raio,
            fill="black"
        )

        self.canvas.bind("<Button-1>", self._clique_mouse_fisico)
        self.canvas.bind("<Motion>", self._hover_mouse_fisico)

        self.botao_iniciar = tk.Button(
            self.root, text="INICIAR", width=12, command=self.iniciar
        )
        self.botao_iniciar.place(x=100, y=690)

        self.botao_parar = tk.Button(
            self.root, text="PARAR", width=12, command=self.parar
        )
        self.botao_parar.place(x=250, y=690)

        self.botao_centro = tk.Button(
            self.root, text="CENTRO", width=12, command=self.voltar_centro
        )
        self.botao_centro.place(x=400, y=690)

        self.botao_limpar = tk.Button(
            self.root, text="LIMPAR", width=12,
            command=lambda: self._processar_tecla("LIMPAR")
        )
        self.botao_limpar.place(x=550, y=690)

        self.botao_fechar = tk.Button(
            self.root, text="FECHAR", width=12, command=self.fechar
        )
        self.botao_fechar.place(x=700, y=690)

        self.root.protocol("WM_DELETE_WINDOW", self.fechar)

        self.root.update()

    def _criar_teclado(self):
        altura_tecla = 55
        espaco = 6

        self._criar_linha("QWERTYUIOP", 300, altura_tecla, espaco, 70)
        self._criar_linha("ASDFGHJKL", 300 + altura_tecla + espaco, altura_tecla, espaco, 70)
        self._criar_linha("ZXCVBNM", 300 + 2 * (altura_tecla + espaco), altura_tecla, espaco, 70)

        y_especiais = 300 + 3 * (altura_tecla + espaco)

        especiais = [
            ("APAGAR", 140),
            ("ESPAÇO", 250),
            ("ENTER", 140),
            ("LIMPAR", 140)
        ]

        largura_total = sum(w for _, w in especiais) + espaco * (len(especiais) - 1)
        x = (self.largura - largura_total) // 2

        for label, largura in especiais:
            self._criar_tecla(label, x, y_especiais, x + largura, y_especiais + altura_tecla)
            x += largura + espaco

    def _criar_linha(self, letras, y_top, altura_tecla, espaco, largura_tecla):
        largura_total = len(letras) * largura_tecla + espaco * (len(letras) - 1)
        x = (self.largura - largura_total) // 2

        for letra in letras:
            self._criar_tecla(letra, x, y_top, x + largura_tecla, y_top + altura_tecla)
            x += largura_tecla + espaco

    def _criar_tecla(self, label, x1, y1, x2, y2):
        cor_fundo = COR_TECLA_NORMAL

        if label == "ENTER":
            cor_fundo = "#ffe08a"

        rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=cor_fundo, outline=COR_BORDA, width=2
        )

        text_id = self.canvas.create_text(
            (x1 + x2) / 2, (y1 + y2) / 2,
            text=label, font=("Arial", 12, "bold")
        )

        self.teclas[label] = {
            "bbox": (x1, y1, x2, y2),
            "rect_id": rect_id,
            "text_id": text_id,
            "cor_normal": cor_fundo
        }

    def _tecla_em(self, x, y):
        for label, dados in self.teclas.items():
            x1, y1, x2, y2 = dados["bbox"]

            if x1 <= x <= x2 and y1 <= y <= y2:
                return label

        return None

    def _tecla_no_cursor(self):
        return self._tecla_em(self.x, self.y)

    def _definir_hover(self, tecla_atual):
        if tecla_atual == self.tecla_hover:
            return

        if self.tecla_hover is not None and self.tecla_hover in self.teclas:
            cor = self.teclas[self.tecla_hover]["cor_normal"]
            self.canvas.itemconfig(
                self.teclas[self.tecla_hover]["rect_id"],
                fill=cor
            )

        if tecla_atual is not None:
            self.canvas.itemconfig(
                self.teclas[tecla_atual]["rect_id"],
                fill=COR_TECLA_HOVER
            )

        self.tecla_hover = tecla_atual

    def _atualizar_hover(self):
        self._definir_hover(self._tecla_no_cursor())

    def _clique_mouse_fisico(self, event):
        if self.fechado:
            return

        tecla = self._tecla_em(event.x, event.y)

        if tecla is None:
            return

        self._processar_tecla(tecla)
        self._flash_tecla(tecla)

    def _hover_mouse_fisico(self, event):
        if self.fechado:
            return

        self._definir_hover(self._tecla_em(event.x, event.y))

    def _processar_tecla(self, tecla):
        if tecla == "APAGAR":
            self.palavra_digitada = self.palavra_digitada[:-1]
        elif tecla == "ESPAÇO":
            self.palavra_digitada += " "
        elif tecla == "LIMPAR":
            self.palavra_digitada = ""
        elif tecla == "ENTER":
            self._falar_palavra()
            self.palavra_digitada = ""
        else:
            self.palavra_digitada += tecla

        self.canvas.itemconfig(
            self.palavra_texto,
            text=f"Palavra: {self.palavra_digitada}"
        )

        self.root.update_idletasks()
        self.root.update()

    def _falar_palavra(self):
        texto = self.palavra_digitada.strip()

        if not texto:
            return

        thread = threading.Thread(
            target=self._executar_tts,
            args=(texto,),
            daemon=True
        )

        thread.start()

    def _executar_tts(self, texto):
        try:
            engine = pyttsx3.init()
            engine.say(texto)
            engine.runAndWait()
            engine.stop()
        except Exception as erro:
            print(f"Erro ao reproduzir voz: {erro}")

    def _flash_tecla(self, tecla):
        if tecla not in self.teclas:
            return

        self.canvas.itemconfig(
            self.teclas[tecla]["rect_id"],
            fill=COR_TECLA_CLIQUE
        )

        self.root.after(
            TEMPO_FLASH_CLIQUE,
            lambda: self._restaurar_cor_tecla(tecla)
        )

    def _restaurar_cor_tecla(self, tecla):
        if self.fechado or tecla not in self.teclas:
            return

        if self.tecla_hover == tecla:
            cor = COR_TECLA_HOVER
        else:
            cor = self.teclas[tecla]["cor_normal"]

        self.canvas.itemconfig(
            self.teclas[tecla]["rect_id"],
            fill=cor
        )

    def _liberar_clique(self):
        self.pode_clicar = True

    def clicar(self):
        if self.fechado:
            return

        if not self.pode_clicar:
            return

        tecla = self._tecla_no_cursor()

        if tecla is None:
            return

        self._processar_tecla(tecla)
        self._flash_tecla(tecla)

        self.canvas.itemconfig(self.clique_texto, text=f"CLIQUE: {tecla}")
        self.root.after(600, lambda: self.canvas.itemconfig(self.clique_texto, text=""))

        self.pode_clicar = False
        self.root.after(TEMPO_DEBOUNCE_CLIQUE, self._liberar_clique)

    def mover(self, classe, intensidade):
        if self.fechado:
            return

        intensidade = max(0.0, min(1.0, float(intensidade)))

        velocidade = int(3 + (intensidade ** 1.5) * 27)

        if classe == "cima":
            self.y -= velocidade
        elif classe == "baixo":
            self.y += velocidade
        elif classe == "esquerda":
            self.x -= velocidade
        elif classe == "direita":
            self.x += velocidade

        self.x = max(self.raio, min(self.largura - self.raio, self.x))
        self.y = max(self.raio, min(self.altura - self.raio, self.y))

        self.canvas.coords(
            self.cursor,
            self.x - self.raio, self.y - self.raio,
            self.x + self.raio, self.y + self.raio
        )

        self.canvas.itemconfig(self.previsao_texto, text=f"Previsão: {classe}")
        self.canvas.itemconfig(
            self.intensidade_texto,
            text=f"Intensidade: {intensidade * 100:.1f}%"
        )
        self.canvas.itemconfig(self.velocidade_texto, text=f"Velocidade: {velocidade} px")

        self._atualizar_hover()

        self.root.update_idletasks()
        self.root.update()

    def voltar_centro(self):
        if self.fechado:
            return

        self.x = self.centro_x
        self.y = self.centro_y

        self.canvas.coords(
            self.cursor,
            self.x - self.raio, self.y - self.raio,
            self.x + self.raio, self.y + self.raio
        )

        self._atualizar_hover()

        self.root.update_idletasks()
        self.root.update()