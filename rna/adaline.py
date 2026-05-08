import numpy as np


class Adaline:
    """
    Rede ADALINE (Adaptive Linear Neuron) com treinamento pela Regra Delta (LMS).

    Estrutura:
        - 4 entradas: x1, x2, x3, x4
        - 1 bias (x0 = +1, peso w0)
        - Total de 5 pesos: [w0, w1, w2, w3, w4]
        - Saída linear durante treinamento, sign() na inferência

    Referência: Adaline.docx — Lab. Inteligência Artificial, CEFET-MG Campus VIII
    """

    def __init__(self, learning_rate: float = 0.0025, precision: float = 1e-6, seed: int = None):
        """
        Parâmetros
        ----------
        learning_rate : float
            Taxa de aprendizado η (padrão: 0.0025 conforme especificação)
        precision : float
            Precisão ε para critério de parada (padrão: 1e-6)
        seed : int
            Semente para o gerador de números aleatórios (garante inicializações distintas)
        """
        self.learning_rate = learning_rate
        self.precision = precision
        self.seed = seed

        # Inicializar gerador e pesos
        self._rng = np.random.default_rng(seed)
        self._init_weights()

        # Histórico para análise e plotagem
        self.eqm_history: list[float] = []          # EQM por época
        self.initial_weights: np.ndarray = None      # Pesos antes do treino
        self.epochs_trained: int = 0                 # Épocas até convergência

    # ------------------------------------------------------------------
    # Inicialização
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Inicializa 5 pesos aleatórios em [0, 1]: [w0_bias, w1, w2, w3, w4]."""
        self.weights = self._rng.uniform(0.0, 1.0, size=5)

    # ------------------------------------------------------------------
    # Operações de inferência
    # ------------------------------------------------------------------

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Acrescenta o termo de bias (x0 = +1) ao vetor de entrada."""
        return np.concatenate(([1.0], x))

    def net_input(self, x: np.ndarray) -> float:
        """Calcula a saída combinada linear: u = w · [1, x1, x2, x3, x4]."""
        return float(np.dot(self.weights, self._augment(x)))

    def predict(self, x: np.ndarray) -> int:
        """
        Classifica uma amostra usando a função sign.

        Retorna
        -------
        +1  →  Válvula B
        -1  →  Válvula A
        """
        u = self.net_input(x)
        return 1 if u >= 0 else -1

    # ------------------------------------------------------------------
    # Treinamento — Regra Delta (on-line, padrão por padrão)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, d: np.ndarray) -> int:
        """
        Treina o ADALINE usando a Regra Delta (LMS on-line).

        Algoritmo:
            Para cada época:
                Para cada padrão p:
                    u(p) = w · x_aug(p)          (saída linear)
                    e(p) = d(p) - u(p)            (erro)
                    Δw   = η · e(p) · x_aug(p)   (regra delta)
                    w   ← w + Δw
                EQM = (1 / (2 * P)) * Σ e(p)²    (após todos os padrões)
            Parar quando |EQM_anterior - EQM_atual| < ε  (variação do EQM)

        Parâmetros
        ----------
        X : ndarray, shape (n_samples, 4)
            Matriz de entradas (x1, x2, x3, x4)
        d : ndarray, shape (n_samples,)
            Saídas desejadas (+1 ou -1)

        Retorna
        -------
        int : número de épocas até convergência
        """
        # Guardar pesos iniciais (antes de qualquer atualização)
        self.initial_weights = self.weights.copy()
        self.eqm_history = []

        epoch = 0
        P = len(X)
        eqm_prev = float("inf")
        max_epochs = 100_000   # teto de segurança

        while epoch < max_epochs:
            epoch += 1
            errors_sq = []

            for i in range(P):
                x_aug = self._augment(X[i])
                u = float(np.dot(self.weights, x_aug))   # saída linear
                e = float(d[i]) - u                       # erro

                # Atualização on-line (Regra Delta)
                self.weights = self.weights + self.learning_rate * e * x_aug

                errors_sq.append(e ** 2)

            # EQM da época
            eqm = sum(errors_sq) / (2 * P)
            self.eqm_history.append(eqm)

            # Critério de parada: variação do EQM entre épocas < precisão
            if abs(eqm_prev - eqm) < self.precision:
                break

            eqm_prev = eqm

        self.epochs_trained = epoch
        return epoch

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def classify_label(self, x: np.ndarray) -> str:
        """Retorna 'A' (Válvula A) ou 'B' (Válvula B)."""
        return "B" if self.predict(x) == 1 else "A"

    def __repr__(self) -> str:
        return (
            f"Adaline(η={self.learning_rate}, ε={self.precision}, "
            f"seed={self.seed}, épocas={self.epochs_trained})"
        )
