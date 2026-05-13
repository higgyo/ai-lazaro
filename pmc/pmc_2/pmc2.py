"""
pmc2.py — Classificação de conservantes (Tipo A, B, C) com PMC
Backpropagation Padrão vs. Backpropagation com Momentum
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ---------------------------------------------------------------------------
# Funções de ativação
# ---------------------------------------------------------------------------

def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

def sigmoid_deriv(y):
    """Derivada da logística quando y = sigmoid(v)"""
    return y * (1.0 - y)

# ---------------------------------------------------------------------------
# Inicialização de pesos
# ---------------------------------------------------------------------------

def init_weights(topology, seed=42):
    """
    Retorna lista de matrizes de peso W[l] com dimensão
    (n_neurons_l, n_neurons_{l-1} + 1)  [+1 para o bias]
    """
    np.random.seed(seed)
    weights = []
    for i in range(1, len(topology)):
        W = np.random.rand(topology[i], topology[i-1] + 1)
        weights.append(W)
    return weights

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(x, weights):
    """
    x : vetor coluna (n_inputs, 1)
    Retorna lista de saídas por camada (com bias já adicionado para camadas ocultas)
    e lista de saídas sem bias (ys)
    """
    activations = [x]
    for l, W in enumerate(weights):
        # Adiciona bias (-1 convenção) ao vetor de entrada da camada
        a_bias = np.vstack([np.array([[-1.0]]), activations[-1]])
        v = W @ a_bias
        y = sigmoid(v)
        activations.append(y)
    return activations

# ---------------------------------------------------------------------------
# Backward pass — backpropagation padrão
# ---------------------------------------------------------------------------

def train_bp(X, d, eta, precisao, W_init, max_epochs=100_000):
    """
    Treina com backpropagation padrão.

    Parâmetros
    ----------
    X          : (n_samples, n_inputs)
    d          : (n_samples, n_outputs)  — rótulos desejados
    eta        : taxa de aprendizado
    precisao   : critério de parada: EQM_atual ≤ precisao
    W_init     : lista de matrizes de peso iniciais
    max_epochs : limite máximo de épocas (segurança)

    Retorna
    -------
    weights_final, epocas, lista_eqm, tempo_s
    """
    n_samples = X.shape[0]
    n_outputs = d.shape[1]

    W = [w.copy() for w in W_init]
    lista_eqm = []
    epocas = 0
    eqm_anterior = float('inf')
    t_start = time.perf_counter()

    while True:
        eqm_epoch = 0.0

        for i in range(n_samples):
            x_i = X[i].reshape(-1, 1)          # (n_inputs, 1)
            d_i = d[i].reshape(-1, 1)           # (n_outputs, 1)

            activations = forward(x_i, W)
            y_out = activations[-1]

            erro = d_i - y_out
            eqm_epoch += np.sum(erro ** 2)

            deltas = [erro * sigmoid_deriv(y_out)]
            for l in range(len(W) - 2, -1, -1):
                W_nobias = W[l + 1][:, 1:]
                delta_next = deltas[0]
                y_l = activations[l + 1]
                delta_l = (W_nobias.T @ delta_next) * sigmoid_deriv(y_l)
                deltas.insert(0, delta_l)

            for l in range(len(W)):
                a_bias = np.vstack([np.array([[-1.0]]), activations[l]])
                W[l] += eta * (deltas[l] @ a_bias.T)

        eqm_atual = eqm_epoch / (n_samples * n_outputs)
        lista_eqm.append(eqm_atual)
        epocas += 1

        # Critério de parada: |ΔEQM| <= ε
        if abs(eqm_atual - eqm_anterior) <= precisao or epocas >= max_epochs:
            break
        eqm_anterior = eqm_atual

    tempo_s = time.perf_counter() - t_start
    return W, epocas, lista_eqm, tempo_s

# ---------------------------------------------------------------------------
# Backward pass — backpropagation com momentum
# ---------------------------------------------------------------------------

def train_bp_momentum(X, d, eta, alpha, precisao, W_init, max_epochs=100_000):
    """
    Treina com backpropagation + momentum.

    alpha      : fator de momentum
    max_epochs : limite máximo de épocas (segurança)
    Usa os mesmos pesos iniciais do treino padrão.
    """
    n_samples = X.shape[0]
    n_outputs = d.shape[1]

    W = [w.copy() for w in W_init]
    dW_prev = [np.zeros_like(w) for w in W]
    lista_eqm = []
    epocas = 0
    eqm_anterior = float('inf')
    t_start = time.perf_counter()

    while True:
        eqm_epoch = 0.0

        for i in range(n_samples):
            x_i = X[i].reshape(-1, 1)
            d_i = d[i].reshape(-1, 1)

            activations = forward(x_i, W)
            y_out = activations[-1]

            erro = d_i - y_out
            eqm_epoch += np.sum(erro ** 2)

            deltas = [erro * sigmoid_deriv(y_out)]
            for l in range(len(W) - 2, -1, -1):
                W_nobias = W[l + 1][:, 1:]
                delta_next = deltas[0]
                y_l = activations[l + 1]
                delta_l = (W_nobias.T @ delta_next) * sigmoid_deriv(y_l)
                deltas.insert(0, delta_l)

            # ΔW(t) = η·δ·x + α·ΔW(t-1)
            for l in range(len(W)):
                a_bias = np.vstack([np.array([[-1.0]]), activations[l]])
                dW = eta * (deltas[l] @ a_bias.T) + alpha * dW_prev[l]
                W[l] += dW
                dW_prev[l] = dW

        eqm_atual = eqm_epoch / (n_samples * n_outputs)
        lista_eqm.append(eqm_atual)
        epocas += 1

        # Critério de parada: |ΔEQM| <= ε
        if abs(eqm_atual - eqm_anterior) <= precisao or epocas >= max_epochs:
            break
        eqm_anterior = eqm_atual

    tempo_s = time.perf_counter() - t_start
    return W, epocas, lista_eqm, tempo_s

# ---------------------------------------------------------------------------
# Predição
# ---------------------------------------------------------------------------

def predict(X, weights):
    """Retorna array (n_samples, n_outputs) com saídas contínuas da rede."""
    preds = []
    for i in range(X.shape[0]):
        x_i = X[i].reshape(-1, 1)
        activations = forward(x_i, weights)
        preds.append(activations[-1].flatten())
    return np.array(preds)

# ---------------------------------------------------------------------------
# Pós-processamento — arredondamento simétrico
# ---------------------------------------------------------------------------

def postprocess(y_real):
    """Arredondamento simétrico: int(y + 0.5) → {0, 1}"""
    return np.array([[int(v + 0.5) for v in row] for row in y_real])

# ---------------------------------------------------------------------------
# Taxa de acerto
# ---------------------------------------------------------------------------

def taxa_acerto(y_int, d):
    """
    Compara vetores de saída. Considera acerto apenas quando o padrão
    completo (todos os 3 bits) coincide.
    """
    acertos = np.all(y_int == d, axis=1).sum()
    return (acertos / len(d)) * 100.0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Carregar dados ---
    train_data = np.genfromtxt(os.path.join(script_dir, 'treinamento.csv'),
                                delimiter=',', skip_header=1)
    test_data  = np.genfromtxt(os.path.join(script_dir, 'teste.csv'),
                                delimiter=',', skip_header=1)

    X_train = train_data[:, :4]
    d_train = train_data[:, 4:]   # colunas d1, d2, d3

    X_test  = test_data[:, :4]
    d_test  = test_data[:, 4:].astype(int)

    print(f"Amostras de treinamento : {X_train.shape[0]}")
    print(f"Amostras de teste       : {X_test.shape[0]}")

    # --- Topologia ---
    # 4 entradas → 15 neurônios ocultos → 3 saídas
    topology = [4, 15, 3]
    eta      = 0.1
    alpha    = 0.9
    precisao = 1e-6
    seed     = 42

    # --- Pesos iniciais (uma única seed, compartilhada pelos dois treinos) ---
    W_init = init_weights(topology, seed=seed)
    print(f"\nTopologia        : {topology}")
    print(f"Seed             : {seed}")
    print(f"Taxa de aprendizado (η)  : {eta}")
    print(f"Fator de momentum  (α)   : {alpha}")
    print(f"Precisão (ε)             : {precisao}")

    print("\nPesos iniciais W1 (shape):", W_init[0].shape)
    print("Pesos iniciais W2 (shape):", W_init[1].shape)
    print("W1[0,:5] =", W_init[0][0, :5])

    # -----------------------------------------------------------------------
    # Item 1 — Backpropagation Padrão
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("ITEM 1 — Backpropagation Padrão")
    print("="*60)
    W_bp, epocas_bp, eqm_bp, tempo_bp = train_bp(
        X_train, d_train, eta, precisao, W_init
    )
    print(f"Épocas        : {epocas_bp}")
    print(f"EQM final     : {eqm_bp[-1]:.8f}")
    print(f"Tempo         : {tempo_bp:.4f} s")

    # -----------------------------------------------------------------------
    # Item 2 — Backpropagation com Momentum
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("ITEM 2 — Backpropagation com Momentum (α=0.9)")
    print("="*60)
    W_mom, epocas_mom, eqm_mom, tempo_mom = train_bp_momentum(
        X_train, d_train, eta, alpha, precisao, W_init
    )
    print(f"Épocas        : {epocas_mom}")
    print(f"EQM final     : {eqm_mom[-1]:.8f}")
    print(f"Tempo         : {tempo_mom:.4f} s")

    # -----------------------------------------------------------------------
    # Item 3 — Gráfico EQM × Época (subplots)
    # -----------------------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(range(1, epocas_bp + 1), eqm_bp, color='royalblue', linewidth=1.5)
    axs[0].set_title('Backpropagation Padrão — EQM × Época', fontsize=13)
    axs[0].set_xlabel('Época')
    axs[0].set_ylabel('EQM')
    axs[0].grid(True, alpha=0.4)
    axs[0].annotate(f'Épocas: {epocas_bp}\nEQM final: {eqm_bp[-1]:.6f}',
                    xy=(0.98, 0.85), xycoords='axes fraction',
                    ha='right', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    axs[1].plot(range(1, epocas_mom + 1), eqm_mom, color='tomato', linewidth=1.5)
    axs[1].set_title('Backpropagation com Momentum (α=0.9) — EQM × Época', fontsize=13)
    axs[1].set_xlabel('Época')
    axs[1].set_ylabel('EQM')
    axs[1].grid(True, alpha=0.4)
    axs[1].annotate(f'Épocas: {epocas_mom}\nEQM final: {eqm_mom[-1]:.6f}',
                    xy=(0.98, 0.85), xycoords='axes fraction',
                    ha='right', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    plt.tight_layout()
    grafico_path = os.path.join(script_dir, 'grafico_eqm.png')
    plt.savefig(grafico_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {grafico_path}")

    # -----------------------------------------------------------------------
    # Items 4 & 5 — Pós-processamento e Validação
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("ITENS 4 & 5 — Pós-processamento e Validação")
    print("="*60)

    for label, W_trained in [("Padrão", W_bp), ("Momentum", W_mom)]:
        y_cont  = predict(X_test, W_trained)
        y_int   = postprocess(y_cont)
        acerto  = taxa_acerto(y_int, d_test)

        print(f"\n--- {label} ---")
        print(f"{'Amostra':>8} | {'d1 d2 d3':>8} | {'y1 y2 y3':>8} | {'Acerto':>6}")
        print("-" * 46)
        for i in range(len(d_test)):
            d_str = ' '.join(str(v) for v in d_test[i])
            y_str = ' '.join(str(v) for v in y_int[i])
            ok = "✓" if np.all(y_int[i] == d_test[i]) else "✗"
            print(f"{i+1:>8} | {d_str:>8} | {y_str:>8} | {ok:>6}")
        print(f"\nTaxa de acerto ({label}): {acerto:.2f}%")

    # -----------------------------------------------------------------------
    # Sumário final (para montagem do respostas.md)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMÁRIO")
    print("="*60)
    print(f"{'Algoritmo':<30} {'Épocas':>8} {'EQM Final':>14} {'Tempo (s)':>12}")
    print("-" * 70)
    print(f"{'Backpropagation Padrão':<30} {epocas_bp:>8} {eqm_bp[-1]:>14.8f} {tempo_bp:>12.4f}")
    print(f"{'Backpropagation c/ Momentum':<30} {epocas_mom:>8} {eqm_mom[-1]:>14.8f} {tempo_mom:>12.4f}")

    # Salvar sumário de resultados para uso no respostas.md
    resultados = {
        'epocas_bp':  epocas_bp,
        'eqm_bp':     eqm_bp[-1],
        'tempo_bp':   tempo_bp,
        'epocas_mom': epocas_mom,
        'eqm_mom':    eqm_mom[-1],
        'tempo_mom':  tempo_mom,
        'W_init':     W_init,
        'W_bp':       W_bp,
        'W_mom':      W_mom,
    }

    # Validação para respostas.md
    y_cont_bp  = predict(X_test, W_bp)
    y_int_bp   = postprocess(y_cont_bp)
    acerto_bp  = taxa_acerto(y_int_bp, d_test)

    y_cont_mom  = predict(X_test, W_mom)
    y_int_mom   = postprocess(y_cont_mom)
    acerto_mom  = taxa_acerto(y_int_mom, d_test)

    resultados['y_int_bp']  = y_int_bp
    resultados['y_int_mom'] = y_int_mom
    resultados['acerto_bp'] = acerto_bp
    resultados['acerto_mom'] = acerto_mom

    return resultados, d_test


if __name__ == '__main__':
    main()
