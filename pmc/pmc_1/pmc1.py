import numpy as np
import matplotlib.pyplot as plt

def logistic(v):
    return 1 / (1 + np.exp(-v))

def logistic_derivative(y):
    # If y is the output of the logistic function:
    # f'(v) = f(v) * (1 - f(v)) = y * (1 - y)
    return y * (1 - y)

def train_pmc(X, d, eta, precisao, seed):
    np.random.seed(seed)
    
    n_samples, n_inputs = X.shape
    n_hidden = 10
    n_outputs = 1
    
    # Initialize weights between 0 and 1
    # W1 is for hidden layer (n_inputs x n_hidden) -> Wait, inputs + bias
    # Let's add bias to inputs
    X_bias = np.c_[np.ones(n_samples) * -1, X]  # using -1 for bias input, common convention, or 1. Let's use 1 for bias
    X_bias = np.c_[np.ones(n_samples), X] # Wait, standard bias is usually input 1 or -1. Using 1.
    
    W1 = np.random.rand(n_hidden, n_inputs + 1)
    W2 = np.random.rand(n_outputs, n_hidden + 1)
    
    W1_initial = W1.copy()
    W2_initial = W2.copy()
    
    eqm_anterior = float('inf')
    eqm_atual = 0
    epoca = 0
    lista_eqm = []
    
    while True:
        eqm_epoch = 0
        
        for i in range(n_samples):
            # Forward pass
            x_i = X_bias[i].reshape(-1, 1) # (4, 1)
            
            v1 = np.dot(W1, x_i) # (10, 1)
            y1 = logistic(v1) # (10, 1)
            
            y1_bias = np.vstack([[[1]], y1]) # (11, 1)
            
            v2 = np.dot(W2, y1_bias) # (1, 1)
            y2 = logistic(v2) # (1, 1)
            
            # Error
            erro = d[i] - y2[0, 0]
            eqm_epoch += erro ** 2
            
            # Backward pass
            # Output layer delta
            delta2 = erro * logistic_derivative(y2) # (1, 1)
            
            # Hidden layer delta
            # W2[:, 1:] removes the bias weight for backprop
            delta1 = np.dot(W2[:, 1:].T, delta2) * logistic_derivative(y1) # (10, 1)
            
            # Update weights
            W2 += eta * np.dot(delta2, y1_bias.T)
            W1 += eta * np.dot(delta1, x_i.T)
            
        eqm_atual = eqm_epoch / n_samples
        lista_eqm.append(eqm_atual)
        epoca += 1
        
        if abs(eqm_atual - eqm_anterior) <= precisao:
            break
            
        eqm_anterior = eqm_atual
        
    return W1_initial, W2_initial, W1, W2, epoca, lista_eqm

def predict_pmc(X, W1, W2):
    n_samples = X.shape[0]
    X_bias = np.c_[np.ones(n_samples), X]
    predictions = []
    for i in range(n_samples):
        x_i = X_bias[i].reshape(-1, 1)
        v1 = np.dot(W1, x_i)
        y1 = logistic(v1)
        y1_bias = np.vstack([[[1]], y1])
        v2 = np.dot(W2, y1_bias)
        y2 = logistic(v2)
        predictions.append(y2[0, 0])
    return np.array(predictions)

def main():
    # Carregar dados
    train_data = np.genfromtxt('treinamento.csv', delimiter=',', skip_header=1)
    test_data = np.genfromtxt('teste.csv', delimiter=',', skip_header=1)
    
    X_train = train_data[:, :3]
    d_train = train_data[:, 3]
    
    X_test = test_data[:, :3]
    d_test = test_data[:, 3]
    
    eta = 0.1
    precisao = 1e-6
    seeds = [42, 43, 44, 45, 46]
    
    resultados = []
    
    for i, seed in enumerate(seeds):
        print(f"Treinando modelo {i+1} (seed={seed})...")
        W1_init, W2_init, W1_final, W2_final, epocas, lista_eqm = train_pmc(X_train, d_train, eta, precisao, seed)
        resultados.append({
            'T': i + 1,
            'epocas': epocas,
            'lista_eqm': lista_eqm,
            'W1_init': W1_init,
            'W2_init': W2_init,
            'W1_final': W1_final,
            'W2_final': W2_final,
            'eqm_final': lista_eqm[-1]
        })
        print(f"T{i+1}: Epocas={epocas}, EQM={lista_eqm[-1]:.6f}")
        
    # Identificar os 2 treinamentos com maior numero de epocas
    resultados_ordenados = sorted(resultados, key=lambda x: x['epocas'], reverse=True)
    top2 = resultados_ordenados[:2]
    
    # Gerar grafico
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    for i, res in enumerate(top2):
        axs[i].plot(range(1, res['epocas'] + 1), res['lista_eqm'])
        axs[i].set_title(f"Treinamento {res['T']} - EQM x Época")
        axs[i].set_xlabel('Época')
        axs[i].set_ylabel('EQM')
        axs[i].grid(True)
        
    plt.tight_layout()
    plt.savefig('grafico_eqm.png')
    print("\nGráfico salvo como 'grafico_eqm.png'")
    
    # Validacao
    print("\n--- Validação ---")
    melhor_treinamento = None
    menor_erro = float('inf')
    
    for res in resultados:
        y_pred = predict_pmc(X_test, res['W1_final'], res['W2_final'])
        erros_relativos = np.abs((d_test - y_pred) / d_test) * 100
        erro_relativo_medio = np.mean(erros_relativos)
        variancia_erro = np.var(erros_relativos)
        
        res['erro_medio'] = erro_relativo_medio
        res['variancia'] = variancia_erro
        
        print(f"T{res['T']}: Erro Relativo Médio = {erro_relativo_medio:.4f}%, Variância = {variancia_erro:.4f}")
        
        if erro_relativo_medio < menor_erro:
            menor_erro = erro_relativo_medio
            melhor_treinamento = res['T']
            
    print(f"\nMelhor treinamento: T{melhor_treinamento}")

if __name__ == '__main__':
    main()
