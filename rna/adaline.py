import numpy as np
import matplotlib.pyplot as plt

# Carregar dados de treinamento do CSV (colunas: x1, x2, x3, x4, d)
treinamento_csv = np.genfromtxt('treinamento.csv', delimiter=',', skip_header=1)
treinamento_data = treinamento_csv.tolist()

# Carregar dados de teste do CSV (colunas: x1, x2, x3, x4)
teste_csv = np.genfromtxt('teste.csv', delimiter=',', skip_header=1)
teste_data = teste_csv.tolist()

# Parâmetros
eta = 0.0025
precisao = 1e-6

# Preparar X e d (Treinamento)
X_train = []
d_train = []
for row in treinamento_data:
    # x0 = -1 (bias)
    X_train.append([-1.0, row[0], row[1], row[2], row[3]])
    d_train.append(row[4])
X_train = np.array(X_train)
d_train = np.array(d_train)

# Preparar X_test
X_test = []
for row in teste_data:
    X_test.append([-1.0, row[0], row[1], row[2], row[3]])
X_test = np.array(X_test)

def train_adaline(X, d, eta, precisao, seed):
    np.random.seed(seed)
    w = np.random.rand(5)  # Pesos iniciais entre 0 e 1
    w_initial = w.copy()
    
    n_samples = len(X)
    eqm_anterior = float('inf')
    eqms = []
    epoca = 0
    
    while True:
        epoca += 1
        eqm_atual = 0.0
        
        # Opcional: embaralhar em cada época, mas pelo enunciado, podemos manter em ordem.
        # Regra Delta
        for i in range(n_samples):
            u = np.dot(X[i], w)
            erro = d[i] - u
            w = w + eta * erro * X[i]
            
        # Calcular EQM para a época
        for i in range(n_samples):
            u = np.dot(X[i], w)
            erro = d[i] - u
            eqm_atual += erro ** 2
        eqm_atual = eqm_atual / n_samples
        eqms.append(eqm_atual)
        
        if abs(eqm_atual - eqm_anterior) <= precisao:
            break
            
        eqm_anterior = eqm_atual
        
        # Limite de segurança para evitar loop infinito
        if epoca > 50000:
            break
            
    return w_initial, w, epoca, eqms

def predict(X, w):
    y_pred = []
    for i in range(len(X)):
        u = np.dot(X[i], w)
        # Classificar como 1 ou -1
        y = 1 if u >= 0 else -1
        y_pred.append(y)
    return y_pred

resultados_treinamentos = []
eqms_graficos = []

# Realizar 5 treinamentos
for i in range(5):
    w_init, w_final, epocas, eqms = train_adaline(X_train, d_train, eta, precisao, seed=42+i)
    resultados_treinamentos.append((w_init, w_final, epocas))
    if i < 2:
        eqms_graficos.append(eqms)

# Gerar gráfico para os 2 primeiros
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eqms_graficos[0]) + 1), eqms_graficos[0], label='Treinamento 1 (T1)')
plt.plot(range(1, len(eqms_graficos[1]) + 1), eqms_graficos[1], label='Treinamento 2 (T2)')
plt.title('Erro Quadrático Médio (EQM) vs Épocas')
plt.xlabel('Épocas')
plt.ylabel('EQM')
plt.legend()
plt.grid(True)
plt.savefig('grafico_eqm.png')
plt.close()

# Classificar o conjunto de testes com as 5 redes
predicoes_teste = []
for i in range(5):
    w_final = resultados_treinamentos[i][1]
    y_pred = predict(X_test, w_final)
    predicoes_teste.append(y_pred)
predicoes_teste = np.array(predicoes_teste).T  # Transpor para ter (Amostras, 5 predições)

# Imprimir resultados para criar o MD depois
print("Tabela 1: Pesos Iniciais, Finais e Épocas")
for i, res in enumerate(resultados_treinamentos):
    w_i = [f"{x:.4f}" for x in res[0]]
    w_f = [f"{x:.4f}" for x in res[1]]
    print(f"Treinamento {i+1}:")
    print(f"  Inicial: {w_i}")
    print(f"  Final:   {w_f}")
    print(f"  Épocas:  {res[2]}")
    
print("\nTabela 2: Predições")
for i, pred in enumerate(predicoes_teste):
    print(f"Amostra {i+1}: {list(pred)}")
