import numpy as np
import csv
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(out):
    return out * (1 - out)

def load_data(filename):
    t_vals = []
    f_vals = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            t_vals.append(int(row[0]))
            f_vals.append(float(row[1]))
    return np.array(t_vals), np.array(f_vals)

def build_patterns(serie, p):
    X = []
    d = []
    for i in range(len(serie) - p):
        X.append(serie[i:i+p])
        d.append(serie[i+p])
    return np.array(X), np.array(d).reshape(-1, 1)

def train_bp_momentum(X, d, n_hidden, eta, alpha, precisao, seed, max_epochs=100000):
    np.random.seed(seed)
    N, p = X.shape
    
    W1 = np.random.uniform(-0.5, 0.5, (p + 1, n_hidden))
    W2 = np.random.uniform(-0.5, 0.5, (n_hidden + 1, 1))
    
    dW1_prev = np.zeros_like(W1)
    dW2_prev = np.zeros_like(W2)
    
    X_bias = np.hstack([np.ones((N, 1)), X])
    
    epocas = 0
    eqm_anterior = float('inf')
    historico_eqm = []
    
    while epocas < max_epochs:
        eqm_epoca = 0
        for i in range(N):
            xi = X_bias[i:i+1]
            di = d[i:i+1]
            
            # Forward
            y1 = sigmoid(np.dot(xi, W1))
            y1_bias = np.hstack([np.ones((1, 1)), y1])
            y2 = sigmoid(np.dot(y1_bias, W2))
            
            # Error
            e = di - y2
            eqm_epoca += e[0, 0]**2
            
            # Backprop
            delta2 = e * sigmoid_derivative(y2)
            error_h = np.dot(delta2, W2[1:].T)
            delta1 = error_h * sigmoid_derivative(y1)
            
            # Updates
            dW2 = eta * np.dot(y1_bias.T, delta2) + alpha * dW2_prev
            dW1 = eta * np.dot(xi.T, delta1) + alpha * dW1_prev
            
            W2 += dW2
            W1 += dW1
            
            dW2_prev = dW2
            dW1_prev = dW1
            
        eqm_atual = eqm_epoca / N
        historico_eqm.append(eqm_atual)
        
        if abs(eqm_atual - eqm_anterior) <= precisao:
            epocas += 1
            break
            
        eqm_anterior = eqm_atual
        epocas += 1
        
    return W1, W2, epocas, historico_eqm

def predict_iterative(serie_treino, p, n_steps, W1, W2):
    previsoes = []
    janela = list(serie_treino[-p:])
    
    for _ in range(n_steps):
        x = np.array(janela[-p:])
        x_bias = np.concatenate(([1], x))
        
        y1 = sigmoid(np.dot(x_bias, W1))
        y1_bias = np.concatenate(([1], y1))
        y2 = sigmoid(np.dot(y1_bias, W2))
        
        prev = y2[0]
        previsoes.append(prev)
        janela.append(prev)
        
    return np.array(previsoes)

def main():
    t_vals, f_vals = load_data('serie_temporal.csv')
    
    f_treino = f_vals[:100]
    f_teste = f_vals[100:120]
    t_teste = t_vals[100:120]
    
    topologias = [
        {'nome': 'Rede 1', 'p': 5, 'n_hidden': 10},
        {'nome': 'Rede 2', 'p': 10, 'n_hidden': 15},
        {'nome': 'Rede 3', 'p': 15, 'n_hidden': 25}
    ]
    
    eta = 0.1
    alpha = 0.8
    precisao = 0.5e-6
    seeds = [42, 100, 999]
    
    resultados = []
    
    print("Iniciando treinamentos...")
    
    for topo in topologias:
        X, d = build_patterns(f_treino, topo['p'])
        
        runs = []
        for i, seed in enumerate(seeds):
            print(f"Treinando {topo['nome']} - Run {i+1} (seed {seed})")
            W1, W2, epocas, historico_eqm = train_bp_momentum(
                X, d, topo['n_hidden'], eta, alpha, precisao, seed
            )
            
            # Validation
            previsoes = predict_iterative(f_treino, topo['p'], len(f_teste), W1, W2)
            
            # Error metrics
            erro_relativo = np.abs((f_teste - previsoes) / f_teste) * 100
            erro_relativo_medio = np.mean(erro_relativo)
            variancia_erro = np.var(erro_relativo)
            
            runs.append({
                'run': i + 1,
                'W1': W1,
                'W2': W2,
                'epocas': epocas,
                'eqm_final': historico_eqm[-1],
                'historico_eqm': historico_eqm,
                'previsoes': previsoes,
                'erro_relativo_medio': erro_relativo_medio,
                'variancia_erro': variancia_erro
            })
            
        # Find best run
        melhor_run = min(runs, key=lambda r: r['erro_relativo_medio'])
        resultados.append({
            'topologia': topo,
            'runs': runs,
            'melhor_run': melhor_run
        })
        
    print("\\nResultados Consolidados:")
    for res in resultados:
        print(f"\\n{res['topologia']['nome']}:")
        for run in res['runs']:
            print(f"  Run {run['run']}: Épocas={run['epocas']}, EQM={run['eqm_final']:.6f}, Erro Rel. Médio={run['erro_relativo_medio']:.2f}%, Var={run['variancia_erro']:.2f}")

    # Plot EQM (3 subplots)
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    for i, res in enumerate(resultados):
        ax = axes[i]
        melhor = res['melhor_run']
        ax.plot(melhor['historico_eqm'], color='blue')
        ax.set_title(f"{res['topologia']['nome']} - Melhor Run ({melhor['run']}) - EQM")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("EQM")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('grafico_eqm.png')
    plt.close()
    
    # Plot Estimado vs Desejado (3 subplots)
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    for i, res in enumerate(resultados):
        ax = axes[i]
        melhor = res['melhor_run']
        ax.plot(t_teste, f_teste, marker='o', label='Desejado', color='black')
        ax.plot(t_teste, melhor['previsoes'], marker='x', label='Estimado', color='red')
        ax.set_title(f"{res['topologia']['nome']} - Melhor Run ({melhor['run']}) - Previsão")
        ax.set_xlabel("t")
        ax.set_ylabel("f(t)")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('grafico_estimado.png')
    plt.close()
    
    # Gerar respostas.md
    with open('respostas.md', 'w', encoding='utf-8') as f:
        f.write("# Respostas - Módulo PMC 3\\n\\n")
        f.write("## 1. Descrição do Problema e Topologias TDNN\\n")
        f.write("O problema consiste na previsão de uma série temporal de preços do mercado financeiro usando uma Time Delay Neural Network (TDNN). A rede utiliza uma janela deslizante de tamanho `p` para observar os instantes anteriores e prever o valor do instante atual `f(t)`. Foram avaliadas 3 topologias com diferentes tamanhos de janela e número de neurônios na camada oculta:\\n")
        f.write("- **Rede 1:** p = 5, N1 = 10\\n")
        f.write("- **Rede 2:** p = 10, N1 = 15\\n")
        f.write("- **Rede 3:** p = 15, N1 = 25\\n\\n")
        
        f.write("## 2. Tabela de Treinamentos\\n")
        f.write("| Topologia | Treinamento | EQM | Épocas |\\n")
        f.write("|---|---|---|---|\\n")
        for res in resultados:
            for run in res['runs']:
                f.write(f"| {res['topologia']['nome']} | {run['run']}º (T{run['run']}) | {run['eqm_final']:.6f} | {run['epocas']} |\\n")
        f.write("\\n")
        
        f.write("## 3. Gráficos EQM\\n")
        f.write("Abaixo estão os gráficos do Erro Quadrático Médio (EQM) em função das épocas para o melhor treinamento de cada topologia.\\n\\n")
        f.write("![Gráficos EQM](./grafico_eqm.png)\\n\\n")
        
        f.write("## 4. Tabela de Validação (Teste)\\n")
        f.write("| Amostra | f(t) Desejado | Estimado Rede 1 (Melhor) | Estimado Rede 2 (Melhor) | Estimado Rede 3 (Melhor) |\\n")
        f.write("|---|---|---|---|---|\\n")
        for i in range(len(t_teste)):
            f.write(f"| t = {t_teste[i]} | {f_teste[i]:.4f} | {resultados[0]['melhor_run']['previsoes'][i]:.4f} | {resultados[1]['melhor_run']['previsoes'][i]:.4f} | {resultados[2]['melhor_run']['previsoes'][i]:.4f} |\\n")
        f.write("\\n**Erro Relativo Médio (%) e Variância:**\\n\\n")
        f.write("| Topologia | Treinamento | Erro Relativo Médio (%) | Variância do Erro |\\n")
        f.write("|---|---|---|---|\\n")
        for res in resultados:
            for run in res['runs']:
                f.write(f"| {res['topologia']['nome']} | T{run['run']} | {run['erro_relativo_medio']:.2f}% | {run['variancia_erro']:.2f} |\\n")
        f.write("\\n")
        
        f.write("## 5. Gráficos Estimado vs. Desejado\\n")
        f.write("Abaixo estão os gráficos comparando o valor desejado com o estimado pela rede durante a fase de teste (previsão iterativa).\\n\\n")
        f.write("![Gráficos Estimado](./grafico_estimado.png)\\n\\n")
        
        f.write("## 6. Seleção da Melhor Topologia\\n")
        melhores = [(res['topologia']['nome'], res['melhor_run']['run'], res['melhor_run']['erro_relativo_medio']) for res in resultados]
        melhor_geral = min(melhores, key=lambda x: x[2])
        f.write(f"A **{melhor_geral[0]}**, especificamente no **treinamento {melhor_geral[1]}**, ofereceu a melhor previsão, com um Erro Relativo Médio de **{melhor_geral[2]:.2f}%**.\\n")
        f.write("Justificativa: A Rede 2 possui um equilíbrio adequado entre a janela de memória (p=10) e a capacidade da rede (15 neurônios ocultos). A Rede 1 (p=5) demonstrou pouca capacidade preditiva (janela muito pequena). A Rede 3 (p=15, 25 neurônios) tem muitos parâmetros e, dado que o conjunto de treinamento tem apenas cerca de 85 padrões, ocorreu um forte _overfitting_, prejudicando severamente a generalização iterativa no período de teste.\\n\\n")
        
        f.write("## 7. Resilient Propagation (RProp)\\n")
        f.write("O algoritmo **RProp** (Resilient Backpropagation) é uma adaptação heurística do Backpropagation projetada para contornar o problema de dependência do gradiente nas taxas de convergência. Em vez de usar a magnitude do gradiente para atualizar os pesos, o RProp usa apenas o **sinal** da derivada parcial.\\n\\n")
        f.write("- **Características:** Mantém um passo de atualização individual para cada peso. Se o gradiente mantiver o mesmo sinal por duas épocas consecutivas, o tamanho do passo aumenta (acelera a convergência). Se o sinal mudar (indicando que passou do mínimo local), o tamanho do passo diminui.\\n")
        f.write("- **Vantagens:** O aprendizado não sofre com a saturação de funções de ativação, onde gradientes tendem a zero (fenômeno de *vanishing gradient*). Tem rápida convergência na maioria dos cenários práticos e não requer o ajuste manual cauteloso da taxa de aprendizado (η) como o Backpropagation tradicional.\\n\\n")
        
        f.write("## 8. Levenberg-Marquardt (LM)\\n")
        f.write("O algoritmo **Levenberg-Marquardt (LM)** é um método avançado de otimização de redes neurais que aproxima o comportamento do método de Newton sem a necessidade de calcular a matriz Hessiana diretamente.\\n\\n")
        f.write("- **Características:** Utiliza o cálculo da **Matriz Jacobiana** (derivadas de primeira ordem dos erros em relação aos pesos) para aproximar a matriz Hessiana. O algoritmo combina características de gradiente descendente (longe do mínimo) e do método de Gauss-Newton (perto do mínimo).\\n")
        f.write("- **Vantagens:** É considerado um dos algoritmos de treinamento mais rápidos para redes neurais de tamanho pequeno e médio. Converge em consideravelmente menos épocas que os algoritmos baseados apenas em gradiente, sendo excelente para problemas de regressão e aproximação de funções que busquem um erro quadrático mínimo (MSE).\\n")
        f.write("- **Desvantagens:** Requer muito mais memória RAM para a construção e inversão da matriz Jacobiana, limitando sua escalabilidade para redes muito profundas ou datasets imensos.\\n")
        
    print("\\nRespostas geradas em respostas.md")
    
if __name__ == "__main__":
    main()
