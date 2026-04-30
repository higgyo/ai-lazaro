import pandas as pd
import numpy as np
from perceptron import Perceptron
from plotter import plot_training_interactive, plot_errors_interactive

def main():
    # 1. Carregar Dados
    print("Carregando os dados de treinamento e teste...")
    try:
        df_treino = pd.read_csv('treinamento.csv')
        df_teste = pd.read_csv('teste.csv')
    except Exception as e:
        print(f"Erro ao ler os arquivos CSV: {e}")
        return
        
    X_train = df_treino[['x1', 'x2', 'x3']].values
    d_train = df_treino['d'].values
    
    X_test = df_teste[['x1', 'x2', 'x3']].values
    
    # 2. Inicializar variáveis para armazenar resultados
    resultados_treino = []
    resultados_teste = []
    
    # 3. Executar 5 Treinamentos
    for i in range(1, 6):
        # Reiniciar semente opcional, mas vamos deixar aleatório mesmo a cada laço
        np.random.seed(i * 42) # Apenas para garantir que não serão iguais, mas reprodutíveis
        
        p = Perceptron(learning_rate=0.01)
        w_inicial = p.weights.copy()
        
        epochs = p.fit(X_train, d_train)
        w_final = p.weights.copy()
        
        # Guardar para a tabela 1
        resultados_treino.append({
            'Treinamento': f"{i}º (T{i})",
            'w0_ini': w_inicial[3], 'w1_ini': w_inicial[0], 'w2_ini': w_inicial[1], 'w3_ini': w_inicial[2],
            'w0_fin': w_final[3], 'w1_fin': w_final[0], 'w2_fin': w_final[1], 'w3_fin': w_final[2],
            'Epocas': epochs
        })
        
        # Testar as amostras para a tabela 2
        y_preds = [p.predict(x) for x in X_test]
        resultados_teste.append(y_preds)
        
        # 4. Plotar gráfico iterativo
        print(f"Gerando gráfico para o treinamento {i} (Épocas: {epochs})...")
        plot_training_interactive(X_train, d_train, p.weights_history, i)
        plot_errors_interactive(p.errors_history, i)

    # 5. Formatar e exibir Tabela de Treinamentos
    print("\n" + "="*80)
    print("TABELA 1 - RESULTADOS DOS 5 TREINAMENTOS")
    print("="*80)
    df_res_treino = pd.DataFrame(resultados_treino)
    # Reordenar colunas conforme solicitado: w0, w1, w2, w3 (inicial), w0, w1, w2, w3 (final), epocas
    cols_ordem = ['Treinamento', 'w0_ini', 'w1_ini', 'w2_ini', 'w3_ini', 'w0_fin', 'w1_fin', 'w2_fin', 'w3_fin', 'Epocas']
    df_res_treino = df_res_treino[cols_ordem]
    print(df_res_treino.to_string(index=False, float_format="%.4f"))

    # 6. Formatar e exibir Tabela de Testes
    print("\n" + "="*80)
    print("TABELA 2 - CLASSIFICAÇÃO DAS AMOSTRAS DE ÓLEO")
    print("="*80)
    
    # Transpor resultados_teste para que fiquem alinhados com as amostras
    # resultados_teste é uma lista de 5 listas com 10 predições cada
    res_teste_t = np.array(resultados_teste).T
    
    df_res_teste = df_teste.copy()
    for i in range(5):
        df_res_teste[f'y(T{i+1})'] = res_teste_t[:, i]
        
    print(df_res_teste.to_string(index=False, float_format="%.4f"))
    print("="*80)

if __name__ == "__main__":
    main()
