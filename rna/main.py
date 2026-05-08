# -*- coding: utf-8 -*-
"""
main.py — ADALINE: Classificação de Sinais Ruidosos para Comutador de Válvulas
==============================================================================

Lab. Inteligência Artificial — CEFET-MG Campus VIII
Referência: Adaline.docx

Atividades executadas:
    1. 5 treinamentos com pesos iniciais aleatórios distintos (seeds únicas).
    2. Tabela de resultados: pesos iniciais, pesos finais e nº de épocas.
    3. Gráficos EQM de T1 e T2 numa mesma figura.
    4. Classificação das 15 amostras de teste para cada treinamento.
    5. Resposta teórica ao item 4 do documento.

Parâmetros:
    η (taxa de aprendizado) = 0.0025
    ε (precisão)           = 1e-6
    Saída -1 → Válvula A | Saída +1 → Válvula B
"""

import os
import sys
import numpy as np
import pandas as pd

from adaline import Adaline
from plotter import plot_eqm_dois_treinamentos, plot_eqm_individual

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.0025
PRECISION = 1e-6
N_TRAININGS = 5
SEEDS = [42, 123, 777, 2024, 9999]   # Seeds distintas → pesos iniciais distintos

# Muda para o diretório do script (para leitura dos CSV e escrita dos HTML)
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 1. Carregar Dados
# ---------------------------------------------------------------------------
def load_data():
    print("Carregando dados de treinamento e teste...")
    try:
        df_treino = pd.read_csv("treinamento.csv")
        df_teste = pd.read_csv("teste.csv")
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        sys.exit(1)

    X_train = df_treino[["x1", "x2", "x3", "x4"]].values
    d_train = df_treino["d"].values
    X_test = df_teste[["x1", "x2", "x3", "x4"]].values

    print(f"  Padrões de treinamento: {len(X_train)}")
    print(f"  Amostras de teste:      {len(X_test)}")
    return X_train, d_train, X_test, df_teste


# ---------------------------------------------------------------------------
# 2. Executar 5 Treinamentos
# ---------------------------------------------------------------------------
def run_trainings(X_train, d_train):
    redes = []
    resultados_treino = []

    for i in range(1, N_TRAININGS + 1):
        seed = SEEDS[i - 1]
        print(f"\n{'=' * 60}")
        print(f"Treinamento {i} (seed={seed})")

        rede = Adaline(learning_rate=LEARNING_RATE, precision=PRECISION, seed=seed)

        # Treinar (initial_weights gravado dentro de fit())
        epocas = rede.fit(X_train, d_train)
        w_ini = rede.initial_weights   # gravado dentro de fit()
        w_fin = rede.weights

        print(f"  Épocas até convergência: {epocas}")
        print(f"  EQM final:               {rede.eqm_history[-1]:.2e}")

        redes.append(rede)
        resultados_treino.append({
            "Treinamento": f"{i}º (T{i})",
            "w0_ini": w_ini[0], "w1_ini": w_ini[1], "w2_ini": w_ini[2],
            "w3_ini": w_ini[3], "w4_ini": w_ini[4],
            "w0_fin": w_fin[0], "w1_fin": w_fin[1], "w2_fin": w_fin[2],
            "w3_fin": w_fin[3], "w4_fin": w_fin[4],
            "Épocas": epocas,
        })

    return redes, resultados_treino


# ---------------------------------------------------------------------------
# 3. Exibir Tabela 1 — Resultados dos 5 Treinamentos
# ---------------------------------------------------------------------------
def print_tabela_treinamentos(resultados_treino):
    print("\n" + "=" * 100)
    print("TABELA 1 — RESULTADOS DOS 5 TREINAMENTOS (ADALINE — Regra Delta)")
    print("=" * 100)
    print(f"  η = {LEARNING_RATE}  |  ε = {PRECISION:.0e}")
    print("=" * 100)

    df = pd.DataFrame(resultados_treino)
    colunas = [
        "Treinamento",
        "w0_ini", "w1_ini", "w2_ini", "w3_ini", "w4_ini",
        "w0_fin", "w1_fin", "w2_fin", "w3_fin", "w4_fin",
        "Épocas",
    ]
    df = df[colunas]

    # Cabeçalho visual
    print(
        f"{'Treinamento':>12} | "
        f"{'w0_ini':>8} {'w1_ini':>8} {'w2_ini':>8} {'w3_ini':>8} {'w4_ini':>8} | "
        f"{'w0_fin':>8} {'w1_fin':>8} {'w2_fin':>8} {'w3_fin':>8} {'w4_fin':>8} | "
        f"{'Épocas':>8}"
    )
    print("-" * 120)
    for _, row in df.iterrows():
        print(
            f"{row['Treinamento']:>12} | "
            f"{row['w0_ini']:>8.4f} {row['w1_ini']:>8.4f} {row['w2_ini']:>8.4f} "
            f"{row['w3_ini']:>8.4f} {row['w4_ini']:>8.4f} | "
            f"{row['w0_fin']:>8.4f} {row['w1_fin']:>8.4f} {row['w2_fin']:>8.4f} "
            f"{row['w3_fin']:>8.4f} {row['w4_fin']:>8.4f} | "
            f"{int(row['Épocas']):>8}"
        )
    print("=" * 100)


# ---------------------------------------------------------------------------
# 4. Classificar Amostras de Teste e Exibir Tabela 2
# ---------------------------------------------------------------------------
def classifica_e_print_tabela_teste(redes, X_test, df_teste):
    print("\n" + "=" * 100)
    print("TABELA 2 — CLASSIFICAÇÃO DAS 15 AMOSTRAS (por Treinamento)")
    print("=" * 100)
    print(f"  Legenda: A = Válvula A (d = -1) | B = Válvula B (d = +1)")
    print("=" * 100)

    # Matriz de classificações: linhas=amostras, colunas=treinamentos
    classificacoes = np.array([
        [rede.classify_label(x) for x in X_test]
        for rede in redes
    ]).T   # shape (15, 5)

    # Cabeçalho
    header = (
        f"{'Amostra':>8} | "
        f"{'x1':>8} {'x2':>8} {'x3':>8} {'x4':>8} | "
        f"{'y(T1)':>6} {'y(T2)':>6} {'y(T3)':>6} {'y(T4)':>6} {'y(T5)':>6}"
    )
    print(header)
    print("-" * 90)

    for i, row in enumerate(X_test):
        labels = classificacoes[i]
        print(
            f"{i + 1:>8} | "
            f"{row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f} | "
            f"{'   '.join(labels)}"
        )

    print("=" * 100)

    # Verificar consistência entre treinamentos
    consistente = all(
        len(set(classificacoes[i])) == 1
        for i in range(len(X_test))
    )
    status = "[OK] Todos os treinamentos convergem para a mesma classificacao." if consistente \
        else "[ATENCAO] Divergencia detectada em pelo menos uma amostra entre treinamentos."
    print(f"\n  {status}")

    return classificacoes


# ---------------------------------------------------------------------------
# 5. Gráficos
# ---------------------------------------------------------------------------
def gerar_graficos(redes):
    print("\n" + "─" * 60)
    print("Gerando gráficos...")

    # Item 3 do documento: T1 e T2 numa mesma folha
    plot_eqm_dois_treinamentos(
        eqm_t1=redes[0].eqm_history,
        eqm_t2=redes[1].eqm_history,
        filename="grafico_eqm_t1_t2.html",
    )

    # Gráficos individuais para todos os 5 treinamentos
    for i, rede in enumerate(redes, start=1):
        plot_eqm_individual(rede.eqm_history, training_index=i)


# ---------------------------------------------------------------------------
# 6. Resposta Teórica — Item 4 do Documento
# ---------------------------------------------------------------------------
def print_resposta_teorica(redes):
    print("\n" + "=" * 100)
    print("ITEM 4 — Por que os pesos finais são praticamente iguais mesmo com épocas diferentes?")
    print("=" * 100)
    print("""
  A Regra Delta (LMS) minimiza o Erro Quadrático Médio (EQM) sobre o conjunto de treinamento.
  A superfície de erro do ADALINE é uma paraboloide convexa no espaço dos pesos — portanto,
  existe um ÚNICO mínimo global W*.

  Independentemente da inicialização (ponto de partida na superfície de erro), o gradiente
  descendente sempre aponta para o mesmo mínimo. Com η pequeno o suficiente para garantir
  convergência (η < 1 / λ_max, onde λ_max é o maior autovalor da matriz de correlação das
  entradas), o algoritmo sempre chega em W*.

  O número de épocas varia porque cada inicialização começa a uma distância diferente de W*
  na superfície de erro: inicializações mais próximas convergem em menos épocas, enquanto
  inicializações mais distantes exigem mais iterações — mas o destino final (W*) é sempre o
  mesmo ponto de mínimo único da superfície quadrática.
""")

    # Verificar empiricamente
    print("  Verificação empírica — Pesos finais dos 5 treinamentos:")
    for i, rede in enumerate(redes, start=1):
        w = rede.weights
        print(f"    T{i}: [{', '.join(f'{wi:+.6f}' for wi in w)}]  (épocas={rede.epochs_trained})")

    std_pesos = np.std([rede.weights for rede in redes], axis=0)
    print(f"\n  Desvio padrão entre pesos finais: [{', '.join(f'{s:.2e}' for s in std_pesos)}]")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" ADALINE — Regra Delta | Lab. IA — CEFET-MG Campus VIII")
    print("=" * 60)

    X_train, d_train, X_test, df_teste = load_data()
    redes, resultados_treino = run_trainings(X_train, d_train)

    print_tabela_treinamentos(resultados_treino)
    classificacoes = classifica_e_print_tabela_teste(redes, X_test, df_teste)
    gerar_graficos(redes)
    print_resposta_teorica(redes)

    print("\n[CONCLUIDO] Execucao finalizada com sucesso!")


if __name__ == "__main__":
    main()
