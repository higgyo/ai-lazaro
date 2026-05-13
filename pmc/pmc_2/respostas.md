# Respostas — Módulo PMC 2

## 1. Descrição da Rede PMC

A rede Perceptron Multicamadas foi projetada para **classificar o tipo de conservante** (A, B ou C) a ser aplicado em bebidas, com base em 4 variáveis de entrada normalizadas.

- **Topologia da Rede**:
  - **Camada de Entrada**: 4 neurônios — $x_1$ (teor de água), $x_2$ (grau de acidez), $x_3$ (temperatura), $x_4$ (tensão superficial).
  - **Camada Oculta**: 15 neurônios.
  - **Camada de Saída**: 3 neurônios — $y_1$, $y_2$, $y_3$ (codificação one-hot).
- **Função de Ativação**: Logística (sigmoid) em todos os neurônios.
- **Algoritmos comparados**: Backpropagation Padrão vs. Backpropagation com Momentum.
- **Taxa de Aprendizado ($\eta$)**: 0.1
- **Fator de Momentum ($\alpha$)**: 0.9 (somente Item 2)
- **Precisão ($\varepsilon$)**: $10^{-6}$
- **Pesos Iniciais**: Aleatórios entre 0 e 1, `seed = 42` (compartilhada por ambos os treinamentos).

**Codificação de saída (one-hot):**

| Conservante | $y_1$ | $y_2$ | $y_3$ |
| :--- | :---: | :---: | :---: |
| Tipo A | 1 | 0 | 0 |
| Tipo B | 0 | 1 | 0 |
| Tipo C | 0 | 0 | 1 |

---

## 2. Tabela Comparativa de Treinamentos

Ambos os treinamentos partem dos **mesmos pesos iniciais** (seed = 42).  
As matrizes $W_1$ têm dimensão $15 \times 5$ (15 neurônios ocultos × 4 entradas + 1 bias) e $W_2$ tem dimensão $3 \times 16$ (3 saídas × 15 neurônios ocultos + 1 bias).

| Algoritmo | Pesos Iniciais | Pesos Finais | EQM Final | Épocas | Tempo de Processamento |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Backpropagation Padrão** | Aleatórios (seed=42) [$W_1$: 15×5; $W_2$: 3×16] | Matrizes ajustadas (BP) | 0.01963994 | 967 | 8.3970 s |
| **Backpropagation com Momentum** | Aleatórios (seed=42) [*mesmos do item anterior*] | Matrizes ajustadas (Momentum) | 0.02219474 | 233 | 2.1610 s |

---

## 3. Gráfico EQM × Época

Os dois subplots abaixo mostram a evolução do Erro Quadrático Médio ao longo das épocas para cada algoritmo:

![Gráfico EQM × Época — Backpropagation Padrão vs. Momentum](/home/alunos/Desktop/ai-lazaro/pmc/pmc_2/grafico_eqm.png)

---

## 4. Tabela de Validação — Conjunto de Teste (18 amostras)

O pós-processamento aplica o critério de **arredondamento simétrico**: $\hat{y} = \lfloor y + 0{,}5 \rfloor \in \{0, 1\}$, convertendo as saídas contínuas da rede em inteiros.

### 4.1 Backpropagation Padrão

| Amostra | $d_1$ | $d_2$ | $d_3$ | $\hat{y}_1$ | $\hat{y}_2$ | $\hat{y}_3$ | Acerto |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1  | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 2  | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 3  | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 4  | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |
| 5  | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 6  | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 7  | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |
| 8  | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |
| 9  | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 10 | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 11 | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |
| 12 | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 13 | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 14 | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 15 | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 16 | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 17 | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 18 | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |

**Taxa de Acerto (Backpropagation Padrão): 100,00%**

### 4.2 Backpropagation com Momentum

| Amostra | $d_1$ | $d_2$ | $d_3$ | $\hat{y}_1$ | $\hat{y}_2$ | $\hat{y}_3$ | Acerto |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1  | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 2  | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 3  | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 4  | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |
| 5  | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 6  | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 7  | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |
| 8  | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |
| 9  | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 10 | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 11 | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |
| 12 | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 13 | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 14 | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 15 | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 16 | 1 | 0 | 0 | 1 | 0 | 0 | ✓ |
| 17 | 0 | 0 | 1 | 0 | 0 | 1 | ✓ |
| 18 | 0 | 1 | 0 | 0 | 1 | 0 | ✓ |

**Taxa de Acerto (Backpropagation com Momentum): 100,00%**

---

## 5. Análise Comparativa

**Por que o backpropagation com momentum converge em menos épocas?**

Ambos os algoritmos partiram dos **mesmos pesos iniciais** e atingiram **100% de taxa de acerto** no conjunto de teste após o pós-processamento. Contudo, observam-se diferenças expressivas no processo de convergência:

| Métrica | BP Padrão | BP c/ Momentum |
| :--- | :---: | :---: |
| **Épocas** | 967 | 233 |
| **Tempo** | 8.40 s | 2.16 s |
| **Redução de épocas** | — | −75.9% |
| **Redução de tempo** | — | −74.3% |
| **Taxa de acerto** | 100% | 100% |

O **momentum** acumula informação direcional das atualizações anteriores de peso por meio do termo $\alpha \cdot \Delta W(t-1)$. Esse efeito de "inércia" produz dois benefícios:

1. **Aceleração em regiões de gradiente estável**: quando o gradiente aponta consistentemente na mesma direção ao longo de várias amostras ou épocas, as atualizações se acumulam, permitindo passos efetivos maiores sem aumentar $\eta$.
2. **Amortecimento de oscilações**: em regiões com gradiente oscilante (fundos de vales estreitos), o momentum cancela parcialmente as variações opostas, estabilizando a trajetória de descida e evitando a lentidão característica do backpropagation padrão nessas regiões.

O resultado observado — **233 épocas vs. 967 épocas** — confirma o comportamento esperado na literatura: o backpropagation com momentum converge significativamente mais rápido, especialmente em superfícies de erro com vales longos e inclinações suaves. O EQM final ligeiramente maior do momentum (0.0222 vs. 0.0196) não se traduz em nenhuma diferença na classificação, já que ambas as redes classificaram corretamente todas as 18 amostras de teste.
