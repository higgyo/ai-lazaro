# Respostas - Módulo PMC 1

## 1. Descrição da Rede PMC

A arquitetura da rede Perceptron Multicamadas (PMC) projetada para estimar a energia absorvida do sistema de ressonância magnética possui a seguinte configuração:

*   **Topologia da Rede**:
    *   **Camada de Entrada**: 3 neurônios (entradas $x_1, x_2, x_3$).
    *   **Camada Oculta**: 10 neurônios.
    *   **Camada de Saída**: 1 neurônio (saída $y$).
*   **Função de Ativação**: Logística (aplicada em todos os neurônios da camada oculta e da camada de saída).
*   **Algoritmo de Aprendizagem**: Backpropagation (Regra Delta Generalizada).
*   **Taxa de Aprendizado ($\eta$)**: 0.1
*   **Precisão ($\epsilon$)**: $10^{-6}$

## 2. Tabela de Treinamentos

A rede foi treinada 5 vezes com inicializações de pesos diferentes (valores aleatórios entre 0 e 1, mudando a _seed_ para garantir matrizes distintas a cada execução). Por se tratar de matrizes de pesos (Camada Oculta $W_1$ com dimensão $10 \times 4$ e Camada de Saída $W_2$ com dimensão $1 \times 11$), representamos as matrizes através de suas respectivas dimensões, pois seria inviável registrar cada peso individualmente.

| Treinamento | Pesos Iniciais ($W_1$ e $W_2$) | Pesos Finais ($W_1$ e $W_2$) | Erro Quadrático Médio (EQM) | Número de Épocas |
| :--- | :--- | :--- | :--- | :--- |
| **1º (T1)** | Aleatórios (seed=42) [Dimensões: 10x4 e 1x11] | Matrizes ajustadas (T1) | 0.001575 | 126 |
| **2º (T2)** | Aleatórios (seed=43) [Dimensões: 10x4 e 1x11] | Matrizes ajustadas (T2) | 0.001617 | 147 |
| **3º (T3)** | Aleatórios (seed=44) [Dimensões: 10x4 e 1x11] | Matrizes ajustadas (T3) | 0.001558 | 163 |
| **4º (T4)** | Aleatórios (seed=45) [Dimensões: 10x4 e 1x11] | Matrizes ajustadas (T4) | 0.001572 | 184 |
| **5º (T5)** | Aleatórios (seed=46) [Dimensões: 10x4 e 1x11] | Matrizes ajustadas (T5) | 0.001575 | 168 |

## 3. Gráfico EQM x Época

Para os dois treinamentos com o maior número de épocas (T4 com 184 épocas e T5 com 168 épocas), foram gerados os gráficos do Erro Quadrático Médio (EQM) ao longo das épocas:

![Gráfico EQM x Épocas](/home/alunos/Desktop/ai-lazaro/pmc/pmc_1/grafico_eqm.png)

## 4. Análise das Variações

**Por que tanto o erro quadrático médio quanto o número de épocas variam de treinamento para treinamento?**

Diferentemente de redes com uma única camada e ativação linear (como o ADALINE), cujo cálculo do erro define uma superfície convexa (com formato de um único paraboloide hiperdimensional) possuindo apenas um **mínimo global** único para o qual todas as inicializações convergem, as Redes Perceptron Multicamadas (PMC) apresentam uma superfície de erro altamente não-convexa.

A combinação de **múltiplas camadas** de pesos com a aplicação de **funções de ativação não-lineares** (como a função logística) cria uma superfície de erro complexa, repleta de vales, platôs e, fundamentalmente, **múltiplos mínimos locais**.

O algoritmo de *Backpropagation* baseia-se na descida do gradiente. A posição inicial na superfície de erro é ditada pelos valores dos **pesos iniciais gerados aleatoriamente**. Uma vez que cada treinamento (T1 a T5) inicia em um ponto diferente dessa superfície (devido a diferentes sementes aleatórias), o gradiente guiou a rede por caminhos topológicos distintos. Consequentemente:

1.  A rede atingiu **mínimos locais diferentes**, resultando em **EQMs finais ligeiramente diferentes** e matrizes de pesos finais também distintas.
2.  A distância a ser percorrida na superfície de erro e a complexidade do caminho (como passar por vales rasos onde o gradiente é muito pequeno) variaram em cada inicialização, justificando a **diferença no número de épocas** necessárias para que o algoritmo atingisse o critério de parada (variação de erro inferior a $10^{-6}$).

## 5. Tabela de Validação

Aplicando os conjuntos de matrizes de pesos finais no conjunto de testes para cada treinamento:

| Treinamento | Erro Relativo Médio (%) | Variância do Erro Relativo |
| :--- | :--- | :--- |
| **T1** | 2.3397% | 2.0920 |
| **T2** | 2.0208% | 2.5400 |
| **T3** | 2.0011% | 2.2677 |
| **T4** | 1.7531% | 2.4176 |
| **T5** | 1.7868% | 2.5402 |

## 6. Melhor Generalização

Com base na Tabela de Validação do conjunto de testes, a configuração que deve ser selecionada para o sistema de ressonância magnética é a **T4**.

**Justificativa:** A rede treinada no experimento T4 demonstrou o **menor Erro Relativo Médio (1.7531%)** na estimativa da variável desejada (energia absorvida $y$) para dados de teste que não foram vistos durante o treinamento. A obtenção do menor erro no conjunto de teste é o indicativo mais forte de que esta rede foi a que obteve a **melhor capacidade de generalização**, conseguindo abstrair corretamente os padrões de relacionamento não-linear entre as grandezas de entrada $\{x_1, x_2, x_3\}$ e a saída, em vez de apenas decorar os dados de treinamento (*overfitting*). A variância do T4 é comparável aos demais (2.4176), o que é aceitável e compensa perfeitamente o ganho em precisão média.
