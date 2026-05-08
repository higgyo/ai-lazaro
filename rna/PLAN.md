# Plano de Implementação — ADALINE (Regra Delta)
**Módulo:** `rna/`  
**Documento base:** `rna/Adaline.docx`  
**Data:** 2026-05-07  
**Status:** Pronto para execução

---

## Contexto do Problema

Um sistema industrial usa um único canal de comunicação para acionar duas válvulas (A e B). Sinais codificados com 4 grandezas {x1, x2, x3, x4} são transmitidos com ruído. Uma rede ADALINE deve classificar cada sinal ruidoso e indicar ao comutador se o sinal é para:
- **Válvula A** → saída desejada `d = -1`
- **Válvula B** → saída desejada `d = +1`

---

## Especificação do ADALINE

| Parâmetro | Valor |
|---|---|
| Entradas | 4 (x1, x2, x3, x4) + bias (x0 = +1) |
| Pesos | w0 (bias), w1, w2, w3, w4 — total 5 |
| Algoritmo | Regra Delta (LMS / Widrow-Hoff) |
| Taxa de aprendizado (η) | 0.0025 |
| Precisão (ε) | 10⁻⁶ |
| Critério de parada | EQM < ε |
| Inicialização dos pesos | Aleatória em [0, 1] com seeds distintas por treinamento |
| Número de treinamentos | 5 |
| Função de ativação | Linear (para treinamento) / sign para inferência |
| Saída de classificação | -1 → Válvula A, +1 → Válvula B |

---

## Dados

### Conjunto de Treinamento (35 padrões — Anexo do documento)

```
Padrão | x1      | x2      | x3     | x4      | d
01     | 0.4329  | -1.3719 | 0.7022 | -0.8535 | 1.0
02     | 0.3024  | 0.2286  | 0.8630 | 2.7909  | -1.0
03     | 0.1349  | -0.6445 | 1.0530 | 0.5687  | -1.0
04     | 0.3374  | -1.7163 | 0.3670 | -0.6283 | -1.0
05     | 1.1434  | -0.0485 | 0.6637 | 1.2606  | 1.0
06     | 1.3749  | -0.5071 | 0.4464 | 1.3009  | 1.0
07     | 0.7221  | -0.7587 | 0.7681 | -0.5592 | 1.0
08     | 0.4403  | -0.8072 | 0.5154 | -0.3129 | 1.0
09     | -0.5231 | 0.3548  | 0.2538 | 1.5776  | -1.0
10     | 0.3255  | -2.0000 | 0.7112 | -1.1209 | 1.0
11     | 0.5824  | 1.3915  | -0.2291| 4.1735  | -1.0
12     | 0.1340  | 0.6081  | 0.4450 | 3.2230  | -1.0
13     | 0.1480  | -0.2988 | 0.4778 | 0.8649  | 1.0
14     | 0.7359  | 0.1869  | -0.0872| 2.3584  | 1.0
15     | 0.7115  | -1.1469 | 0.3394 | 0.9573  | -1.0
16     | 0.8251  | -1.2840 | 0.8452 | 1.2382  | -1.0
17     | 0.1569  | 0.3712  | 0.8825 | 1.7633  | 1.0
18     | 0.0033  | 0.6835  | 0.5389 | 2.8249  | -1.0
19     | 0.4243  | 0.8313  | 0.2634 | 3.5855  | -1.0
20     | 1.0490  | 0.1326  | 0.9138 | 1.9792  | 1.0
21     | 1.4276  | 0.5331  | -0.0145| 3.7286  | 1.0
22     | 0.5971  | 1.4865  | 0.2904 | 4.6069  | -1.0
23     | 0.8475  | 2.1479  | 0.3179 | 5.8235  | -1.0
24     | 1.3967  | -0.4171 | 0.6443 | 1.3927  | 1.0
25     | 0.0044  | 1.5378  | 0.6099 | 4.7755  | -1.0
26     | 0.2201  | -0.5668 | 0.0515 | 0.7829  | 1.0
27     | 0.6300  | -1.2480 | 0.8591 | 0.8093  | -1.0
28     | -0.2479 | 0.8960  | 0.0547 | 1.7381  | 1.0
29     | -0.3088 | -0.0929 | 0.8659 | 1.5483  | -1.0
30     | -0.5180 | 1.4974  | 0.5453 | 2.3993  | 1.0
31     | 0.6833  | 0.8266  | 0.0829 | 2.8864  | 1.0
32     | 0.4353  | -1.4066 | 0.4207 | -0.4879 | 1.0
33     | -0.1069 | -3.2329 | 0.1856 | -2.4572 | -1.0
34     | 0.4662  | 0.6261  | 0.7304 | 3.4370  | -1.0
35     | 0.8298  | -1.4089 | 0.3119 | 1.3235  | -1.0
```

### Amostras para Classificação Final (15 padrões — Tabela 2)

```
Amostra | x1     | x2      | x3     | x4
1       | 0.9694 | 0.6909  | 0.4334 | 3.4965
2       | 0.5427 | 1.3832  | 0.6390 | 4.0352
3       | 0.6081 | -0.9196 | 0.5925 | 0.1016
4       | -0.1618| 0.4694  | 0.2030 | 3.0117
5       | 0.1870 | -0.2578 | 0.6124 | 1.7749
6       | 0.4891 | -0.5276 | 0.4378 | 0.6439
7       | 0.3777 | 2.0149  | 0.7423 | 3.3932
8       | 1.1498 | -0.4067 | 0.2469 | 1.5866
9       | 0.9325 | 1.0950  | 1.0359 | 3.3591
10      | 0.5060 | 1.3317  | 0.9222 | 3.7174
11      | 0.0497 | -2.0656 | 0.6124 | -0.6585
12      | 0.4004 | 3.5369  | 0.9766 | 5.3532
13      | -0.1874| 1.3343  | 0.5374 | 3.2189
14      | 0.5060 | 1.3317  | 0.9222 | 3.7174
15      | 1.6375 | -0.7911 | 0.7537 | 0.5515
```

---

## Algoritmo da Regra Delta

**Saída combinada (sem ativação):**
```
u = w0·x0 + w1·x1 + w2·x2 + w3·x3 + w4·x4
  = w0·(+1) + w1·x1 + w2·x2 + w3·x3 + w4·x4
```

**Erro para o padrão p:**
```
e(p) = d(p) - u(p)
```

**Atualização de pesos (Regra Delta on-line):**
```
w(novo) = w(atual) + η · e(p) · x(p)
```

**EQM por época:**
```
EQM = (1/2P) · Σ e(p)²    (soma sobre todos os P padrões da época)
```

**Critério de parada:**
```
|EQM(época_atual) - EQM(época_anterior)| < ε  OU  EQM(época_atual) < ε
```
> O critério exato: a convergência é alcançada quando EQM < ε = 10⁻⁶ (ou variação do EQM < ε entre épocas consecutivas).

**Classificação final (inferência):**
```
y = sign(u)  →  +1 se u >= 0 (Válvula B), -1 se u < 0 (Válvula A)
```

---

## Estrutura de Arquivos a Criar

```
rna/
├── adaline.py          # Classe Adaline — núcleo do algoritmo
├── main.py             # Orquestração: 5 treinamentos, tabelas, gráficos
├── plotter.py          # Funções de plotagem (EQM) com Plotly
├── treinamento.csv     # Dados do conjunto de treinamento (35 padrões)
├── teste.csv           # Amostras para classificação final (15 padrões)
├── requirements.txt    # Dependências Python
└── PLAN.md             # Este documento
```

---

## Planos de Implementação

### Plano 1 — Dados (treinamento.csv e teste.csv)

**Arquivo:** `rna/treinamento.csv`
- Colunas: `x1, x2, x3, x4, d`
- 35 linhas com os padrões do Anexo do documento
- Valores de `d`: 1.0 ou -1.0

**Arquivo:** `rna/teste.csv`
- Colunas: `x1, x2, x3, x4`
- 15 linhas com as amostras da Tabela 2

---

### Plano 2 — Classe Adaline (`rna/adaline.py`)

```python
class Adaline:
    def __init__(self, learning_rate=0.0025, precision=1e-6, seed=None):
        ...
    
    def _init_weights(self):
        """Inicializa pesos aleatórios em [0, 1] com 5 pesos: [w0, w1, w2, w3, w4]"""
        ...
    
    def net_input(self, x):
        """Combinador linear: u = w·[1, x1, x2, x3, x4]"""
        ...
    
    def predict(self, x):
        """Classificação: sign(net_input)"""
        ...
    
    def fit(self, X, d):
        """
        Treina o ADALINE usando Regra Delta (on-line, padrão por padrão).
        Armazena: weights_history, eqm_history, initial_weights
        Retorna: número de épocas até convergência
        Critério de parada: EQM < precision
        """
        ...
```

**Detalhes críticos de implementação:**
- Bias como entrada fixa `x0 = +1` (w0 é o peso do bias)
- Entrada aumentada: `x_aug = [1, x1, x2, x3, x4]`
- Pesos: vetor de 5 elementos `[w0, w1, w2, w3, w4]`
- EQM calculado após processar TODOS os padrões de uma época
- Armazenar `eqm_history` para plot posterior
- Armazenar `initial_weights` (cópia antes do treinamento) para a tabela

---

### Plano 3 — Módulo de Plotagem (`rna/plotter.py`)

**Função 1: `plot_eqm_duas_curvas(eqm_t1, eqm_t2, filename)`**
- Plota EQM por época para T1 e T2 num mesmo gráfico
- Eixo X: época, Eixo Y: EQM (escala logarítmica recomendada)
- Linha colorida distinta para cada treinamento
- Exporta como HTML interativo com Plotly
- Arquivo: `rna/grafico_eqm_t1_t2.html`

**Função 2: `plot_eqm_individual(eqm, training_index, filename)`**
- Gráfico EQM individual por época para cada treinamento
- Arquivo: `rna/grafico_eqm_treinamento_{i}.html`

---

### Plano 4 — Orquestração (`rna/main.py`)

```
FLUXO PRINCIPAL:
1. Carregar treinamento.csv e teste.csv
2. Para i = 1 até 5:
   a. Definir seed = i * 42 (seeds distintas e reprodutíveis)
   b. Instanciar Adaline(learning_rate=0.0025, precision=1e-6, seed=seed)
   c. Registrar initial_weights
   d. Chamar adaline.fit(X_train, d_train)
   e. Registrar final_weights e numero_de_epocas
   f. Classificar as 15 amostras de teste
   g. Armazenar eqm_history para gráficos
3. Exibir Tabela 1: Pesos iniciais, pesos finais, nº de épocas (5 treinamentos)
4. Exibir Tabela 2: Classificação das 15 amostras para cada treinamento (y = A ou B)
5. Gerar gráfico EQM T1 e T2 juntos → grafico_eqm_t1_t2.html
6. (Opcional) Gerar gráficos EQM individuais para todos os 5 treinamentos
```

**Formato da saída Tabela 1:**
```
Treinamento | w0_ini | w1_ini | w2_ini | w3_ini | w4_ini | w0_fin | w1_fin | w2_fin | w3_fin | w4_fin | Épocas
```

**Formato da saída Tabela 2:**
```
Amostra | x1 | x2 | x3 | x4 | y(T1) | y(T2) | y(T3) | y(T4) | y(T5)
```
> Onde y = "A" se sign=-1, "B" se sign=+1

---

### Plano 5 — requirements.txt

```
numpy
pandas
plotly
```

---

## Questão Teórica (Item 4 do documento)

**Por que os pesos finais são praticamente idênticos mesmo com número de épocas diferentes?**

A Regra Delta minimiza o EQM, que é uma função convexa quadrática dos pesos. Por isso, existe um único mínimo global. Independentemente do ponto de partida (inicialização aleatória), o algoritmo converge para o mesmo ponto ótimo W* — o vetor de pesos que minimiza o EQM sobre todo o conjunto de treinamento. O número de épocas varia porque cada inicialização começa em uma distância diferente do ótimo, mas o destino final é sempre o mesmo mínimo da superfície de erro quadrática.

---

## Critérios de Verificação (UAT)

- [ ] `treinamento.csv` e `teste.csv` criados com os dados corretos do documento
- [ ] Classe `Adaline` com 5 pesos (w0 bias + w1..w4)
- [ ] 5 treinamentos com seeds distintas (nenhum vetor de pesos inicial igual)
- [ ] η = 0.0025 e ε = 10⁻⁶ utilizados
- [ ] Tabela 1 exibida com pesos iniciais, finais e nº de épocas para os 5 treinamentos
- [ ] Tabela 2 exibida com classificação das 15 amostras para todos os 5 treinamentos
- [ ] Gráfico EQM de T1 e T2 no mesmo arquivo HTML
- [ ] Classificação da amostra indica "A" (válvula A) ou "B" (válvula B)
- [ ] Pesos finais dos 5 treinamentos praticamente iguais (convergência ao mesmo W*)

---

## Referência Cruzada com Perceptron

O módulo `perceptron/` usa uma estrutura similar. O ADALINE difere nos seguintes pontos:

| Aspecto | Perceptron | ADALINE |
|---|---|---|
| Regra de atualização | Perceptron (erro na saída ativada) | Delta (erro na saída linear) |
| Função de ativação no treino | Sign (degrau) | Linear (sem ativação) |
| Critério de parada | Zero erros de classificação | EQM < ε |
| Métrica de erro | Número de erros | EQM (erro quadrático médio) |
| Entradas | 3 (x1, x2, x3) | 4 (x1, x2, x3, x4) |
| Pesos | 4 (w1, w2, w3, w0_bias) | 5 (w0, w1, w2, w3, w4) |
