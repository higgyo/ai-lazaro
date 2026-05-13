# Implementação do Módulo `pmc/pmc_2`

## Contexto

O arquivo `PMC2.docx` descreve um problema de **classificação de padrões** com uma rede **Perceptron Multicamadas (PMC)** para identificar qual tipo de conservante (A, B ou C) deve ser aplicado em bebidas, a partir de 4 variáveis de entrada normalizadas: `x1` (teor de água), `x2` (grau de acidez), `x3` (temperatura) e `x4` (tensão superficial).

A rede possui **4 entradas e 3 saídas** (one-hot encoding: A = [1,0,0], B = [0,1,0], C = [0,0,1]).  
O diferencial em relação ao `pmc_1` é a **comparação entre backpropagation padrão e backpropagation com momentum** usando os **mesmos pesos iniciais**.

- **Dados de treinamento:** 140 padrões (tabela com 70 linhas × 2 amostras por linha = 140 amostras)
- **Dados de teste:** 18 amostras (com saídas desejadas `d1, d2, d3` para validação)
- **Codificação da saída:**

| Conservante | y1 | y2 | y3 |
|---|---|---|---|
| Tipo A | 1 | 0 | 0 |
| Tipo B | 0 | 1 | 0 |
| Tipo C | 0 | 0 | 1 |

A estrutura de arquivos deve espelhar o módulo `rna/`.

---

## Tarefas Pedidas no Docx

### Item 1 — Backpropagation Padrão
Treinar a rede com o **algoritmo backpropagation padrão**:
- Pesos iniciais aleatórios entre 0 e 1 (uma única seed)
- Função de ativação: **logística (sigmoid)**
- Taxa de aprendizado: `η = 0.1`
- Critério de parada: `|EQM_atual − EQM_anterior| ≤ 10⁻⁶`
- Registrar: pesos iniciais, pesos finais, número de épocas, **tempo de processamento**

### Item 2 — Backpropagation com Momentum
Treinar com **os mesmos pesos iniciais** do Item 1, agora com momentum:
- Parâmetros iguais ao Item 1
- Fator de momentum: `α = 0.9`
- Registrar: pesos finais, número de épocas, **tempo de processamento**

### Item 3 — Gráfico EQM (subplots)
Para os **2 treinamentos** (padrão e com momentum), traçar EQM × Época numa mesma figura com **subplots não sobrepostos**, salvo como `.png`.

### Item 4 — Pós-processamento (arredondamento simétrico)
Implementar a rotina de pós-processamento que converte as saídas reais da rede em inteiros `{0, 1}` usando **arredondamento simétrico** (`round(y) → 0 ou 1`), aplicado exclusivamente no conjunto de teste.

### Item 5 — Validação e Taxa de Acerto
Aplicar **ambas as redes treinadas** ao conjunto de teste (18 amostras). Após o pós-processamento, calcular a **taxa de acerto (%)** comparando as saídas classificadas com os valores desejados `d1, d2, d3`.

---

## Proposta de Arquivos

### Estrutura final de `pmc/pmc_2/` (espelhando `rna/`)

```
pmc/pmc_2/
├── PMC2.docx             # já existe
├── treinamento.csv       # [NEW] 140 padrões extraídos do Anexo do docx
├── teste.csv             # [NEW] 18 padrões extraídos da tabela de teste
├── pmc2.py               # [NEW] script principal
├── grafico_eqm.png       # [GERADO] subplots EQM padrão vs. momentum
└── respostas.md          # [NEW] respostas formatadas em Markdown
```

---

## Mudanças Propostas

### Dados

#### [NEW] `treinamento.csv`
- **140 padrões** extraídos da Table 3 do docx (70 linhas × 2 colunas de amostras por linha)
- Colunas: `x1, x2, x3, x4, d1, d2, d3`

#### [NEW] `teste.csv`
- **18 amostras** extraídas da Table 2 do docx (última linha "Taxa de Acerto" excluída)
- Colunas: `x1, x2, x3, x4, d1, d2, d3`

---

### Script Principal

#### [NEW] `pmc2.py`

Estrutura análoga ao `adaline.py` do módulo `rna`:

1. **Carregar dados** de `treinamento.csv` e `teste.csv`
2. **Definir topologia da rede** — 4 entradas + camada(s) oculta(s) + 3 saídas

   > [!IMPORTANT]
   > O número de neurônios ocultos precisa ser lido da figura no docx (não extraível automaticamente). A ser confirmado pelo usuário.

3. **Função `train_bp(X, d, eta, precisao, W_init)`** — backpropagation padrão:
   - Forward pass: `u = W·x`, `y = sigmoid(u)` camada a camada
   - Backward pass: cálculo dos deltas e atualização dos pesos pela Regra Delta Generalizada
   - Calcular EQM por época; parar quando `|EQM_atual − EQM_anterior| ≤ precisao`
   - Medir tempo total com `time.time()` ou `time.perf_counter()`
   - Retornar: pesos finais, épocas, lista de EQMs, tempo

4. **Função `train_bp_momentum(X, d, eta, alpha, precisao, W_init)`** — backpropagation com momentum:
   - Igual ao anterior, mas a atualização de pesos inclui o termo de momentum:
     `ΔW(t) = η·δ·x + α·ΔW(t-1)`
   - **Usa os mesmos pesos iniciais** (`W_init`) do treinamento padrão
   - Medir tempo total

5. **Pós-processamento**: função `postprocess(y_real) → y_int` usando arredondamento simétrico (`round`)

6. **Gerar gráfico** EQM × Época em subplots (2 linhas, 1 coluna ou lado a lado), salvar como `grafico_eqm.png`

7. **Validação**: aplicar ambas as redes ao conjunto de teste, aplicar pós-processamento, comparar com `d` e calcular taxa de acerto (%)

8. **Imprimir** resultados para montagem do `respostas.md`

---

### Respostas

#### [NEW] `respostas.md`

Seções (espelhando a estrutura do `respostas.md` do módulo `rna`):

1. **Descrição da rede PMC** — topologia, codificação de saída, parâmetros
2. **Tabela comparativa de treinamentos** — épocas, tempo e EQM final para backpropagation padrão e com momentum
3. **Gráfico EQM** — embed do `grafico_eqm.png`
4. **Tabela de validação** — saídas pós-processadas vs. desejadas + taxa de acerto para cada treinamento
5. **Análise comparativa** — discussão sobre as diferenças de velocidade de convergência entre backpropagation padrão e com momentum

---

## Questões em Aberto

> [!IMPORTANT]
> **Número de neurônios ocultos**: O docx menciona "rede perceptron com 04 entradas e 03 saídas" e exibe uma figura com a topologia completa. Não é possível extrair automaticamente o número de neurônios ocultos da imagem embutida no docx. **Quantos neurônios há na(s) camada(s) oculta(s)?**

> [!NOTE]
> **Número de amostras de treinamento**: A tabela do Anexo tem 70 linhas com 2 amostras por linha, mas as últimas linhas do lado direito parecem estar vazias (amostra 71–140 ou apenas 140?). A extração indica "maior amostra no lado direito = 130". Isso sugere **130 amostras de treinamento** (não 140). A ser confirmado após extração completa no script.

> [!NOTE]
> **Arredondamento simétrico**: O critério mencionado no docx é o arredondamento simétrico para `{0, 1}`. Em Python, `round()` já implementa arredondamento bancário (para par). Pode ser necessário usar `int(y + 0.5)` para garantir arredondamento simétrico correto. Isso será tratado no script.

---

## Plano de Verificação

### Automatizado
- Executar `python3 pmc2.py` e verificar que:
  - Ambos os treinamentos convergem
  - `grafico_eqm.png` é gerado com 2 subplots legíveis
  - Taxa de acerto é impressa para ambas as redes

### Manual
- Confirmar que `respostas.md` renderiza corretamente
- Comparar visualmente o número de épocas entre backpropagation padrão e com momentum (momentum deve convergir mais rápido)
- Confirmar que os pesos iniciais são idênticos nos dois treinamentos
