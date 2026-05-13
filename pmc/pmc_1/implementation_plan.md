# Implementação do Módulo `pmc/pmc_1`

## Contexto

O arquivo `PMC1.docx` descreve um problema de regressão com uma rede **Perceptron Multicamadas (PMC)** para estimar a energia absorvida `{y}` de um sistema de ressonância magnética a partir de três variáveis de entrada `{x1, x2, x3}`.

A rede possui **topologia de duas camadas** (conforme figura no docx) e deve ser treinada com o algoritmo **Backpropagation (Regra Delta Generalizada)**, usando **função de ativação logística**, taxa de aprendizado `η = 0.1` e precisão `ε = 10⁻⁶`.

A estrutura de arquivos deve espelhar o módulo `rna`, que contém:
- Um script Python principal
- Arquivos CSV de treinamento e teste
- Um gráfico `.png` gerado pelo script
- Um arquivo `respostas.md` com as respostas formatadas em Markdown

---

## Tarefas Pedidas no Docx

### Item 1 — Treinamento (5 runs)
Executar **5 treinamentos independentes** com pesos iniciais aleatórios entre 0 e 1 (seeds distintas). Para cada treinamento registrar:
- Vetor/Matriz de pesos iniciais
- Vetor/Matriz de pesos finais
- Número de épocas até convergência

### Item 2 — Gráfico EQM
Para os **2 treinamentos com maior número de épocas**, traçar os gráficos de **EQM × Época** numa mesma figura (não sobrepostos, i.e., subplots), salvo como `.png`.

### Item 3 — Explicação textual
Explicar, com base nos resultados da tabela do item 1, **por que o EQM e o número de épocas variam** de treinamento para treinamento (ao contrário do ADALINE, que sempre converge para o mesmo ponto).

### Item 4 — Validação e Erro Relativo Médio
Aplicar as **5 redes treinadas** ao conjunto de teste fornecido no docx. Para cada treinamento, calcular:
- **Erro Relativo Médio (%)** entre valores desejados e estimados
- **Variância** do erro relativo

### Item 5 — Escolha da melhor rede
Com base na tabela de validação, **indicar qual treinamento (T1–T5) oferece a melhor generalização** e justificar a escolha.

---

## Proposta de Arquivos

### Estrutura final de `pmc/pmc_1/` (espelhando `rna/`)

```
pmc/pmc_1/
├── PMC1.docx           # já existe
├── treinamento.csv     # [NEW] extraído do Anexo do docx
├── teste.csv           # [NEW] extraído da tabela de teste do docx
├── pmc1.py             # [NEW] script principal de treinamento e validação
├── grafico_eqm.png     # [GERADO] pelo script
└── respostas.md        # [NEW] respostas formatadas em Markdown
```

---

## Mudanças Propostas

### Dados

#### [NEW] `treinamento.csv`
- Extrair os 200 padrões do Anexo do docx (colunas: `x1, x2, x3, y`)
- As entradas `x1, x2, x3` já estão normalizadas (conforme enunciado)

#### [NEW] `teste.csv`
- Extrair os padrões de teste do docx com os respectivos valores desejados `d`
- Colunas: `x1, x2, x3, d`

---

### Script Principal

#### [NEW] `pmc1.py`

Estrutura análoga ao `adaline.py` do módulo `rna`:

1. **Carregar dados** de `treinamento.csv` e `teste.csv` com `numpy.genfromtxt`
2. **Definir a topologia da rede** — 2 camadas conforme figura do docx (a determinar exatamente o número de neurônios ocultos a partir da figura)
3. **Função `train_pmc(X, d, eta, precisao, seed)`**:
   - Inicializar pesos aleatórios entre 0 e 1
   - Implementar forward pass com ativação logística
   - Implementar backpropagation (Regra Delta Generalizada)
   - Calcular EQM por época; parar quando `|EQM_atual − EQM_anterior| ≤ precisao`
   - Retornar: pesos iniciais, pesos finais, épocas, lista de EQMs
4. **Executar 5 treinamentos** com `seed = 42, 43, 44, 45, 46`
5. **Identificar os 2 treinamentos com maior número de épocas**
6. **Gerar gráfico** EQM × Época para esses 2 treinamentos em subplots (não sobrepostos), salvar como `grafico_eqm.png`
7. **Validação**: aplicar as 5 redes ao conjunto de teste, calcular erro relativo médio (%) e variância para cada rede
8. **Imprimir** tabelas de resultados para montagem do `respostas.md`

---

### Respostas

#### [NEW] `respostas.md`

Com as seguintes seções (espelhando `respostas.md` do módulo `rna`):

1. **Descrição da rede PMC** — topologia, parâmetros, algoritmo
2. **Tabela de Treinamentos** — pesos iniciais, pesos finais, épocas (5 treinamentos)
3. **Gráfico EQM** — embed do `grafico_eqm.png`
4. **Análise das variações** — resposta textual ao item 3 do docx
5. **Tabela de Validação** — erro relativo médio (%) e variância por treinamento
6. **Melhor generalização** — indicação e justificativa do melhor treinamento

---

## Questões em Aberto

> [!IMPORTANT]
> **Topologia da rede**: O docx menciona "duas camadas neurais" e exibe uma figura, mas o número exato de neurônios na camada oculta precisa ser confirmado visualmente no arquivo `.docx`. Qual é o número de neurônios ocultos?

> [!IMPORTANT]
> **Conjunto de teste**: O docx possui uma tabela de teste separada (diferente do Anexo de treinamento). É necessário confirmar se essa tabela inclui os valores desejados `d` para calcular o erro relativo, ou se `d` vem de outra fonte.

> [!NOTE]
> **Formato dos pesos**: Como a PMC tem matrizes de pesos (não vetores simples como no ADALINE), a tabela de treinamentos no `respostas.md` precisará representar as matrizes de forma compacta (ex.: norma, ou apenas dimensões + valores finais resumidos).

---

## Plano de Verificação

### Automatizado
- Executar `python3 pmc1.py` e verificar que:
  - Todos os 5 treinamentos convergem
  - O arquivo `grafico_eqm.png` é gerado corretamente
  - As métricas (EQM, erro relativo, variância) são impressas sem erros

### Manual
- Confirmar que `respostas.md` renderiza corretamente no GitHub/VSCode
- Confirmar que o gráfico é legível e os subplots estão não sobrepostos
- Confirmar que a estrutura de arquivos espelha fielmente o módulo `rna`
