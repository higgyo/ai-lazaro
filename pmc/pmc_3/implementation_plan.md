# Implementação do Módulo `pmc/pmc_3`

## Contexto

O arquivo `PMC3.docx` descreve um problema de **previsão de série temporal** de preços no mercado financeiro. O histórico completo da série é `f(t)` para `t = 1..120`, onde:

- **Treinamento:** `t = 1..100` (100 pontos, Anexo do docx)
- **Teste/Validação:** `t = 101..120` (20 pontos, valores desejados conhecidos)

A arquitetura proposta é uma **TDNN (Time Delay Neural Network)**, uma PMC que usa janelas de tempo anteriores como entradas para prever o próximo valor:

```
entrada: [f(t-p), f(t-p+1), ..., f(t-1)]  →  saída: f(t)
```

São avaliadas **3 topologias candidatas** com **backpropagation + momentum**:

| Rede | Janela de entrada (p) | Neurônios ocultos (N1) |
|---|---|---|
| Rede 1 | 5 | 10 |
| Rede 2 | 10 | 15 |
| Rede 3 | 15 | 25 |

Parâmetros fixos para todos: função logística, `η = 0.1`, `α = 0.8`, `ε = 0.5×10⁻⁶`.

A estrutura de arquivos deve espelhar o módulo `rna/`.

---

## Tarefas Pedidas no Docx

### Item 1 — 3 treinamentos × 3 topologias (9 treinamentos no total)
Para cada uma das 3 redes, executar **3 treinamentos independentes** com pesos iniciais aleatórios distintos. Registrar:
- EQM final de cada treinamento
- Número de épocas até convergência

### Item 2 — Tabela de resultados
Preencher a tabela com EQM e Épocas para os 9 treinamentos (3 redes × 3 runs).

### Item 3 — Validação com erro relativo médio e variância
Para **todos os 9 treinamentos**, aplicar as redes no conjunto de teste (`t = 101..120`).  
Calcular para cada um:
- **Erro relativo médio (%)** entre `f(t)` desejado e estimado
- **Variância** do erro relativo

### Item 4 — Gráficos EQM (subplots, melhor treinamento por rede)
Para cada topologia, selecionar o **melhor treinamento** (menor EQM/erro relativo) e traçar o gráfico EQM × Época.  
Os **3 gráficos** numa mesma figura em **subplots não sobrepostos** → salvo como `grafico_eqm.png`.

### Item 5 — Gráficos desejado vs. estimado (subplots, melhor treinamento por rede)
Para os mesmos 3 melhores treinamentos, traçar `f(t)` desejado vs. estimado em `t = 101..120`.  
Os **3 gráficos** numa mesma figura em **subplots não sobrepostos** → salvo como `grafico_estimado.png`.

### Item 6 — Seleção da melhor topologia
Indicar qual das 3 redes e qual treinamento (T1/T2/T3) oferece a **melhor previsão** e justificar.

### Item 7 — Pesquisa teórica (texto dissertativo)
Investigar e comentar as principais características e vantagens de:
- **RProp (Resilient Propagation)**
- **Levenberg-Marquardt (LM)**

---

## Proposta de Arquivos

### Estrutura final de `pmc/pmc_3/` (espelhando `rna/`)

```
pmc/pmc_3/
├── PMC3.docx              # já existe
├── serie_temporal.csv     # [NEW] f(t) para t=1..120 (treinamento + teste juntos)
├── pmc3.py                # [NEW] script principal
├── grafico_eqm.png        # [GERADO] subplots EQM dos 3 melhores treinamentos
├── grafico_estimado.png   # [GERADO] subplots estimado vs. desejado (t=101..120)
└── respostas.md           # [NEW] respostas formatadas em Markdown
```

> [!NOTE]
> Diferentemente dos módulos anteriores, aqui não há `treinamento.csv` e `teste.csv` separados. Os dados são uma única série temporal contínua `f(t)` — a janela deslizante gera os padrões de treinamento dinamicamente no script. Usar um único arquivo `serie_temporal.csv` é mais fiel à natureza do problema.

---

## Mudanças Propostas

### Dados

#### [NEW] `serie_temporal.csv`
- Série temporal completa: `t = 1..120`, coluna única `f_t`
- `t = 1..100` → usados para gerar os padrões de treinamento via janela deslizante
- `t = 101..120` → usados como alvos de validação

**Como gerar os padrões de treinamento para cada rede:**

Para a Rede `r` com janela `p`:
- Entrada do padrão `i`: `[f(i), f(i+1), ..., f(i+p-1)]`
- Saída desejada: `f(i+p)`
- Número de padrões: `100 - p` (ex.: Rede 1 → 95 padrões, Rede 2 → 90, Rede 3 → 85)

**Para a fase de teste (previsão iterativa `t = 101..120`):**
A rede usa os últimos `p` valores conhecidos como entrada, produz `f(t)`, e esse valor é realimentado nas próximas previsões.

---

### Script Principal

#### [NEW] `pmc3.py`

Estrutura análoga ao `adaline.py` do módulo `rna`, porém com três blocos de rede:

1. **Carregar série temporal** de `serie_temporal.csv`

2. **Função `build_patterns(serie, p)`** — constrói os padrões de treinamento por janela deslizante

3. **Função `train_bp_momentum(X, d, n_hidden, eta, alpha, precisao, seed)`**:
   - Topologia: `p` entradas → `N1` neurônios ocultos → `1` saída
   - Forward pass com sigmoid em ambas as camadas
   - Backward pass com Regra Delta Generalizada + momentum
   - Calcular EQM por época; parar quando `|EQM_atual − EQM_anterior| ≤ 0.5×10⁻⁶`
   - Retornar: pesos iniciais, pesos finais, épocas, lista de EQMs

4. **Loop principal**: para cada uma das 3 redes, executar 3 treinamentos (seeds distintas)
   - Armazenar todos os resultados (9 treinamentos × EQM + épocas + pesos)

5. **Validação**: previsão iterativa de `f(t)` para `t = 101..120` com cada uma das 9 redes
   - Calcular erro relativo médio (%) e variância para cada treinamento

6. **Identificar melhor treinamento por rede** (menor erro relativo médio no teste)

7. **Gerar `grafico_eqm.png`**: 3 subplots verticais, um por rede, com o EQM do melhor treinamento

8. **Gerar `grafico_estimado.png`**: 3 subplots verticais, um por rede, com `f(t)` desejado e estimado para `t = 101..120`

9. **Imprimir** resultados para montagem do `respostas.md`

---

### Respostas

#### [NEW] `respostas.md`

Seções (espelhando a estrutura do `respostas.md` do módulo `rna`):

1. **Descrição do problema e das topologias TDNN** — série temporal, janela deslizante, 3 arquiteturas
2. **Tabela de treinamentos** — EQM final e épocas para os 9 treinamentos (3 redes × 3 runs)
3. **Gráfico EQM** — embed de `grafico_eqm.png`
4. **Tabela de validação** — `f(t)` desejado vs. estimado para `t=101..120`, erro relativo médio e variância (por treinamento e rede)
5. **Gráfico estimado vs. desejado** — embed de `grafico_estimado.png`
6. **Melhor topologia e treinamento** — indicação e justificativa
7. **RProp** — características e vantagens (pesquisa teórica)
8. **Levenberg-Marquardt** — características e vantagens (pesquisa teórica)

---

## Questões em Aberto

> [!IMPORTANT]
> **Camadas ocultas**: O docx especifica `N1` (número de neurônios na **primeira** camada oculta). Não fica claro na descrição textual se há apenas **uma camada oculta** ou mais. A figura no docx detalha isso, mas não é extraível automaticamente. Confirmar: cada rede tem **apenas 1 camada oculta** com os neurônios especificados?

> [!NOTE]
> **Previsão iterativa no teste**: Para prever `f(101)` com a Rede 1 (p=5), a janela é `[f(96), f(97), f(98), f(99), f(100)]`. Para prever `f(102)`, a janela inclui `f(101)` já previsto. Essa realimentação acumula erro — é o comportamento esperado e será registrado.

> [!NOTE]
> **EQM no critério de parada**: O critério `|EQM_atual − EQM_anterior| ≤ 0.5×10⁻⁶` é ligeiramente diferente dos módulos anteriores (`10⁻⁶`). Isso será parametrizado corretamente no script como `precisao = 0.5e-6`.

---

## Plano de Verificação

### Automatizado
- Executar `python3 pmc3.py` e verificar que:
  - Os 9 treinamentos convergem para todas as 3 redes
  - `grafico_eqm.png` e `grafico_estimado.png` são gerados com 3 subplots cada
  - Erros relativos médios e variâncias são impressos para todos os 9 treinamentos

### Manual
- Verificar que os gráficos `estimado vs. desejado` mostram visualmente qual rede generaliza melhor
- Confirmar que `respostas.md` renderiza corretamente com as duas imagens embutidas
- Conferir que a previsão iterativa realimenta corretamente os valores previstos
