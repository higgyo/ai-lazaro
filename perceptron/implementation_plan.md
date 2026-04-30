# Implementação de Perceptron com Regra de Hebb (Supervisionado)

O objetivo é implementar um Perceptron para classificar amostras de óleo em duas classes (-1 e +1) baseadas em três características ($x_1, x_2, x_3$). A implementação será feita usando NumPy (sem bibliotecas de Machine Learning como `scikit-learn`), com a extração dos dados para arquivos `.csv` e a geração de gráficos interativos das épocas.

## User Review Required

> [!IMPORTANT]
> A plotagem interativa para cada época, considerando que os dados possuem 3 dimensões ($x_1, x_2, x_3$), requer a visualização de um plano separador 3D. 
> Utilizarei a biblioteca **Plotly** para gerar um arquivo HTML com o gráfico 3D interativo para cada um dos 5 treinamentos. Nesses gráficos, poderemos ver os pontos (amostras de óleo) e o plano de decisão se ajustando a cada época. Por favor, confirme se o uso do **Plotly** atende à sua expectativa para os gráficos interativos.

## Open Questions

- Como deseja que as tabelas de resultados e as respostas teóricas sejam entregues? Posso gerar um arquivo `relatorio.md` (Markdown) ou exibir tudo no próprio terminal ao final da execução.

## Proposed Changes

### Dados e Estrutura

#### [NEW] [treinamento.csv](file:///home/alunos/Desktop/perceptron/treinamento.csv)
Arquivo contendo os 30 padrões de treinamento (x1, x2, x3, d) extraídos do documento.

#### [NEW] [teste.csv](file:///home/alunos/Desktop/perceptron/teste.csv)
Arquivo contendo as 10 amostras de teste (x1, x2, x3) extraídas do documento.

### Código Fonte

#### [NEW] [perceptron.py](file:///home/alunos/Desktop/perceptron/perceptron.py)
A classe base do Perceptron implementando:
- Inicialização de pesos com `numpy.random.uniform(0, 1, 4)`.
- Método de treinamento `fit` usando a Regra de Hebb: $w_i(t+1) = w_i(t) + \eta \cdot d \cdot x_i$.
- Uma taxa de aprendizado $\eta = 0.01$.
- Registro do histórico de pesos ao final de cada época para posterior plotagem.

#### [NEW] [main.py](file:///home/alunos/Desktop/perceptron/main.py)
Script principal que irá:
1. Carregar os dados de `treinamento.csv` e `teste.csv`.
2. Executar 5 instâncias de treinamento, registrando pesos iniciais/finais e o número de épocas para cada um.
3. Prever as classes das amostras de teste e gerar as tabelas solicitadas no documento.
4. Chamar a função de plotagem.

#### [NEW] [plotter.py](file:///home/alunos/Desktop/perceptron/plotter.py)
Módulo dedicado a usar o **Plotly** para gerar arquivos HTML iterativos (ex: `grafico_treinamento_1.html`). Vai ler o histórico de pesos da classe `Perceptron` e criar animações ou sliders do plano de separação 3D se ajustando ao longo das épocas.

### Respostas Teóricas

#### [NEW] [respostas.md](file:///home/alunos/Desktop/perceptron/respostas.md)
Documento Markdown contendo a explicação teórica solicitada:
1. Por que o número de épocas varia a cada execução (devido à inicialização aleatória dos pesos que muda o ponto de partida na superfície de erro).
2. A principal limitação do Perceptron (só consegue resolver problemas linearmente separáveis, como demonstrado pelo XOR).

## Verification Plan

### Automated Tests
- Executaremos o `main.py` para garantir que o Perceptron convirja nas 5 execuções com a regra de Hebb.
- Validaremos se os arquivos HTML com os gráficos interativos 3D são gerados corretamente e podem ser abertos no navegador.
- Validaremos se a matriz de respostas das amostras de teste se alinha com o esperado para classes linearmente separáveis.
