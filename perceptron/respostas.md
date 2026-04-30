# Respostas Teóricas - Laboratório de Perceptron

## 1. Explique por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron.

O número de épocas varia porque em cada treinamento o vetor de pesos ($w_0, w_1, w_2, w_3$) é inicializado com **valores aleatórios diferentes** (neste caso, entre 0 e 1). 
Geometricamente, os pesos representam a posição e a inclinação inicial do hiperplano de separação no espaço tridimensional. Dependendo de quão próximo ou longe o hiperplano gerado aleatoriamente já está de uma solução que separe corretamente os dados (as duas classes de óleo), o algoritmo exigirá um número maior ou menor de ajustes (iterações) até encontrar a superfície de decisão ótima, refletindo assim num número de épocas variável.

---

## 2. Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões.

A principal limitação de um Perceptron de camada única é que ele **só consegue aprender e classificar padrões que sejam linearmente separáveis**. 
Ou seja, ele só encontra uma solução se for possível traçar um único hiperplano reto (uma linha em 2D, um plano em 3D, etc.) que divida completamente os conjuntos de dados de entrada em suas respectivas classes. Caso as classes estejam sobrepostas de forma não-linear (como o clássico problema da porta lógica XOR), o Perceptron não conseguirá convergir, pois seu algoritmo ficará em um loop infinito de ajustes tentando (sem sucesso) encontrar um separador linear onde ele não existe.
