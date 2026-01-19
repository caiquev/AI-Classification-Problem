# Classificação de Imagens CIFAR-10: KNN do Zero com Descritores HOG

Este projeto foi desenvolvido com o objetivo de explorar os fundamentos da Visão Computacional e do Aprendizado de Máquina. Ele implementa o algoritmo **K-Nearest Neighbors (KNN)** sem o uso de bibliotecas de alto nível para o modelo, focando na compreensão matemática da extração de características e métricas de distância.

## ⚙️ Funcionalidades
- **Extração de Características:** Uso de HOG (Histogram of Oriented Gradients) para representar formas e texturas.
- **KNN "From Scratch":** Implementação manual do cálculo de distância Euclidiana otimizada e votação majoritária.
- **Validação Cruzada (N-Fold):** Implementação manual da divisão de dados para garantir a robustez do modelo.
- **Visualização de Resultados:** Gráficos de acurácia vs. hiperparâmetro K com desvio padrão.

## Abordagem Técnica
Diferente de abordagens que usam pixels brutos, este projeto utiliza o HOG para reduzir a dimensionalidade e focar na estrutura dos objetos, o que é essencial para o algoritmo KNN, que sofre com a "maldição da dimensionalidade".

A fórmula de distância utilizada foi a Euclidiana, otimizada via álgebra linear para performance em NumPy:
$$Dist(X, Y) = \sqrt{\sum X^2 - 2XY^T + \sum Y^2}$$

## Tecnologias
- Python 3.x
- NumPy (Processamento matricial)
- Scikit-image (Extração de HOG)
- Matplotlib (Visualização de dados)

## Como Executar
1. Certifique-se de ter o dataset CIFAR-10 baixado.
2. Altere a variável `path` no script para o diretório dos seus dados.
3. Execute: `python TD1_kppv.py`