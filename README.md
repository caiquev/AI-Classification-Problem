# 🤖 AI Classification Problem: CIFAR-10 🖼️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Tools-Jupyter%20Notebook-orange.svg)](https://jupyter.org/)
[![Machine Learning](https://img.shields.io/badge/Focus-Machine%20Learning-green.svg)]()

Este projeto explora os fundamentos da **Visão Computacional** e **Aprendizado de Máquina**, focando na classificação do dataset **CIFAR-10**. O diferencial aqui é a implementação manual (from scratch) de algoritmos, permitindo uma compreensão profunda da matemática por trás dos modelos.

## 📋 Visão Geral do Projeto

O objetivo principal é classificar imagens em 10 categorias diferentes (aviões, carros, pássaros, etc.). Para isso, o projeto aborda duas frentes principais:
1.  **KNN (K-Nearest Neighbors):** Implementado manualmente, utilizando otimização matricial com NumPy.
2.  **Redes Neurais (NN):** Notebook dedicado à experimentação de arquiteturas neurais para classificação.

## 🚀 Funcionalidades Técnicas

-   **Extração de Características:** Implementação de descritores HOG para capturar formas e texturas.
-   **KNN "From Scratch":** -   Cálculo de Distância Euclidiana otimizado via álgebra linear ($Dist(X, Y) = \sqrt{\sum X^2 - 2XY^T + \sum Y^2}$).
    -   Votação majoritária eficiente.
-   Rede Neural "From Scratch": ** - Testes com diferentes funções de ativação (RELU6, sigmoid e softmax), implementação de métodos para combater o overfitting (Regularização L2)
-   **Validação Cruzada (N-Fold):** Divisão de dados manual para busca de hiperparâmetros (K ideal).
-   **Visualização:** Scripts para gerar gráficos de acurácia vs. hiperparâmetros e análise de desvio padrão.

## 📁 Estrutura do Repositório

| Arquivo/Pasta | Descrição |
| :--- | :--- |
| `knn_classification.ipynb` | Implementação e testes do modelo KNN. |
| `nn_classification.ipynb` | Experimentos com Redes Neurais. |
| `src/` | Código fonte e módulos auxiliares. |
| `lecture_cifar.py` | Script para carregamento e pré-processamento do dataset. |
| `requirements.txt` | Lista de dependências para rodar o projeto. |

## 🛠️ Instalação e Execução

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/caiquev/AI-Classification-Problem.git](https://github.com/caiquev/AI-Classification-Problem.git)
    cd AI-Classification-Problem
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare o Dataset:**
    - Baixe o [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
    - Certifique-se de que o caminho dos dados esteja correto em `lecture_cifar.py`.

4.  **Execute os Notebooks:**
    - Abra o Jupyter Lab/Notebook e execute o `knn_classification.ipynb`.

## 📊 Resultados

O projeto foca na análise do desempenho de diferentes algoritmos de classificação. O modelo de Rede Neural (MLP) atingiu uma acurácia ligeiramente superior ao KNN nos dados de teste. Entretanto o modelo ainda sofre bastante com overfitting, sempre atingindo 100% de acuracia com os dados de apredizado.

| Modelo | Acurácia (Teste) | Complexidade de Inferência |
| :--- | :--- | :--- |
| KNN (K=7) | ~29% | $O(N \cdot D)$ (Lento/Custoso) |
| **ANN (MLP)** | **~38%+** | **$O(1)$ (Rápido/Paramétrico)** |

**Próximo Projeto:**
Para superar a barreira dos 38% e atingir níveis de performance de produção (>90%), o próximo passo lógico não é refinar este modelo, mas sim adotar **Transfer Learning**. Utilizaremos uma arquitetura **ResNet-18** (pré-treinada na ImageNet), aproveitando a extração de features robustas já aprendidas para focar apenas no *fine-tuning* para as classes do CIFAR-10.

## 📚 Tecnologias Utilizadas

-   [NumPy](https://numpy.org/) - Processamento matricial.
-   [Scikit-image](https://scikit-image.org/) - Processamento de imagens (HOG).
-   [Matplotlib](https://matplotlib.org/) - Visualização de dados.
-   [Scikit-learn](https://scikit-learn.org/) - Apenas para métricas de avaliação.

---
Desenvolvido por [Caique V.](https://github.com/caiquev)
