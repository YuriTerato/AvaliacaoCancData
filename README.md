# Breast Cancer (UCI)

Classificador que indica o atributo `Class` (`no-recurrence-events` ou `recurrence-events`) de uma paciente, usando a base [Breast Cancer](https://archive.ics.uci.edu/dataset/14/breast+cancer) do UCI.

## Como rodar

```bash
pip install -r requirements.txt
python treinamento.py
python inferencia.py
```

## Arquivos

- `treinamento.py`: treina e salva o modelo.
- `inferencia.py`: classifica uma nova paciente. Os dados dela ficam no dicionário `nova_paciente` no começo do arquivo, ou em um JSON: `python inferencia.py paciente.json`.
- `dados/breast-cancer.csv`: base de dados.
- `modelos/`: normalizador, preparação dos dados e modelo salvos pelo treinamento.

## O que o treinamento faz

1. Carrega os dados e troca os valores ausentes (`?`) pela moda.
2. Converte os atributos categóricos em numéricos (`get_dummies`).
3. Divide em treino (70%) e teste (30%).
4. Balanceia o treino com SMOTE.
5. Normaliza os atributos com MinMaxScaler e salva o normalizador.
6. Testa 5 algoritmos (Árvore de Decisão, KNN, SVM, Random Forest e Regressão Logística), cada um com RandomizedSearchCV para achar os melhores hiperparâmetros.
7. Escolhe o de maior acurácia no teste e salva em `modelos/`.

## Resultado

| Algoritmo | Acurácia no teste |
|---|---|
| Árvore de Decisão | 0,547 |
| KNN | 0,593 |
| Regressão Logística | 0,628 |
| Random Forest | 0,640 |
| **SVM** | **0,698** |

Valores aceitos na inferência: `age` (10-19 até 90-99), `menopause` (lt40, ge40, premeno), `tumor-size` (0-4 até 55-59), `inv-nodes` (0-2 até 36-39), `node-caps` (yes, no), `deg-malig` (1, 2, 3), `breast` (left, right), `breast-quad` (left_up, left_low, right_up, right_low, central), `irradiat` (yes, no). Para `node-caps` e `breast-quad` dá para usar `?`.

Fonte dos dados: Zwitter e Soklic (1988), UCI Machine Learning Repository.
