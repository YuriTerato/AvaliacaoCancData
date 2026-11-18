import numpy as np
import pandas as pd
from pathlib import Path
from pickle import dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

pasta = Path(__file__).parent

#1 Carregar os dados
dados = pd.read_csv(pasta / 'dados' / 'breast-cancer.csv', na_values='?')
print(dados['Class'].value_counts())

#2 Tratar valores ausentes (substituir pela moda)
dados_atributos = dados.drop(columns=['Class'])
dados_classes = dados['Class']
modas = dados_atributos.mode().iloc[0].to_dict()
dados_atributos = dados_atributos.fillna(modas)

#3 Transformar os atributos categoricos em numericos (get_dummies)
dados_atributos = pd.get_dummies(dados_atributos, dtype=int)
colunas = list(dados_atributos.columns)

#4 Dividir em treino e teste
atributos_treino, atributos_teste, classes_treino, classes_teste = train_test_split(
    dados_atributos, dados_classes, test_size=0.3, random_state=42, stratify=dados_classes)

#5 Balancear os dados de treino
resampler = SMOTE(random_state=42)
atributos_treino, classes_treino = resampler.fit_resample(atributos_treino, classes_treino)

#6 Normalizar os atributos e salvar o normalizador
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(atributos_treino)
dump(modelo_normalizador, open(pasta / 'modelos' / 'normalizador_cancer.model', 'wb'))
atributos_treino = modelo_normalizador.transform(atributos_treino)
atributos_teste = modelo_normalizador.transform(atributos_teste)

#7 Hiperparametrizar varios algoritmos
algoritmos = {
    'Arvore de Decisao': (DecisionTreeClassifier(random_state=42), {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [2, 3, 4, 5, None],
        'min_samples_split': [2, 4, 6, 8, 10]}),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance']}),
    'SVM': (SVC(random_state=42), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']}),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 4, 6]}),
    'Regressao Logistica': (LogisticRegression(max_iter=1000), {
        'C': [0.01, 0.1, 1, 10]}),
}

melhor_nome, melhor_modelo, melhor_acuracia = None, None, 0
print('\nAcuracia no teste por algoritmo:')
for nome, (algoritmo, grade) in algoritmos.items():
    combinacoes = int(np.prod([len(v) for v in grade.values()]))
    busca = RandomizedSearchCV(estimator=algoritmo, param_distributions=grade,
                               n_iter=min(10, combinacoes), cv=3, random_state=42, n_jobs=-1)
    busca.fit(atributos_treino, classes_treino)
    acuracia = accuracy_score(classes_teste, busca.predict(atributos_teste))
    print(f'{nome:22s} {acuracia:.3f}  {busca.best_params_}')
    if acuracia > melhor_acuracia:
        melhor_nome, melhor_modelo, melhor_acuracia = nome, busca.best_estimator_, acuracia

#8 Avaliar o melhor modelo
classes_preditas = melhor_modelo.predict(atributos_teste)
print(f'\nMelhor modelo: {melhor_nome} | Acuracia: {melhor_acuracia:.3f}\n')
print(classification_report(classes_teste, classes_preditas, zero_division=0))
print(confusion_matrix(classes_teste, classes_preditas))

#9 Salvar o modelo e as informacoes de preparacao
dump(melhor_modelo, open(pasta / 'modelos' / 'cancer_modelo.model', 'wb'))
dump({'colunas': colunas, 'modas': modas, 'nome': melhor_nome},
     open(pasta / 'modelos' / 'preparacao_cancer.model', 'wb'))
