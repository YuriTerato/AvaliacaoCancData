import json
import sys
import pandas as pd
from pathlib import Path
from pickle import load

pasta = Path(__file__).parent / 'modelos'

#Nova paciente (edite os valores ou passe um arquivo JSON: python inferencia.py paciente.json)
nova_paciente = {
    'age': '50-59',
    'menopause': 'ge40',
    'tumor-size': '30-34',
    'inv-nodes': '3-5',
    'node-caps': 'yes',
    'deg-malig': 3,
    'breast': 'left',
    'breast-quad': 'left_low',
    'irradiat': 'yes',
}
if len(sys.argv) > 1:
    nova_paciente = json.load(open(sys.argv[1], encoding='utf-8'))

#Carregar normalizador, informacoes de preparacao e modelo
normalizador = load(open(pasta / 'normalizador_cancer.model', 'rb'))
preparacao = load(open(pasta / 'preparacao_cancer.model', 'rb'))
modelo = load(open(pasta / 'cancer_modelo.model', 'rb'))

#Preparar a paciente da mesma forma que no treinamento
paciente = pd.DataFrame([nova_paciente]).replace('?', None)
paciente = paciente.fillna(preparacao['modas'])
paciente = pd.get_dummies(paciente, dtype=int)
invalidos = [c for c in paciente.columns if c not in preparacao['colunas']]
if invalidos:
    sys.exit(f'Valor invalido: {invalidos}')
paciente = paciente.reindex(columns=preparacao['colunas'], fill_value=0)
paciente = normalizador.transform(paciente)

#Classificar
classe = modelo.predict(paciente)[0]
print('Class prevista:', classe)
print('Modelo:', preparacao['nome'])
