import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
data_cars = {
    'Combustivel': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel', 'Etanol', 'Gasolina'],
    'Idade': [5, 3, 7, 2, 6, 4, 8, 3, 5, 1],
    'Quilometragem': [50000, 30000, 70000, 20000, 60000, 40000, 80000, 30000, 50000, 10000],
    'Preco': [30000, 28000, 25000, 35000, 26000, 27000, 24000, 29000, 31000, 37000]
}
df_cars = pd.DataFrame(data_cars)

# Definindo variáveis preditoras e alvo
X_cars = df_cars[['Combustivel', 'Idade', 'Quilometragem']]
y_cars = df_cars['Preco']

# Divisão dos dados em treino e teste
from sklearn.model_selection import train_test_split
X_train_cars, X_test_cars, y_train_cars, y_test_cars = train_test_split(X_cars, y_cars, test_size=0.2, random_state=42)

# Importações para pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Definindo as transformações para dados numéricos e categóricos
numeric_features = ['Idade', 'Quilometragem']
numeric_transformer = StandardScaler()

categorical_features = ['Combustivel']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Criação do ColumnTransformer que aplica as transformações correspondentes
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline completo: pré-processamento + modelo de regressão linear
pipeline_cars = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Treinamento do modelo
pipeline_cars.fit(X_train_cars, y_train_cars)

# Previsões e avaliação do desempenho (Erro Quadrático Médio)
y_pred_cars = pipeline_cars.predict(X_test_cars)
mse_cars = mean_squared_error(y_test_cars, y_pred_cars)

print("----- Previsão do Preço de Automóveis -----")
print("MSE:", mse_cars)
print("Previsões:", y_pred_cars)

# Visualização: Preço real x Preço previsto (para os dados de teste)
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test_cars)), y_test_cars, color='blue', label='Preço Real')
plt.scatter(range(len(y_test_cars)), y_pred_cars, color='red', label='Preço Previsto')
plt.title("Preço Real x Previsto dos Automóveis")
plt.xlabel("Amostras")
plt.ylabel("Preço")
plt.legend()
plt.show()
