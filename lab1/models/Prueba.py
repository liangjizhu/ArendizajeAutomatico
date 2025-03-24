import pandas as pd
import joblib

# Cargar el modelo entrenado
model_path = 'modelo_final.pkl'  # Ajusta la ruta al modelo entrenado
model = joblib.load(model_path)

# Cargar el nuevo archivo de datos
data_path = '/Users/alfredofelices/PycharmProjects/ArendizajeAutomatico/lab1/data/attrition_competition_05.csv'
df_new = pd.read_csv(data_path)

# Preprocesar los datos (asegúrate de que el preprocesamiento sea el mismo que el usado en el entrenamiento)
X_new = df_new.drop(columns=['EmployeeID'])
# Realizar las predicciones
predictions = model.predict(X_new)

# Crear un DataFrame con los resultados
results = pd.DataFrame({
    'EmployeeID': df_new['EmployeeID'],
    'Attrition': predictions
})

# Imprimir los resultados
print(results)