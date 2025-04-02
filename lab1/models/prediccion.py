import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Obtener la ruta del directorio del script actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Cargar el modelo final entrenado
model_path = os.path.join(current_dir, "modelo_final.pkl")
model = joblib.load(model_path)
print("Modelo final cargado correctamente.")

# 2. Cargar el dataset de competición
# Ajusta la ruta relativa según la ubicación del archivo
data_comp_path = os.path.join(current_dir, "..", "data", "attrition_competition_05.csv")
df_comp = pd.read_csv(data_comp_path)
print("Dimensiones del dataset de competición:", df_comp.shape)

# 3. Generar predicciones
# El modelo final es un pipeline que ya incluye el preprocesamiento,
# por lo que podemos aplicar directamente predict() sobre el dataframe de competición.
y_pred = model.predict(df_comp)

# 4. Convertir las predicciones a las etiquetas originales
# Se asume que durante el entrenamiento se utilizó LabelEncoder y que las clases son "No" y "Yes"
le = LabelEncoder()
le.classes_ = np.array(["No", "Yes"])  # Asegúrate de que el orden de clases coincide con el usado en el entrenamiento
y_pred_inversed = le.inverse_transform(y_pred)

# 5. Crear un DataFrame con los resultados
# Se asume que df_comp tiene una columna 'EmployeeID'
df_result = pd.DataFrame({
    "EmployeeID": df_comp["EmployeeID"],
    "Attrition_Pred": y_pred_inversed
})

# 6. Guardar las predicciones en un archivo CSV
output_path = "predicciones.csv"
df_result.to_csv(output_path, index=False)
print(f"Archivo '{output_path}' guardado correctamente.")
