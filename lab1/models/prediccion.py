import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# 1. Cargar el modelo final
modelo_final = joblib.load("modelo_final.pkl")
print("Modelo cargado correctamente.")

# 2. Cargar el dataset de competición
data_comp_path = "/Users/alfredofelices/PycharmProjects/ArendizajeAutomatico/lab1/data/attrition_competition_05.csv"  # Ajusta la ruta si es necesario
df_comp = pd.read_csv(data_comp_path)
print("Dimensiones del dataset de competición:", df_comp.shape)

# 3. Aplicar el mismo preprocesamiento que en el entrenamiento
# (Definir las mismas columnas numéricas y categóricas)
# Supongamos que usamos las mismas variables que en el entrenamiento:
num_cols = df_comp.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_comp.select_dtypes(include=['object']).columns.tolist()
cols_excluir = ['EmployeeID', 'EmployeeCount']
num_cols = [col for col in num_cols if col not in cols_excluir]
cat_cols = [col for col in cat_cols if col not in cols_excluir]

# Reconstruir el pipeline de preprocesamiento utilizado en el entrenamiento
# Nota: Es crucial que este pipeline sea idéntico al usado para entrenar el modelo.
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessing_pipeline = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# Aplicar el pipeline de preprocesamiento al dataset de competición
df_comp_preprocessed = preprocessing_pipeline.fit_transform(df_comp)

# 4. Realizar las predicciones
predicciones = modelo_final.predict(df_comp)
# O si tu modelo ya incluye el pipeline completo, puedes usarlo directamente:
# predicciones = modelo_final.predict(df_comp)

# 5. Guardar las predicciones en un archivo CSV
df_result = pd.DataFrame({
    'EmployeeID': df_comp['EmployeeID'],  # Asumiendo que la columna EmployeeID existe
    'Attrition_Pred': predicciones
})
df_result.to_csv("predicciones.csv", index=False)
print("Archivo 'predicciones.csv' guardado correctamente.")
