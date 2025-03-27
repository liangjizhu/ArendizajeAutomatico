import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.dummy import DummyClassifier


# Fijamos la semilla para la reproducibilidad (NIA)
SEED = 495723
np.random.seed(SEED)

# 1. Cargamos el dataset
data_path = "lab1/data/attrition_availabledata_05.csv"
df = pd.read_csv(data_path)

# 2. Dimensiones y primeras filas
print("En esta primera exploración hemos podido observar que la dimensión de nuestro dataset es:", df.shape)
print("\nPor ejemplo, las 5 primeras filas del dataset se ven así:")
# display(df.head())

# 3. Información general de columnas
print("\nInformación general del DataFrame:")
df.info()

# 4. Tipos de variables
print("\nLos tipos de variable de las 31 columnas:")
print(df.dtypes)

# 5. Estadísticas descriptivas de columnas numéricas
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("\nEstadísticas descriptivas (numéricas), para ello seleccionamos las de tipo entero y coma floantante:")
# display(df[num_cols].describe())

# 6. Revisión de valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# 7. Análisis de columnas categóricas
cat_cols = df.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    print("\nColumnas categóricas y sus valores únicos:")
    for col in cat_cols:
        print(f"\nColumna: {col}")
        print("Valores únicos:", df[col].unique())
        print("Conteo de valores:\n", df[col].value_counts(dropna=False))
else:
    print("\nNo se encontraron columnas categóricas (object).")

# 8. Verificar filas duplicadas
dup_rows = df.duplicated().sum()
print(f"\nNúmero de filas duplicadas: {dup_rows}")
print("\nNo se han encontrado filas duplicadas")

# 9. Columnas constantes o casi constantes
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"La columna '{col}' es constante (valor único: {df[col].unique()[0]})")

# 10. Distribución de la variable objetivo (si existe la columna 'Attrition')
if 'Attrition' in df.columns:
    print("\nDistribución de la variable objetivo (Attrition):")
    print(df['Attrition'].value_counts(dropna=False))

plt.figure(figsize=(5, 5))
sns.countplot(x='Attrition', data=df)
plt.title("Distribución de Attrition")
# plt.show()

# Eliminar columnas constantes e irrelevantes para el modelado:
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours']
df = df.drop(columns=cols_to_drop)
# Seleccionar las columnas numéricas y categóricas (excluyendo la variable objetivo 'Attrition')
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
# Excluir Attrition
num_cols = [col for col in num_cols if col != 'Attrition']
cat_cols = [col for col in cat_cols if col != 'Attrition']
# Codificar la variable objetivo
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Attrition'])
X = df.drop(columns=['Attrition'])

# Dividir en entrenamiento y test (holdout: 2/3 train, 1/3 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=SEED, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)

# Definir pipeline para variables categóricas
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numeric_pipelines = {
    'MinMax + median': Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())]),
    'Standard + median': Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]),
    'Robust + median': Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]),
    'MinMax + mean': Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', MinMaxScaler())]),
    'Standard + mean': Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]),
    'Robust + mean': Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', RobustScaler())])
}

inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
results_pipelines = {}

print("\nEvaluando distintos pipelines numéricos para KNN:")
for name, num_pipe in numeric_pipelines.items():
    preproc = ColumnTransformer(transformers=[
        ('num', num_pipe, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessing', preproc),
        ('knn', KNeighborsClassifier())
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=inner, scoring='balanced_accuracy')
    results_pipelines[name] = scores.mean()
    print(f"{name}: CV balanced accuracy = {scores.mean():.4f}")

# Seleccionar la mejor combinación
best_pipeline_name = max(results_pipelines, key=results_pipelines.get)
print("\nMejor pipeline numérico según KNN:", best_pipeline_name)

# La mejor opción de escalado fue "Robust + median", definimos el pipeline definitivo:
numeric_pipeline_best = numeric_pipelines[best_pipeline_name]
preprocessing_pipeline = ColumnTransformer(transformers=[
    ('num', numeric_pipeline_best, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# KNN default
knn_default = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('knn', KNeighborsClassifier())
])
start_time = time.time()
knn_default.fit(X_train, y_train)
knn_default_time = time.time() - start_time
knn_default_scores = cross_val_score(knn_default, X_train, y_train, cv=inner, scoring='balanced_accuracy')
print("\nKNN default:")
print("Tiempo de entrenamiento:", knn_default_time, "segundos")
print("CV Balanced Accuracy:", knn_default_scores.mean())

# Árboles default
tree_default = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('tree', DecisionTreeClassifier(random_state=SEED))
])
start_time = time.time()
tree_default.fit(X_train, y_train)
tree_default_time = time.time() - start_time
tree_default_scores = cross_val_score(tree_default, X_train, y_train, cv=inner, scoring='balanced_accuracy')
print("\nÁrboles default:")
print("Tiempo de entrenamiento:", tree_default_time, "segundos")
print("CV Balanced Accuracy:", tree_default_scores.mean())

# Evaluación con un modelo trivial (dummy) para referencia
dummy = Pipeline(steps=[
    ('dummy', DummyClassifier(strategy='most_frequent'))
])
dummy_scores = cross_val_score(dummy, X_train, y_train, cv=inner, scoring='balanced_accuracy')
print("\nModelo Dummy (estrategia 'most_frequent') CV Balanced Accuracy:", dummy_scores.mean())

# KNN HPO
knn_param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}
grid_search_knn = GridSearchCV(estimator=knn_default, param_grid=knn_param_grid, cv=inner, scoring='balanced_accuracy')
start_time = time.time()
grid_search_knn.fit(X_train, y_train)
knn_hpo_time = time.time() - start_time
print("\nKNN HPO:")
print("Mejores parámetros:", grid_search_knn.best_params_)
print("Mejor CV Balanced Accuracy:", grid_search_knn.best_score_)
print("Tiempo de HPO para KNN:", knn_hpo_time, "segundos")

# Árboles HPO
tree_param_grid = {
    'tree__max_depth': [None, 10, 15, 20, 30],
    'tree__min_samples_split': [2, 5, 10],
    'tree__min_samples_leaf': [1, 2, 4]
}
grid_search_tree = GridSearchCV(estimator=tree_default, param_grid=tree_param_grid, cv=inner, scoring='balanced_accuracy')
start_time = time.time()
grid_search_tree.fit(X_train, y_train)
tree_hpo_time = time.time() - start_time
print("\nÁrboles HPO:")
print("Mejores parámetros:", grid_search_tree.best_params_)
print("Mejor CV Balanced Accuracy:", grid_search_tree.best_score_)
print("Tiempo de HPO para Árboles:", tree_hpo_time, "segundos")

knn_results = grid_search_knn.cv_results_['mean_test_score']
neighbors = knn_param_grid['knn__n_neighbors']
# Como GridSearch evalúa combinaciones, se puede extraer el efecto de n_neighbors para un valor fijo de weights y metric,
# por ejemplo, weights='distance' y metric='manhattan'.
selected_indices = [i for i, params in enumerate(grid_search_knn.cv_results_['params'])
                    if params['knn__weights']=='distance' and params['knn__metric']=='manhattan']
neighbors_scores = [grid_search_knn.cv_results_['mean_test_score'][i] for i in selected_indices]

plt.figure(figsize=(16,9))
plt.plot(neighbors, neighbors_scores, marker='o', linestyle='-')
plt.xlabel("Número de vecinos")
plt.ylabel("CV Balanced Accuracy")
plt.title("Efecto de n Vecinos en KNN\n(weights='distancia', métrica='manhattan')")
# plt.show()

tree_results = grid_search_tree.cv_results_['mean_test_score']
max_depths = tree_param_grid['tree__max_depth']
# Filtramos para un valor fijo de min_samples_split y min_samples_leaf, por ejemplo, min_samples_split=2, min_samples_leaf=1
selected_indices_tree = [i for i, params in enumerate(grid_search_tree.cv_results_['params'])
                         if params['tree__min_samples_split']==2 and params['tree__min_samples_leaf']==1]
depth_scores = [grid_search_tree.cv_results_['mean_test_score'][i] for i in selected_indices_tree]

plt.figure(figsize=(16, 9))
plt.plot(max_depths, depth_scores, marker='o', linestyle='-')
plt.xlabel("max_depth")
plt.ylabel("CV Balanced Accuracy")
plt.title("Efecto de max_depth en Árboles\n(min_samples_split=2, min_samples_leaf=1)")
# plt.show()

# Para esta gráfica, fijamos max_depth a un valor (por ejemplo, 15) y min_samples_leaf a 1
# Se extraen las combinaciones con max_depth == 15 y min_samples_leaf == 1
selected_indices_split = [i for i, params in enumerate(grid_search_tree.cv_results_['params'])
                          if params['tree__max_depth'] == 15 and params['tree__min_samples_leaf'] == 1]
split_values = [grid_search_tree.cv_results_['params'][i]['tree__min_samples_split'] for i in selected_indices_split]
split_scores = [grid_search_tree.cv_results_['mean_test_score'][i] for i in selected_indices_split]

plt.figure(figsize=(16,9))
plt.plot(split_values, split_scores, marker='o', linestyle='-')
plt.xlabel("min_samples_split")
plt.ylabel("CV Balanced Accuracy")
plt.title("Efecto de min_samples_split en Árboles\n(max_depth=15, min_samples_leaf=1)")
# plt.show()

# Definimos los pipelines, categórico y numérico
# aunque ya estaban definidas previamente para el KNN y los Árboles
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Combinar ambos pipelines:
preprocessing_pipeline = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

lr_no_reg_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('log_reg', LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=SEED))
])
# Entrenar el pipeline en los datos de entrenamiento (X_train, y_train)
start_time = time.time()
lr_no_reg_pipeline.fit(X_train, y_train)
lr_no_reg_time = time.time() - start_time

# Predecir usando el pipeline (se aplican todas las transformaciones)
y_pred_lr_no_reg = lr_no_reg_pipeline.predict(X_train)
print("Logistic Regression (sin regularización) Tiempo de entrenamiento: {:.4f} segundos".format(lr_no_reg_time))
print("Balanced Accuracy (LR sin regularización): {:.4f}".format(balanced_accuracy_score(y_train, y_pred_lr_no_reg)))

lr_l1 = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('log_reg', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=SEED))
])
start_time = time.time()
lr_l1.fit(X_train, y_train)
lr_l1_time = time.time() - start_time
y_pred_lr_l1 = lr_l1.predict(X_train)
print("\nLogistic Regression (L1 regularización) Tiempo de entrenamiento: {:.4f} segundos".format(lr_l1_time))
print("Balanced Accuracy (LR L1): {:.4f}".format(balanced_accuracy_score(y_train, y_pred_lr_l1)))

# Extraer los nombres de las características resultantes del preprocesamiento:
feature_names = lr_l1.named_steps['preprocessing'].get_feature_names_out()
# Construir la serie con los coeficientes y los nombres correctos:
coef_lr_l1 = lr_l1.named_steps['log_reg'].coef_[0]
feature_importance_lr = pd.Series(coef_lr_l1, index=feature_names)
print("\nImportancia de atributos (coeficientes) en LR L1:")
print(feature_importance_lr.sort_values(ascending=False))

lr_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('log_reg', LogisticRegression(max_iter=1000, solver='liblinear', random_state=SEED))
])
lr_param_grid = {
    'log_reg__C': [0.01, 0.1, 1, 10, 100],
    'log_reg__penalty': ['l1', 'l2']
}
grid_search_lr = GridSearchCV(estimator=lr_pipeline, param_grid=lr_param_grid, cv=inner, scoring='balanced_accuracy')
start_time = time.time()
grid_search_lr.fit(X_train, y_train)
lr_hpo_time = time.time() - start_time
print("\n[HPO] Mejor LR parameters:", grid_search_lr.best_params_)
print("[HPO] Mejor CV Balanced Accuracy (LR): {:.4f}".format(grid_search_lr.best_score_))
print("Tiempo HPO LR: {:.4f} seconds".format(lr_hpo_time))

svm_default_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('svm', SVC(random_state=SEED))
])

start_time = time.time()
svm_default_pipeline.fit(X_train, y_train)
svm_default_time = time.time() - start_time
y_pred_svm_default = svm_default_pipeline.predict(X_train)
print("\nSVM (default) Training Time: {:.4f} seconds".format(svm_default_time))
print("Balanced Accuracy (SVM default): {:.4f}".format(balanced_accuracy_score(y_train, y_pred_svm_default)))

# Creamos nuevamente el pipeline para SVM (para mayor claridad)
svm_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('svm', SVC(random_state=SEED))
])

# Definimos el grid de hiperparámetros para SVM
svm_param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf']
}

# Búsqueda de hiperparámetros con validación cruzada interna
grid_search_svm = GridSearchCV(estimator=svm_pipeline, param_grid=svm_param_grid, cv=inner, scoring='balanced_accuracy')
start_time = time.time()
grid_search_svm.fit(X_train, y_train)
svm_hpo_time = time.time() - start_time
print("\n[HPO] Mejor SVM parameters:", grid_search_svm.best_params_)
print("[HPO] Mejor CV Balanced Accuracy (SVM): {:.4f}".format(grid_search_svm.best_score_))
print("Tiempo HPO SVM: {:.4f} seconds".format(svm_hpo_time))

# Extracción de importancia de atributos para SVM
if grid_search_svm.best_params_['svm__kernel'] == 'linear':
    svm_best_linear = grid_search_svm.best_estimator_.named_steps['svm']
    svm_coef = svm_best_linear.coef_[0]
    feature_importance_svm = pd.Series(svm_coef, index=X_train.columns)
    print("\nImportancia de atributos (coeficientes) en SVM lineal:")
    print(feature_importance_svm.sort_values(ascending=False))
else:
    print("\nSVM con kernel no lineal (RBF) no permite extraer directamente importancia de atributos.")
