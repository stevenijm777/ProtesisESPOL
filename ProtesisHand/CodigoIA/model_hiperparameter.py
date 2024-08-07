import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
except ImportError:
    from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Cambiar el backend de Matplotlib
plt.switch_backend('Agg')

# Procesamiento de Datos

# Obtener la ruta absoluta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta a la subcarpeta que contiene los archivos de datos
subfolder = 'HandLeftP2'  # Cambiar aquí para trabajar con otra subcarpeta
folder_path = os.path.join(script_dir, 'Datos', subfolder)

# Función para cargar los datos desde un archivo de texto
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [int(line.strip()) for line in file]
    return np.array(data)

# Archivos de datos
files = ["BrazoArriba.txt", "Descanso.txt", "AbreYCierra.txt", "Pinza.txt"]

# Inicializar listas para datos y etiquetas
all_data = []
all_labels = []

# Cargar los datos y crear etiquetas
for i, file_name in enumerate(files):
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        data = load_data(file_path)
        labels = np.full(len(data), i)
        all_data.append(data)
        all_labels.append(labels)
    else:
        print(f"Archivo no encontrado: {file_path}")

# Concatenar los datos y las etiquetas
X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)

# Normalizar los datos
X = X / 1023.0

# Asegurarse de que X tenga la forma correcta (n_samples, n_features, 1)
X = X.reshape(-1, 1, 1)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(-1, 1, 1)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(-1, 1, 1)

print("Datos procesados y listos para el entrenamiento del modelo.")

def create_model(optimizer='adam', kernel_size=1, filters=32, dense_units=100, dropout_rate=0.0):
    model = Sequential([
        Input(shape=(1, 1)),
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(5, activation='softmax')  # 5 movimientos diferentes
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Crear el clasificador de Keras
model = KerasClassifier(build_fn=create_model, verbose=0)

# Definir los hiperparámetros para la búsqueda de cuadrícula
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 50, 100],
    'optimizer': ['adam', 'rmsprop'],
    'kernel_size': [1, 2, 3],
    'filters': [16, 32, 64],
    'dense_units': [50, 100, 150],
    'dropout_rate': [0.0, 0.2, 0.5]
}

# Realizar la búsqueda de cuadrícula
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Imprimir los mejores parámetros y la mejor puntuación
print(f"Mejores Parámetros: {grid_result.best_params_}")
print(f"Mejor Precisión: {grid_result.best_score_}")
