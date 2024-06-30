import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

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
files = ["BrazoArriba.txt", "Descanso.txt", "Codo.txt", "AbreYCierra.txt", "Pinza.txt"]

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

# Construcción del modelo
model = Sequential([
    Input(shape=(1, 1)),
    Conv1D(filters=32, kernel_size=1, activation='relu'),  # kernel_size ajustado a 1
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(5, activation='softmax')  # 5 movimientos diferentes
])

# Compilación del modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluación del modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
