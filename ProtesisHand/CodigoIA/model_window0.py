import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Comentar o eliminar la siguiente línea si quieres ver las gráficas directamente
# plt.switch_backend('Agg')  # Cambiar el backend de Matplotlib para guardar imágenes

# Ruta de datos
script_dir = os.path.dirname(os.path.abspath(__file__))  # Obtener la ruta absoluta del directorio del script
subfolder = 'HandLeftP2'  # Subcarpeta con los datos
folder_path = os.path.join(script_dir, 'Datos', subfolder)

# Parámetros de ventana
window_size = 50  # Tamaño de la ventana
step_size = 5    # Paso entre ventanas consecutivas (solapamiento)

# Función para cargar datos y segmentar en ventanas
def load_data_and_segment(file_path):
    with open(file_path, 'r') as file:
        data = np.array([int(line.strip()) for line in file])
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)

# Carga y etiquetado de datos
all_data = []
all_labels = []
files = ["AbreYCierra.txt", "BrazoArriba.txt", "Descanso.txt", "Pinza.txt"]
for i, file_name in enumerate(files):
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        data_windows = load_data_and_segment(file_path)
        labels = np.full(len(data_windows), i)
        all_data.append(data_windows)
        all_labels.append(labels)

# Preparar datos para entrenamiento
X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)
X = X / 1023.0  # Normalización

# Reshape de X para que sea compatible con Conv1D
X = X.reshape(X.shape[0], X.shape[1], 1)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, window_size)).reshape(-1, window_size, 1)
X_test = scaler.transform(X_test.reshape(-1, window_size)).reshape(-1, window_size, 1)

# Construcción del modelo
model = Sequential([
    Input(shape=(window_size, 1)),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento con Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluación
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Funciones adicionales para análisis (matriz de confusión, historia de entrenamiento, etc.)

# Función para graficar la matriz de confusión
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()  # Mostrar la figura directamente
    plt.close()

# Clases para las etiquetas
classes = ["Abre y Cierra", "Brazo Arriba", "Descanso", "Pinza"]

# Generar y mostrar la matriz de confusión
plot_confusion_matrix(y_test, np.argmax(model.predict(X_test), axis=1), classes)

model.save('modelo_entrenado_por_ventanas.keras')
print("Modelo guardado como 'modelo_entrenado_por_ventanas.keras'")
