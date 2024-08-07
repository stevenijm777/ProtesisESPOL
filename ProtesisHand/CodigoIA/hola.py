import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# Funciones de activación a probar
activation_functions = ['relu', 'tanh', 'sigmoid']

for activation_function in activation_functions:
    print(f"Entrenando modelo con función de activación: {activation_function}")
    
    # Construcción del modelo
    model = Sequential([
        Input(shape=(1, 1)),
        Conv1D(filters=32, kernel_size=1, activation='relu'),  # kernel_size ajustado a 1
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(100, activation=activation_function),
        Dense(4, activation='softmax')  # 4 movimientos diferentes
    ])

    # Compilación del modelo con una tasa de aprendizaje ajustada
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Definir Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenamiento del modelo con Early Stopping
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Evaluación del modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Función de activación: {activation_function} - Loss: {loss}, Accuracy: {accuracy}')

    # Realizar predicciones con el modelo entrenado
    predictions = model.predict(X_test)

    # Convertir las predicciones a etiquetas de clase
    predicted_classes = np.argmax(predictions, axis=1)

    # Evaluar la precisión
    accuracy = accuracy_score(y_test, predicted_classes)
    report = classification_report(y_test, predicted_classes)
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')

    # Guardar el modelo entrenado
    model_filename = f'modelo_entrenado_{activation_function}.keras'
    model.save(model_filename)
    print(f"Modelo guardado como '{model_filename}'")

    # Generar la matriz de confusión
    def plot_confusion_matrix(y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {activation_function}')
        plt.savefig(filename)
        plt.close()

    # Clases de ejemplo
    classes = ['Brazo Arriba', 'Descanso', 'Abre y Cierra', 'Pinza']

    # Guardar la matriz de confusión
    plot_confusion_matrix(y_test, predicted_classes, classes, f'confusion_matrix_{activation_function}.png')
