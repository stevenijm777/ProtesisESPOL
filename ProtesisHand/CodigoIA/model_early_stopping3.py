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
subfolder = 'MyHand2'  # Cambiar aquí para trabajar con otra subcarpeta
folder_path = os.path.join(script_dir, 'Datos', subfolder)

# Función para cargar los datos desde un archivo de texto
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [int(line.strip()) for line in file]
    return np.array(data)

# Archivos de datos
files = ["AbreYCierra.txt", "BrazoArriba.txt","Codo.txt", "Descanso.txt", "Pinza.txt"]

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

# Imprimir los parámetros del StandardScaler
print("StandardScaler mean:", scaler.mean_)
print("StandardScaler scale:", scaler.scale_)

# Construcción del modelo
model = Sequential([
    Input(shape=(1, 1)),
    Conv1D(filters=32, kernel_size=1, activation='relu'),  # kernel_size ajustado a 1
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(100, activation='sigmoid'),
    Dense(5, activation='softmax')  # Ajustado a 5 movimientos diferentes
])

# Compilación del modelo con una tasa de aprendizaje ajustada
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Definir Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento del modelo con Early Stopping
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluación del modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Realizar predicciones con el modelo entrenado
predictions = model.predict(X_test)

# Convertir las predicciones a etiquetas de clase
predicted_classes = np.argmax(predictions, axis=1)

# Imprimir las predicciones
print(f'Predicciones: {predicted_classes}')
print(f'Labels reales: {y_test}')

# Evaluar la precisión
accuracy = accuracy_score(y_test, predicted_classes)
report = classification_report(y_test, predicted_classes)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Guardar el modelo entrenado en el formato recomendado por Keras
model.save('modelo_entrenado_relu2.keras')
print("Modelo guardado como 'modelo_entrenado_relu2.keras'")

# 1. Matriz de Confusión

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)   
    plt.figure(figsize=(10, 7)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)    
    plt.xlabel('Predicted') 
    plt.ylabel('True')  
    plt.title('Confusion Matrix')   
    plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x para que sean más legibles
    plt.yticks(rotation=45, ha='right')  # Rotar las etiquetas del eje y para que sean más legibles
    plt.tight_layout()  # Ajustar el layout para que todo se vea bien
    plt.savefig('confusion_matrix.png') 
    plt.close()

# Clases de ejemplo
classes = ["Abre y Cierra",  "Brazo Arriba", "Codo", "Descanso", "Pinza"]

# Generar la matriz de confusión
plot_confusion_matrix(y_test, predicted_classes, classes)

def plot_training_history(history):
    # Resumir historia para precisión
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    
    # Resumir historia para pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    
    plt.savefig('training_history.png')
    plt.close()

# Generar las gráficas de historia de entrenamiento
plot_training_history(history)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curves(y_test, predictions, n_classes):
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 5))
    colors = plt.cm.get_cmap('tab10', n_classes)

    for i, color in enumerate(colors.colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig('roc_curves.png')
    plt.close()

# Generar las curvas ROC
plot_roc_curves(y_test, predictions, n_classes=5)


def plot_error_distribution(y_true, y_pred, classes):
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=np.arange(-0.5, len(classes)-0.5, 1), alpha=0.7, color='blue', edgecolor='black')
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=45, ha='right')  # Etiquetas en el eje x
    plt.grid(True)
    plt.tight_layout()  # Ajustar el layout para que todo se vea bien
    plt.savefig('error_distribution.png')
    plt.close()

# Generar la distribución de errores con etiquetas de clase
plot_error_distribution(y_test, predicted_classes, classes)
