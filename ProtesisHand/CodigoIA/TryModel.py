import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Obtener la ruta absoluta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta a la subcarpeta que contiene los archivos de datos
subfolder = 'HandLeftP2'  # Cambiar aquí para trabajar con otra subcarpeta
folder_path = os.path.join(script_dir, 'Datos', subfolder)

# Ruta al archivo Pinza.txt
file_path = os.path.join(folder_path, 'Pinza.txt')

# Función para cargar los datos desde un archivo de texto
def load_data(file_path, limit=None):
    with open(file_path, 'r') as file:
        data = [int(line.strip()) for line in file]
    if limit:
        data = data[:limit]
    return data

# Cargar el modelo entrenado
model = load_model('modelo_entrenado.keras')

# Escalador (StandardScaler) parámetros
scaler_mean = 0.15850927
scaler_scale = 0.04388263

# Leer el archivo Pinza.txt línea por línea, normalizar, escalar y predecir
if os.path.exists(file_path):
    data = load_data(file_path, limit=100)
    predictions = []
    for dato in data:
        # Normalizar y escalar los datos
        dato_normalizado = dato / 1023.0
        dato_escalado = (dato_normalizado - scaler_mean) / scaler_scale

        # Preparar el dato para la predicción
        dato_escalado = np.array(dato_escalado).reshape(-1, 1, 1)
        
        # Realizar la predicción
        prediccion = model.predict(dato_escalado)
        clase_predicha = np.argmax(prediccion, axis=1)[0]
        predictions.append(clase_predicha)
        print(f'Dato: {dato}, Normalizado: {dato_normalizado}, Escalado: {dato_escalado[0][0]}, Clase Predicha: {clase_predicha}')
    
    unique, counts = np.unique(predictions, return_counts=True)
    print(f'Resumen de predicciones para Pinza.txt: {dict(zip(unique, counts))}')
else:
    print(f"Archivo no encontrado: {file_path}")

