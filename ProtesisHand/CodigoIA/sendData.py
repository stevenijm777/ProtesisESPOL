import serial
import time
import os
from collections import Counter

# Configurar la conexión serie
ser = serial.Serial('COM8', 115200)  # Reemplaza 'COM8' con el puerto serie correcto

# Obtener la ruta absoluta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta a la subcarpeta que contiene los archivos de datos
subfolder = 'HandLeftP2'
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

# Leer los primeros 100 datos de un archivo y enviarlos
if os.path.exists(file_path):
    data = load_data(file_path, limit=100)
    
    for dato in data:
        ser.write(f"{dato}\n".encode())
        print(f"Datos enviados: {dato}")
        time.sleep(0.1)  # Espera 100 ms antes de enviar el siguiente dato

    # Hacer un resumen de las predicciones enviadas
    resumen = Counter(data)
    print(f"Resumen de datos enviados para Pinza.txt: {dict(resumen)}")
else:
    print(f"Archivo no encontrado: {file_path}")

ser.close()

