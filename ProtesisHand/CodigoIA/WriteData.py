import serial
import time

# Configuración del puerto serial
ser = serial.Serial('COM8', 115200)  # Cambia 'COM8' por el puerto adecuado en tu sistema
time.sleep(2)  # Esperar a que se establezca la conexión

# Abrir el archivo en modo de escritura
with open('BrazoArriba.txt', 'w') as file:
    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip()  # Leer la línea del puerto serial
                print(f"Received: {data}")  # Imprimir los datos recibidos
                file.write(data + '\n')  # Escribir los datos en el archivo
                file.flush()  # Asegurarse de que los datos se escriban inmediatamente
    except KeyboardInterrupt:
        print("Interrumpido por el usuario")
