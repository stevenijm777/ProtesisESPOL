import time
import serial

# Configurar la conexión serie
try:
    ser = serial.Serial('COM8', 115200)  # Reemplaza 'COM8' con el puerto serie correcto
except serial.SerialException as e:
    print(f"No se pudo abrir el puerto: {e}")
    exit()

# Lista de clases para enviar
clases = [1, 2, 3, 4, 5]

# Enviar mensajes cada cierto tiempo
try:
    for clase in clases:
        ser.write(f"{clase}\n".encode())
        print(f"Clase enviada: {clase}")
        time.sleep(2)  # Espera 2 segundos antes de enviar el siguiente mensaje
except KeyboardInterrupt:
    print("Interrupción del usuario. Cerrando puerto.")
finally:
    ser.close()
    print("Puerto cerrado.")
