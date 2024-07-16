import time
import serial

# Configurar la conexión serie
try:
    ser = serial.Serial('COM8', 115200)  # Reemplaza 'COM8' con el puerto serie correcto
except serial.SerialException as e:
    print(f"No se pudo abrir el puerto: {e}")
    exit()

# Enviar mensajes cada cierto tiempo
try:
    while True:
        ser.write(b"Hola ESP32\n")
        print("Mensaje enviado: Hola ESP32")
        time.sleep(2)  # Espera 2 segundos antes de enviar el siguiente mensaje
except KeyboardInterrupt:
    print("Interrupción del usuario. Cerrando puerto.")
finally:
    ser.close()
    print("Puerto cerrado.")
