import serial
import time

file_path = 'data.txt'  # Ruta del archivo donde se guardarán los datos

try:
    ser = serial.Serial('COM7', 9600)
    time.sleep(2)  # Espera a que se establezca la conexión

    with open(file_path, 'a') as file:  # Abrir el archivo en modo de añadir
        while True:
            if ser.in_waiting > 0:  # Comprobar si hay datos en el búfer de entrada
                line = ser.readline().decode('utf-8').rstrip()
                print(line)
                file.write(line + '\n')  # Escribir la línea en el archivo
except serial.SerialException as e:
    print(f'Error abriendo el puerto serial: {e}')
finally:
    if ser.is_open:
        ser.close()
