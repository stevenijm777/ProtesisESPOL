import matplotlib.pyplot as plt
import os

def plot_emg_data(file_path, title, save_dir):
    # Leer los datos del archivo
    with open(file_path, 'r') as file:
        data = file.read().split()
    
    # Convertir los datos a números enteros
    data = [int(value) for value in data]
    
    # Crear el gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='EMG Signal')
    plt.title(title)
    plt.xlabel('Muestra')
    plt.ylabel('Valor EMG')
    plt.legend()
    plt.grid(True)
    
    # Guardar el gráfico en la carpeta correspondiente
    output_path = os.path.join(save_dir, f"{title}.png")
    plt.savefig(output_path)
    plt.close()

# Obtener la ruta absoluta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directorios de los datos
base_dir = os.path.join(script_dir, 'Datos')
save_base_dir = os.path.join(script_dir, 'DatosImgEMG')

# Subcarpetas de los datos
subfolders = ['HandLeftP1', 'HandLeftP2', 'HandRight']

# Generar gráficos para cada archivo en cada subcarpeta
for subfolder in subfolders:
    data_dir = os.path.join(base_dir, subfolder)
    save_dir = os.path.join(save_base_dir, subfolder)
    
    # Verificar si la carpeta de datos existe
    if not os.path.exists(data_dir):
        print(f"Directorio no encontrado: {data_dir}")
        continue
    
    os.makedirs(save_dir, exist_ok=True)
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            title = os.path.splitext(filename)[0]
            plot_emg_data(file_path, title, save_dir)
            print(f"Gráfico guardado en: {save_dir}")

print("Procesamiento completo.")
