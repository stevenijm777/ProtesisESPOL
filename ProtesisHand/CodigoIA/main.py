import tensorflow as tf

# Lista de dispositivos físicos disponibles
devices = tf.config.list_physical_devices()
for device in devices:
    print(device)
    
