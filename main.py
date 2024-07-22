import tensorflow as tf

# Carga el modelo Keras
model = tf.keras.models.load_model('modelo_entrenado.keras')

# Convertir el modelo a formato TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TFLite
with open('modelo_entrenado.tflite', 'wb') as f:
    f.write(tflite_model)
    
def tflite_to_header(input_file, output_file):
    with open(input_file, 'rb') as f:
        byte_array = f.read()

    with open(output_file, 'w') as f:
        f.write('#ifndef MODELO_ENTRENADO_H\n')
        f.write('#define MODELO_ENTRENADO_H\n')
        f.write('unsigned char modelo_entrenado_tflite[] = {')
        for i, byte in enumerate(byte_array):
            if i % 12 == 0:
                f.write('\n    ')
            f.write('0x{:02x}, '.format(byte))
        f.write('\n};\n')
        f.write('unsigned int modelo_entrenado_tflite_len = {};\n'.format(len(byte_array)))
        f.write('#endif // MODELO_ENTRENADO_H\n')

# Usar la función para convertir el archivo
tflite_to_header('modelo_entrenado.tflite', 'modelo_entrenado.h')
print()