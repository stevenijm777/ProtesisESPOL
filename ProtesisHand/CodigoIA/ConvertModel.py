import tensorflow as tf

# Cargar el modelo Keras entrenado
model = tf.keras.models.load_model('modelo_entrenado_relu.keras')

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TensorFlow Lite en un archivo
tflite_model_filename = 'modelo_entrenado_relu.tflite'
with open(tflite_model_filename, 'wb') as f:
    f.write(tflite_model)
print(f"Modelo convertido a TensorFlow Lite y guardado como '{tflite_model_filename}'")

# Convertir el modelo TensorFlow Lite a un archivo de cabecera (.h)
def convert_to_c_header(tflite_model, output_filename):
    hex_content = ', '.join('0x{:02x}'.format(b) for b in tflite_model)
    header_content = f"""
#ifndef {output_filename.upper().replace('.', '_')}
#define {output_filename.upper().replace('.', '_')}

const unsigned char model[] = {{
    {hex_content}
}};
const int model_len = {len(tflite_model)};

#endif
"""
    with open(output_filename, 'w') as f:
        f.write(header_content)

# Nombre del archivo de cabecera
header_filename = 'modelo_entrenado_relu.h'
convert_to_c_header(tflite_model, header_filename)
print(f"Modelo convertido a archivo de cabecera y guardado como '{header_filename}'")
