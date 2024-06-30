import tensorflow as tf

# Cargar el modelo guardado
model = tf.keras.models.load_model('modelo_entrenado.keras')

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo de TensorFlow Lite
with open('modelo_entrenado.tflite', 'wb') as f:
    f.write(tflite_model)
print("Modelo convertido y guardado como 'modelo_entrenado.tflite'")
