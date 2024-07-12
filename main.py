import tensorflow as tf

# Carga el modelo Keras
model = tf.keras.models.load_model('modelo_entrenado.keras')

# Convertir el modelo a formato TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TFLite
with open('modelo_entrenado.tflite', 'wb') as f:
    f.write(tflite_model)
