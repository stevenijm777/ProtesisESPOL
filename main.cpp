#include <Arduino.h>
#include <ESP32Servo.h>
#include "modelo_entrenado.h"  // Archivo de cabecera con el modelo entrenado

// Asegúrate de que las rutas son correctas según la ubicación de las bibliotecas
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Configuración de TFLite
namespace {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::AllOpsResolver resolver;
    constexpr int tensor_arena_size = 16 * 1024;  // Aumentar el tamaño del tensor arena
    uint8_t tensor_arena[tensor_arena_size];
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
}

Servo myservo;  // Crea un objeto servo para controlar un servomotor
QueueHandle_t dataQueue;  // Cola para comunicación entre tareas

// Variables de escalado de StandardScaler en Python
const float scaler_mean = 0.15850927;
const float scaler_scale = 0.04388263;

// Función para inicializar el modelo TFLite
void initModel() {
    // Cargar el modelo
    model = tflite::GetModel(modelo_entrenado_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Modelo incompatible con la versión de TFLite.");
        return;
    }

    // Crear el intérprete
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, tensor_arena_size, &micro_error_reporter);
    interpreter = &static_interpreter;

    // Alocar tensores
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("Fallo al alocar tensores.");
        return;
    }

    // Obtener tensores de entrada y salida
    input = interpreter->input(0);
    output = interpreter->output(0);
}

// Función para realizar la predicción con el modelo TFLite
int predict(float dato) {
    // Preparar el tensor de entrada
    input->data.f[0] = dato;

    // Ejecutar el intérprete
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Fallo en la invocación del intérprete.");
        return -1;
    }

    // Obtener el tensor de salida y encontrar el índice de la clase con mayor probabilidad
    float max_value = -1;
    int max_index = -1;
    for (int i = 0; i < output->dims->data[output->dims->size - 1]; ++i) {
        if (output->data.f[i] > max_value) {
            max_value = output->data.f[i];
            max_index = i;
        }
    }
    return max_index;
}

// Tarea para recibir datos por comunicación serial y procesarlos
void receiveData(void *pvParameters) {
    while (1) {
        if (Serial.available() > 0) {
            String inputStr = Serial.readStringUntil('\n');  // Leer hasta el final de la línea
            float dato = inputStr.toFloat();  // Convertir a float

            // Normalizar y escalar datos como en Python
            float dato_normalizado = dato / 1023.0;  // Normalizar
            float dato_escalado = (dato_normalizado - scaler_mean) / scaler_scale;  // Escalar usando parámetros de StandardScaler

            // Crear estructura para almacenar los datos
            struct {
                float original;
                float normalizado;
                float escalado;
            } datos_procesados = {dato, dato_normalizado, dato_escalado};

            // Enviar datos a la cola
            xQueueSend(dataQueue, &datos_procesados, portMAX_DELAY);
        }
        delay(10);  // Evitar saturar la CPU
    }
}

// Función para controlar el servo según la clase predicha
void controlarServo(int clase) {
    switch (clase) {
        case 0: // Brazo Arriba
            myservo.write(90);
            break;
        case 1: // Descanso
            myservo.write(180);
            break;
        case 2: // Codo
            for (int pos = 0; pos <= 360; pos += 10) {
                myservo.write(pos);
                delay(100);
            }
            break;
        case 3: // Abre y Cierra
            for (int pos = 0; pos <= 270; pos += 10) {
                myservo.write(pos);
                delay(75); // Duración total de 2 segundos
            }
            for (int pos = 270; pos >= 0; pos -= 10) {
                myservo.write(pos);
                delay(75); // Duración total de 2 segundos
            }
            break;
        case 4: // Pinza
            myservo.write(0);
            delay(1500);
            myservo.write(45);
            delay(1500);
            break;
        default: // Otro
            myservo.write(0);
            break;
    }
}

// Tarea para realizar predicciones y mover el servomotor
void predictAndMoveServo(void *pvParameters) {
    struct {
        float original;
        float normalizado;
        float escalado;
    } datos_procesados;

    while (1) {
        // Esperar hasta recibir un dato de la cola
        if (xQueueReceive(dataQueue, &datos_procesados, portMAX_DELAY) == pdTRUE) {
            int clase = predict(datos_procesados.escalado);  // Realizar predicción usando el modelo

            // Controlar el servo según la clase predicha
            controlarServo(clase);
        }
    }
}

void setup() {
    myservo.attach(12);  // Asigna el pin 12 al servomotor
    Serial.begin(115200);  // Inicializa la comunicación serie para debug
    initModel();  // Inicializar el modelo TFLite

    // Crear la cola para comunicación entre tareas
    dataQueue = xQueueCreate(10, sizeof(float) * 3);

    // Crear las tareas en diferentes núcleos
    xTaskCreatePinnedToCore(receiveData, "ReceiveData", 2048, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(predictAndMoveServo, "PredictAndMoveServo", 2048, NULL, 1, NULL, 1);
}

void loop() {
    // El loop principal no necesita hacer nada, ya que las tareas se están ejecutando
}
