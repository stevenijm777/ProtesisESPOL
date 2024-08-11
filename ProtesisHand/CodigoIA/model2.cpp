#include <Arduino.h>
#include <ESP32Servo.h>
#include "modelo_entrenado_relu1.h"  // Usar el nuevo modelo entrenado
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Configuración de TFLite
namespace {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::AllOpsResolver resolver;
    constexpr int tensor_arena_size = 16 * 1024;
    uint8_t tensor_arena[tensor_arena_size];
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
}

Servo servoPinza;
Servo servoAbreCierra;
QueueHandle_t dataQueue;

//Modelo Redu 2
//const float scaler_mean = 0.20755254;
//const float scaler_scale = 0.09132263;
// Modelo Relu 1
const float scaler_mean = 0.19525006;
const float scaler_scale = 0.08936783;
const unsigned long predictionInterval = 2000;  // Intervalo de tiempo en milisegundos
unsigned long lastPredictionTime = 0;
int predicciones[9] = {0};  // Array para contar las predicciones de cada clase

void initModel() {
    model = tflite::GetModel(modelo_entrenado_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Modelo incompatible con la versión de TFLite.");
        return;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, tensor_arena_size, &micro_error_reporter);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("Fallo al alocar tensores.");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
}

  

int predict(float dato) {

    input->data.f[0] = dato;

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Fallo en la invocación del intérprete.");
        return -1;
    }

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


void readSensorData(void *pvParameters) {

    while (1) {
        int sensorValue = analogRead(34);  // Leer el valor del sensor en el pin 34
        // Mapear el valor del sensor de 0-4095 a 0-1023
        int mapeadoValue = map(sensorValue, 0, 4095, 0, 1023);
        Serial.println(mapeadoValue);  // Imprimir el valor mapeado para depuración
        float dato = static_cast<float>(mapeadoValue);
        float dato_normalizado = dato / 1023.0;
        float dato_escalado = (dato_normalizado - scaler_mean) / scaler_scale;
        struct {
            float original;
            float normalizado;
            float escalado;
        } datos_procesados = {dato, dato_normalizado, dato_escalado};
        xQueueSend(dataQueue, &datos_procesados, portMAX_DELAY);
        delay(100);  // Leer datos cada 100 ms
    }

}



void receiveData(void *pvParameters) {
    while (1) {
        if (Serial.available() > 0) {
            String inputStr = Serial.readStringUntil('\n');
            float dato = inputStr.toFloat();
            float dato_normalizado = dato / 1023.0;
            float dato_escalado = (dato_normalizado - scaler_mean) / scaler_scale;
            struct {
                float original;
                float normalizado;
                float escalado;
            } datos_procesados = {dato, dato_normalizado, dato_escalado};
            xQueueSend(dataQueue, &datos_procesados, portMAX_DELAY);
        }
        delay(100);
    }
}

void controlarServoPinza() {
    for (int pos = 0; pos <= 180; pos += 10) {
        servoPinza.write(pos);
        delay(200);
    }
    delay(2000);
    for (int pos = 180; pos >= 0; pos -= 10) {
        servoPinza.write(pos);
        delay(200);
    }
}

void controlarServoAbreCierra() {
    for (int pos = 0; pos <= 360; pos += 10) {
        servoAbreCierra.write(pos);
        delay(200);
    }
    delay(2000);
    for (int pos = 360; pos >= 0; pos -= 10) {
        servoAbreCierra.write(pos);
        delay(200);
    }
}

void controlarServos(int clase) {
    // Resetear ambos servos antes de mover el correcto
    servoPinza.write(90); // Poner el servo en posición neutral

    servoAbreCierra.write(90); // Poner el servo en posición neutral

    // Solo ejecutar movimientos si es "Abre y Cierra" o "Pinza"
    switch (clase) {
        case 0: // Abre y Cierra (Solo mover el servo en pin 14)
            controlarServoAbreCierra();
            break;
        case 1: // Balance Pie
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 2: // Brazo Arriba
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 3: // Codo
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 4: // Descanso
            controlarServoPinza();
            break;
        case 5: // Descanso2
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 6: // DescansoDePie
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 7: // Pinza
            controlarServoPinza();
            break;
        case 8: //Pinza2 
            controlarServoPinza();
            break;
        
        default: // No hacer nada para las demás clases
            servoPinza.write(90);      // Poner el servo en posición neutral
            servoAbreCierra.write(90); // Poner el servo en posición neutral
            break;
    }
}

void procesarPredicciones() {
    int claseMasFrecuente = 0;
    int maxConteo = 0;
    const int pesoAbreYCierra = 2;  // Peso adicional para la clase "Abre y Cierra"
    const int pesoPinza = 2;        // Peso adicional para la clase "Pinza"

    for (int i = 0; i < 5; i++) {  // Ajustado para 5 clases
        int conteoAjustado = predicciones[i];

        // Aplicar peso adicional a las clases "Abre y Cierra" y "Pinza"
        if (i == 0) { // "Abre y Cierra"
            conteoAjustado *= pesoAbreYCierra;
        } else if (i == 7 or  i == 8) { // "Pinza"
            conteoAjustado *= pesoPinza;
        }

        if (conteoAjustado > maxConteo) {
            maxConteo = conteoAjustado;
            claseMasFrecuente = i;
        }
    }

    Serial.print("Clase más frecuente: ");
    Serial.println(claseMasFrecuente);

    controlarServos(claseMasFrecuente);

    // Resetear el contador de predicciones
    for (int i = 0; i < 5; i++) {  // Ajustado para 5 clases
        predicciones[i] = 0;
    }
}


void predictAndMoveServo(void *pvParameters) {
    struct {
        float original;
        float normalizado;
        float escalado;
    } datos_procesados;

    while (1) {
        if (xQueueReceive(dataQueue, &datos_procesados, portMAX_DELAY) == pdTRUE) {
            int clase = predict(datos_procesados.escalado);
            if (clase >= 0 && clase < 5) {  // Ajustado para 5 clases
                predicciones[clase]++;
            }
            unsigned long currentTime = millis();
            if (currentTime - lastPredictionTime >= predictionInterval) {
                procesarPredicciones();
                lastPredictionTime = currentTime;
            }
        }
    }
}

void setup() {
    servoPinza.attach(12);
    servoAbreCierra.attach(14);
    Serial.begin(115200);
    initModel();

    dataQueue = xQueueCreate(10, sizeof(float) * 3);

    //xTaskCreatePinnedToCore(readSensorData, "ReadSensorData", 2048, NULL, 1, NULL, 0);
    //xTaskCreatePinnedToCore(receiveData, "ReceiveData", 2048, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(predictAndMoveServo, "PredictAndMoveServo", 2048, NULL, 1, NULL, 1);
}

void loop() {
    // El loop principal no necesita hacer nada, ya que las tareas se están ejecutando
}