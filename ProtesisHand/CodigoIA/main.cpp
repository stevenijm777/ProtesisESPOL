#include <Arduino.h>
#include <ESP32Servo.h>


// Incluir cabezales de TensorFlow Lite
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Incluir modelo entrenado

#include "modelo_entrenado_por_ventanas.h"

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

Servo servoPinza;
Servo servoAbreCierra;
QueueHandle_t dataQueue;

// Constantes para el preprocesamiento de datos

const float mean_values[200] = {
    0.19919276, 0.20042254, 0.20075889, 0.20150517, 0.20246166, 0.19902459,
    0.20056969, 0.19936094, 0.19941349, 0.20052765, 0.19808911, 0.1980786,
    0.19704853, 0.19992853, 0.20104268, 0.19971831, 0.20039101, 0.19919276,
    0.20187305, 0.20255626, 0.20007568, 0.20087451, 0.20068531, 0.19974984,
    0.20136852, 0.20110575, 0.20091655, 0.20193611, 0.20197816, 0.20041203,
    0.20131597, 0.2044167,  0.20483713, 0.2034602,  0.20243013, 0.20214633,
    0.20347071, 0.2046164,  0.20243013, 0.20281903, 0.20255626, 0.20323947,
    0.20046458, 0.19967626, 0.20203071, 0.20088502, 0.1996132,  0.20051714,
    0.20051714, 0.20039101, 0.19943451, 0.20227246, 0.20323947, 0.20425903,
    0.20523655, 0.20280852, 0.20424852, 0.20266137, 0.20248268, 0.2039437,
    0.20218838, 0.20070633, 0.19988648, 0.20217787, 0.20501582, 0.20361786,
    0.20175743, 0.20164181, 0.20241962, 0.20447976, 0.20277699, 0.20402779,
    0.20357582, 0.20272443, 0.20209378, 0.2030713,  0.20223042, 0.20064327,
    0.20171538, 0.20162079, 0.20343918, 0.20464794, 0.20497378, 0.20335509,
    0.20249319, 0.20197816, 0.204711,   0.20467947, 0.20133699, 0.202304,
    0.20258779, 0.20294516, 0.20214633, 0.20104268, 0.20102166, 0.20197816,
    0.20133699, 0.20357582, 0.20260881, 0.19997057, 0.19917174, 0.20232502,
    0.20312385, 0.20453232, 0.20467947, 0.20253524, 0.20231451, 0.20116882,
    0.20182049, 0.20483713, 0.20323947, 0.20187305, 0.20226195, 0.20190458,
    0.20385962, 0.20313436, 0.2023986,  0.20096911, 0.2013475,  0.20376502,
    0.20257728, 0.2040383,  0.20335509, 0.20179947, 0.20065378, 0.20187305,
    0.20217787, 0.20195714, 0.20296619, 0.2022094,  0.20434312, 0.20500531,
    0.20443772, 0.20305027, 0.20262984, 0.20104268, 0.20372297, 0.2040383,
    0.20313436, 0.20433261, 0.20383859, 0.20270341, 0.20309232, 0.2015367,
    0.20256677, 0.20397524, 0.2021148,  0.20361786, 0.20387013, 0.20158925,
    0.20187305, 0.20417494, 0.20243013, 0.20280852, 0.20296619, 0.20206225,
    0.20320794, 0.20183101, 0.20350224, 0.20625611, 0.20426954, 0.202304,
    0.2038491,  0.20424852, 0.20513144, 0.20445874, 0.20300823, 0.20052765,
    0.20152619, 0.20339713, 0.20280852, 0.20301874, 0.20217787, 0.20170487,
    0.20114779, 0.20141057, 0.20191509, 0.20185203, 0.20224093, 0.20035947,
    0.2029767,  0.20418545, 0.20464794, 0.20237757, 0.20428005, 0.20274546,
    0.2045218,  0.204711,   0.20210429, 0.20280852, 0.20284006, 0.20082196,
    0.20314487, 0.20126341, 0.20135801, 0.20317641, 0.20196765, 0.20298721,
    0.20320794, 0.20158925
};
const float scale_values[200] = {
    0.08922078, 0.08687682, 0.08554696, 0.08585491, 0.08568256, 0.08389594,
    0.08381811, 0.08239591, 0.08302407, 0.08358287, 0.08461475, 0.08559773,
    0.08360816, 0.08621486, 0.08549438, 0.08469453, 0.08373104, 0.08413368,
    0.08358115, 0.08578095, 0.08401918, 0.08481572, 0.0831986,  0.08204302,
    0.08406293, 0.08231491, 0.08323302, 0.08321262, 0.08362715, 0.08310521,
    0.08375096, 0.08440396, 0.08451134, 0.08378129, 0.08352348, 0.08281547,
    0.08321591, 0.08352059, 0.08175035, 0.08153943, 0.08177299, 0.08195897,
    0.08289195, 0.08290978, 0.08328641, 0.08383339, 0.08390861, 0.08334438,
    0.08261507, 0.08181735, 0.08259315, 0.0848256,  0.08396957, 0.08352696,
    0.08363231, 0.08126376, 0.08185921, 0.08200503, 0.08202237, 0.08490001,
    0.08466327, 0.08571928, 0.08561727, 0.08758567, 0.08485151, 0.08492917,
    0.08376271, 0.08366025, 0.08355061, 0.08604025, 0.0840895,  0.08617537,
    0.083286,   0.08243551, 0.08230952, 0.08204295, 0.08145713, 0.08175566,
    0.08423004, 0.08451005, 0.08891459, 0.08463782, 0.08369557, 0.08283977,
    0.08287431, 0.08253204, 0.08321818, 0.08278786, 0.08132441, 0.08101814,
    0.08198216, 0.08145375, 0.08169439, 0.08228797, 0.08284786, 0.08466868,
    0.08487128, 0.08517888, 0.08298126, 0.08196922, 0.08224655, 0.08259862,
    0.08210337, 0.08290399, 0.08329577, 0.0809781,  0.08136331, 0.08043594,
    0.0807937,  0.08433316, 0.08393249, 0.08473075, 0.08525433, 0.08493446,
    0.0836276,  0.08327704, 0.08298797, 0.0823949,  0.08219323, 0.08387761,
    0.08170745, 0.0840858,  0.08168696, 0.08217104, 0.08203346, 0.08092268,
    0.0807656,  0.08199024, 0.08254186, 0.08402449, 0.08886778, 0.08505666,
    0.08508483, 0.08369009, 0.08382236, 0.08365374, 0.08410756, 0.08204573,
    0.08064257, 0.08112277, 0.08165489, 0.08136348, 0.08207632, 0.0817335,
    0.08250056, 0.08450913, 0.08450585, 0.08541376, 0.08251044, 0.08117515,
    0.08284748, 0.08533077, 0.08501186, 0.0843881,  0.08373801, 0.08118444,
    0.08194119, 0.08140575, 0.08196076, 0.08430105, 0.08418402, 0.08559894,
    0.08548768, 0.08446718, 0.08371655, 0.08402506, 0.08380956, 0.08197952,
    0.08204588, 0.08387023, 0.08282081, 0.08467767, 0.083467,   0.08269702,
    0.08405769, 0.08208214, 0.08217279, 0.08252764, 0.08355126, 0.08456359,
    0.08963766, 0.08589012, 0.08575577, 0.08344531, 0.0844445,  0.08363297,
    0.08342791, 0.08185418, 0.08142961, 0.08211727, 0.0831404,  0.0834027,
    0.08435685, 0.08362403, 0.08399268, 0.08576419, 0.08642547, 0.08680193,
    0.08386369, 0.08327357
};

const unsigned long predictionInterval = 1000;  // Intervalo de tiempo en milisegundos
unsigned long lastPredictionTime = 0;

float sensorData[200];
int dataIndex = 0;

// Inicialización y configuración del modelo

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

  

int predict(float data[200]) {
    for (int i = 0; i < 200; ++i) {
        input->data.f[i] = data[i];
    }
    if (interpreter->Invoke() == kTfLiteOk) {
        float max_value = output->data.f[0];
        int max_index = 0;
        for (int i = 1; i < output->dims->data[output->dims->size - 1]; ++i) {
            if (output->data.f[i] > max_value) {
                max_value = output->data.f[i];
                max_index = i;
            }
        }
        //Serial.println()
        return max_index;
    }
    return -1;  // Retornar -1 en caso de error
}


void readSensorData(void *pvParameters) {

    while (1) {
        if (dataIndex <200) {
            int sensorValue = analogRead(34);  // Leer el valor del sensor en el pin 34
            // Mapear el valor del sensor de 0-4095 a 0-1023
            int mapeadoValue = map(sensorValue, 0, 4095, 0, 1023);       
            //Serial.print("Valor del sensor (mapeado): "); // Imprimir el valor mapeado para depuración
            float dato = static_cast<float>(mapeadoValue);
            float dato_normalizado = dato / 1023.0;
            float dato_escalado = (dato_normalizado - mean_values[dataIndex]) / scale_values[dataIndex];
            sensorData[dataIndex++] = dato_escalado
        }
        if (dataIndex == 200){
            xQueueSend(dataQueue, &sensorData, portMAX_DELAY);
            dataIndex = 0;
        }
        delay(100);
    }
}

void receiveData(void *pvParameters) {
    while (1) {
        if (Serial.available() > 0) {
            String inputStr = Serial.readStringUntil('\n');
            float dato = inputStr.toFloat();

            float dato_normalizado = dato / 1023.0;

            // Asegúrate de que dataIndex no exceda el tamaño del array
            if (dataIndex < 200) {
                float dato_escalado = (dato_normalizado - mean_values[dataIndex]) / scale_values[dataIndex];
                sensorData[dataIndex++] = dato_escalado;

                // Verificar si se llenó el buffer y, en caso afirmativo, enviar a la cola
                if (dataIndex == 200) {
                    xQueueSend(dataQueue, &sensorData, portMAX_DELAY);
                    dataIndex = 0;  // Resetear el índice para comenzar de nuevo
                }
            }
        }
        delay(100);  // Pequeña pausa para estabilizar la lectura de datos
    }
}



void controlarServoPinza() {
    for (int pos = 0; pos <= 180; pos += 10) {
        servoPinza.write(pos);
        delay(100);
    }
    delay(1000);
    for (int pos = 180; pos >= 0; pos -= 10) {
        servoPinza.write(pos);
        delay(50);
    }
}

void controlarServoAbreCierra() {
    for (int pos = 0; pos <= 180; pos += 10) {
        servoAbreCierra.write(pos);
        delay(50);
    }
    delay(1000);
    for (int pos = 180; pos >= 0; pos -= 10) {
        servoAbreCierra.write(pos);
        delay(50);
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
        case 1: // Balance pie
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 2: // BrazoArriba
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 3: // Codo
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 4: // Descanso
            servoPinza.write(90);
            servoAbreCierra.write(90);
        case 5: // Descnaso2
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 6: // Descanso de Pie
            servoPinza.write(90);
            servoAbreCierra.write(90);
            break;
        case 7: // Pinza (Solo mover el servo en pin 12)
            controlarServoPinza();
            break;
      case 8: // pinza2
            controlarServoPinza();
            break;
        default: // No hacer nada para las demás clases
            servoPinza.write(90);      // Poner el servo en posición neutral
            servoAbreCierra.write(90); // Poner el servo en posición neutral
            break;
    }
}


void servoTask(void *pvParameters) {
    float data[200];
    while (1) {
        if (xQueueReceive(dataQueue, &data, portMAX_DELAY) == pdTRUE) {
            int classIndex = predict(data);
            controlServos(classIndex);
        }
    }
}


void setup() {
    servoPinza.attach(12);
    servoAbreCierra.attach(14);
    Serial.begin(115200);
    initModel();
    dataQueue = xQueueCreate(10, sizeof(sensorData));

    xTaskCreatePinnedToCore(readSensorData, "ReadSensorData", 2048, NULL, 1, NULL, 0);
    //xTaskCreatePinnedToCore(receiveData, "ReceiveData", 2048, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(servoTask, "ServoTask", 2048, NULL, 1, NULL, 1);    // Tarea en el núcleo 1
}

void loop() {
    // El loop principal no necesita hacer nada, ya que las tareas se están ejecutando
}