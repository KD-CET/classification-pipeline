// Part 1 -- training-data logger (ESP32 + GUVA-S12SD UV + TSL2591 IR)
// Prints filtered readings as CSV rows "uv,ir,label" while a label is set.
// Serial commands: L:<label> to start logging that class, L: (empty) to pause.
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>

Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

const int UV_PIN = 34;

const int SDA_PIN = 21;
const int SCL_PIN = 22;

const float ALPHA = 0.2f;                     // EMA smoothing factor (0-1): lower = smoother/slower
const unsigned long SAMPLE_INTERVAL_MS = 100; // ~10 Hz target (TSL2591 integration time may stretch this)

float uvFiltered = -1;  // -1 = "not yet initialized"
float irFiltered = -1;

String currentLabel = "";   // empty = not logging

unsigned long rowCountForLabel = 0;
unsigned long lastSampleTime = 0;

void configureTSL2591() {
  // Start conservative (MED gain, 100ms integration) to avoid the saturation.
  // Raise TSL2591_GAIN_HIGH or integration time later if readings sit too low/flat.
  // Lower them if 0xFFFF (saturated).
  tsl.setGain(TSL2591_GAIN_MED);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);
}

void handleSerialCommands();

void setup() {
  Serial.begin(115200);
  analogReadResolution(12); // ESP32 ADC: 0-4095

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!tsl.begin()) {
    Serial.println("# ERROR: TSL2591 not found. Check wiring (SDA=21, SCL=22) and power.");
    while (1) { delay(1000); }
  }
  configureTSL2591();

  Serial.println("# Ready. Send L:<label> to start logging that class, L: (empty) to pause.");
  Serial.println("# Example: L:sun   then   L:   to stop.");
}

void loop() {
  handleSerialCommands();

  unsigned long now = millis();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;

    int rawUV = analogRead(UV_PIN);

    uint32_t lum = tsl.getFullLuminosity(); // ir<<16 | full
    uint16_t full = lum & 0xFFFF;
    uint16_t ir = lum >> 16;

    if (full == 0xFFFF || ir == 0xFFFF) {
      Serial.println("# WARNING: TSL2591 saturated — lower gain/integration time.");
    }

    int rawIR = ir; // use the IR-only channel as sensor2

    // Apply EMA filter: y_n = alpha*x_n + (1-alpha)*y_(n-1)
    if (uvFiltered < 0) {
      uvFiltered = rawUV;   // initialize on first sample
      irFiltered = rawIR;
    } else {
      uvFiltered = ALPHA * rawUV + (1 - ALPHA) * uvFiltered;
      irFiltered = ALPHA * rawIR + (1 - ALPHA) * irFiltered;
    }

    if (currentLabel.length() > 0) {
      // CSV row: sensor1,sensor2,label
      Serial.print(uvFiltered, 2);
      Serial.print(",");
      Serial.print(irFiltered, 2);
      Serial.print(",");
      Serial.println(currentLabel);

      rowCountForLabel++;
      if (rowCountForLabel % 50 == 0) {
        Serial.print("# ");
        Serial.print(currentLabel);
        Serial.print(" rows so far: ");
        Serial.println(rowCountForLabel);
      }
    }
  }
}

void handleSerialCommands() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("L:")) {
      String newLabel = line.substring(2);
      newLabel.trim();

      if (newLabel != currentLabel) {
        currentLabel = newLabel;
        rowCountForLabel = 0;
        if (currentLabel.length() > 0) {
          Serial.print("# Logging started for label: ");
          Serial.println(currentLabel);
        } else {
          Serial.println("# Logging paused.");
        }
      }
    }
  }
}
