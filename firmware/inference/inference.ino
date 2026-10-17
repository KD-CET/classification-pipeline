// Part 4 -- On-device inference (ESP32 + GUVA-S12SD UV + TSL2591 IR)
//
// Algorithm 1: read -> same EMA filter as collection -> features -> scale with
// SCALE_BOUNDS -> grid cell -> one pgm_read_byte per class -> argmax -> reject.
//
// Every sample prints one CSV row (lines starting with '#' are comments):
//   rawUV,rawIR,uvF,irF,label,pred,score
// raw*  = unfiltered readings     (Part 5 raw-vs-filtered plot)
// *F    = EMA-filtered readings   (what the classifier sees)
// label = what you said the lighting is (for the held-out batch), "-" if none
// pred  = CLASS_NAMES[c] or UNKNOWN, score = winning table value (0-1)
//
// Serial commands (115200 baud, newline):
//   L:sun   tag the following rows as "sun" (also resets the filter)
//   L:      stop tagging (label = "-")
//   B       benchmark: average time per lookup over 1000 lookups
//
// Flash/SRAM measurement: compile once with INCLUDE_MODEL 1 and once with
// INCLUDE_MODEL 0, and subtract the "Sketch uses" / "Global variables use"
// numbers the Arduino IDE prints. The difference is the model's real cost.

#define INCLUDE_MODEL 1

#include <Wire.h>
#include <math.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#if INCLUDE_MODEL
#include "model.h"
#endif

Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

const int UV_PIN = 34;
const int SDA_PIN = 21;
const int SCL_PIN = 22;

// MUST match the collection firmware, or the device sees a different signal
// from the one the model was trained on.
const float ALPHA = 0.2f;
const unsigned long SAMPLE_INTERVAL_MS = 100;   // ~10 Hz target; TSL integration stretches it

float uvFiltered = -1;   // -1 = not initialised
float irFiltered = -1;
String currentLabel = "-";
unsigned long lastSampleTime = 0;

volatile int sink;       // stops the compiler optimising the benchmark away

// ---------------------------------------------------------------- the model
// Returns the class index, or -1 for UNKNOWN. *score gets the winning value.
int classify(float uvF, float irF, float *score) {
#if INCLUDE_MODEL
  // features: identical to to_features() in pipeline_common.py
  float x = uvF;
  float y = log10f(1.0f + irF);

  // scale with the TRAINING bounds
  float sx = (x - SCALE_BOUNDS[0]) / (SCALE_BOUNDS[1] - SCALE_BOUNDS[0]);
  float sy = (y - SCALE_BOUNDS[2]) / (SCALE_BOUNDS[3] - SCALE_BOUNDS[2]);

  // grid cell, clipped: identical to cell_of() in pipeline_common.py
  int col = (int)(sx * GRID_N);
  int row = (int)(sy * GRID_N);
  if (col < 0) col = 0;
  if (col > GRID_N - 1) col = GRID_N - 1;
  if (row < 0) row = 0;
  if (row > GRID_N - 1) row = GRID_N - 1;

  // one byte per class, argmax
  int best = 0;
  float bestV = -1.0f;
  for (int c = 0; c < NUM_CLASSES; c++) {
    float v = pgm_read_byte(&CLASS_TABLE[c][row][col]) / 255.0f;
    if (v > bestV) { bestV = v; best = c; }
  }
  *score = bestV;
  return (bestV < REJECT_TAU) ? -1 : best;
#else
  *score = 0;
  return -1;
#endif
}

const char *className(int c) {
#if INCLUDE_MODEL
  return (c < 0) ? "UNKNOWN" : CLASS_NAMES[c];
#else
  return "UNKNOWN";
#endif
}

// ---------------------------------------------------------------- sensors
void configureTSL2591() {
  // Same settings as collection: MED gain, 100 ms integration.
  tsl.setGain(TSL2591_GAIN_MED);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);
}

void readSensors(int &rawUV, int &rawIR) {
  rawUV = analogRead(UV_PIN);
  uint32_t lum = tsl.getFullLuminosity();   // ir << 16 | full
  rawIR = lum >> 16;
  if ((lum & 0xFFFF) == 0xFFFF || rawIR == 0xFFFF) {
    Serial.println("# WARNING: TSL2591 saturated");
  }
}

void resetFilter() { uvFiltered = irFiltered = -1; }

// ---------------------------------------------------------------- benchmark
void runBenchmark() {
  const int N = 1000;
  float score;
  float uv = (uvFiltered < 0) ? 0 : uvFiltered;
  float ir = (irFiltered < 0) ? 200 : irFiltered;

  unsigned long t0 = micros();
  for (int i = 0; i < N; i++) {
    sink = classify(uv + (i & 1) * 0.001f, ir, &score);   // vary input slightly
  }
  unsigned long lookupUs = micros() - t0;

  // full sample for comparison: ADC + I2C read + filter + lookup
  const int M = 20;
  int rUV, rIR;
  t0 = micros();
  for (int i = 0; i < M; i++) {
    readSensors(rUV, rIR);
    sink = classify(rUV, rIR, &score);
  }
  unsigned long fullUs = micros() - t0;

  Serial.printf("# BENCH lookup: %.2f us/query (%d queries, total %lu us)\n",
                lookupUs / (float)N, N, lookupUs);
  Serial.printf("# BENCH full sample (ADC + TSL2591 + lookup): %.1f ms\n",
                fullUs / (float)M / 1000.0f);
  Serial.printf("# free heap: %u B\n", ESP.getFreeHeap());
}

// ---------------------------------------------------------------- commands
void handleSerialCommands() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.startsWith("L:")) {
    String lab = line.substring(2);
    lab.trim();
    currentLabel = lab.length() ? lab : "-";
    resetFilter();   // the EMA must not carry the previous condition into this one
    Serial.print("# label = ");
    Serial.println(currentLabel);
  } else if (line == "B") {
    runBenchmark();
  }
}

// ---------------------------------------------------------------- main
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(UV_PIN, ADC_11db);   // full 0-3.3 V range
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!tsl.begin()) {
    Serial.println("# ERROR: TSL2591 not found (SDA=21, SCL=22)");
    while (1) delay(1000);
  }
  configureTSL2591();
#if INCLUDE_MODEL
  Serial.printf("# model: %d classes, %dx%d grid, table %u B, tau %.2f\n",
                NUM_CLASSES, GRID_N, GRID_N, (unsigned)sizeof(CLASS_TABLE), REJECT_TAU);
#endif
  Serial.printf("# free heap at boot: %u B\n", ESP.getFreeHeap());
  Serial.println("rawUV,rawIR,uvF,irF,label,pred,score");
}

void loop() {
  handleSerialCommands();

  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  int rawUV, rawIR;
  readSensors(rawUV, rawIR);

  // EMA filter: y_n = alpha * x_n + (1 - alpha) * y_(n-1)
  if (uvFiltered < 0) {
    uvFiltered = rawUV;
    irFiltered = rawIR;
  } else {
    uvFiltered = ALPHA * rawUV + (1 - ALPHA) * uvFiltered;
    irFiltered = ALPHA * rawIR + (1 - ALPHA) * irFiltered;
  }

  float score;
  int c = classify(uvFiltered, irFiltered, &score);

  Serial.printf("%d,%d,%.2f,%.2f,%s,%s,%.3f\n", rawUV, rawIR, uvFiltered, irFiltered,
                currentLabel.c_str(), className(c), score);
}
