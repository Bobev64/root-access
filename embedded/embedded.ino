#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_SHT31.h>

constexpr uint8_t ADC_PIN = 1;
constexpr uint8_t SDA_PIN = 4;
constexpr uint8_t SCL_PIN = 5;
constexpr uint8_t SHT_ADDRESS = 0x44; // Change to 0x45 if needed

const char* AP_SSID = "ESP32-Sensors";
const char* AP_PASSWORD = "sensor123";

constexpr uint32_t SAMPLE_INTERVAL_MS = 1000;
constexpr uint32_t REQUEST_TIMEOUT_MS = 3000;

WiFiServer server(80);
WiFiClient client;
Adafruit_SHT31 sht;

bool shtAvailable = false;
bool streaming = false;

String requestHeaders;
uint32_t connectionStarted = 0;
uint32_t lastSample = 0;

void closeClient() {
  client.stop();
  streaming = false;
  requestHeaders = "";
}

void sendSample() {
  uint32_t timestamp = millis();
  int adcRaw = analogRead(ADC_PIN);
  uint32_t adcMillivolts = analogReadMilliVolts(ADC_PIN);

  float temperature = NAN;
  float humidity = NAN;

  if (shtAvailable) {
    temperature = sht.readTemperature();
    humidity = sht.readHumidity();
  }

  String row;
  row.reserve(100);

  row += String(timestamp);
  row += ",";
  row += String(adcRaw);
  row += ",";
  row += String(adcMillivolts);
  row += ",";

  // Leave failed sensor measurements blank in the CSV.
  if (!isnan(temperature)) {
    row += String(temperature, 2);
  }

  row += ",";

  if (!isnan(humidity)) {
    row += String(humidity, 2);
  }

  row += "\n";

  Serial.print(row);

  if (streaming && client.connected()) {
    if (client.print(row) != row.length()) {
      closeClient();
    }
  }
}

void handleClient() {
  if (!client.connected()) {
    closeClient();

    client = server.accept();
    if (!client) {
      return;
    }

    client.setNoDelay(true);
    connectionStarted = millis();
  }

  // After the HTTP request has been accepted, keep the
  // connection open and let sendSample() write CSV rows.
  if (streaming) {
    return;
  }

  // Read the HTTP headers without waiting for more bytes.
  while (client.available()) {
    requestHeaders += static_cast<char>(client.read());

    if (requestHeaders.length() > 2048) {
      closeClient();
      return;
    }

    if (requestHeaders.endsWith("\r\n\r\n")) {
      if (requestHeaders.startsWith("GET /data.csv ")) {
        // HTTP/1.0 allows the response body to continue until
        // the connection closes, without a Content-Length.
        client.print(
          "HTTP/1.0 200 OK\r\n"
          "Content-Type: text/csv; charset=utf-8\r\n"
          "Cache-Control: no-store\r\n"
          "Connection: close\r\n"
          "\r\n"
          "uptime_ms,adc_raw,adc_mV,temperature_C,humidity_percent\n"
        );

        streaming = true;
        requestHeaders = "";
      } else {
        client.print(
          "HTTP/1.0 404 Not Found\r\n"
          "Content-Type: text/plain\r\n"
          "Connection: close\r\n"
          "\r\n"
          "Use /data.csv for the live CSV stream.\n"
        );
        closeClient();
      }

      return;
    }
  }

  if (millis() - connectionStarted >= REQUEST_TIMEOUT_MS) {
    closeClient();
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(ADC_PIN, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  shtAvailable = sht.begin(SHT_ADDRESS);
  Serial.println(
    shtAvailable ? "SHT3x initialized."
                 : "SHT3x not found; temperature/humidity will be blank."
  );

  WiFi.mode(WIFI_AP);

  // Give the AP a fixed address.
  IPAddress apIP(192, 168, 4, 1);
  IPAddress subnet(255, 255, 255, 0);

  if (!WiFi.softAPConfig(apIP, apIP, subnet) ||
      !WiFi.softAP(AP_SSID, AP_PASSWORD)) {
    Serial.println("Failed to start Wi-Fi access point.");
    while (true) {
      delay(1000);
    }
  }

  server.begin();

  Serial.print("Connect to Wi-Fi: ");
  Serial.println(AP_SSID);
  Serial.print("CSV stream: http://");
  Serial.print(WiFi.softAPIP());
  Serial.println("/data.csv");

  lastSample = millis();
}

void loop() {
  handleClient();

  uint32_t now = millis();

  if (now - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = now;
    sendSample();
  }

  delay(1);
}