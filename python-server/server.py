import threading
import time
import os
from datetime import datetime, timezone

import serial
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

app = FastAPI(title="GreenVision Live Sensor Server")

# ESP32 serial connection
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

# Store recent sensor readings
data_lock = threading.Lock()
sensor_rows = []

CSV_HEADER = [
    "timestamp",
    "uptime_ms",
    "adc_raw",
    "adc_mV",
    "soil_moisture_percent",
    "temperature_C",
    "humidity_percent"
]


def read_esp32():
    """Continuously read CSV lines from the ESP32."""

    global sensor_rows

    # Store the most recent moisture readings for smoothing
    moisture_history = []

    try:
        ser = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1
        )

        print(f"Connected to ESP32 on {SERIAL_PORT}")

        time.sleep(2)

        while True:
            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            print("ESP32:", line)

            parts = line.split(",")

            if len(parts) != 5:
                continue

            try:
                uptime_ms = int(parts[0])
                adc_raw = int(parts[1])
                adc_mV = int(parts[2])

                temperature = parts[3]
                humidity = parts[4]

                # Convert voltage to a basic 0–100% value
                raw_moisture = (adc_mV / 3000) * 100

                # Keep value between 0 and 100
                raw_moisture = max(0, min(100, raw_moisture))

                # Add newest reading to history
                moisture_history.append(raw_moisture)

                # Keep only the latest 5 readings
                moisture_history = moisture_history[-5:]

                # Average the latest 5 readings
                soil_moisture_percent = (
                    sum(moisture_history) / len(moisture_history)
                )

                timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                row = {
                    "timestamp": timestamp,
                    "uptime_ms": uptime_ms,
                    "adc_raw": adc_raw,
                    "adc_mV": adc_mV,
                    "soil_moisture_percent": round(
                        soil_moisture_percent, 1
                    ),
                    "temperature_C": temperature,
                    "humidity_percent": humidity
                }

                with data_lock:
                    sensor_rows.append(row)

                    # Keep the most recent 500 readings
                    sensor_rows = sensor_rows[-500:]

            except ValueError:
                continue

    except Exception as e:
        print(f"ESP32 serial error: {e}")


@app.get("/")
def home():
    return {
        "project": "GreenVision",
        "status": "online",
        "esp32_port": SERIAL_PORT,
        "endpoints": [
            "/data.csv",
            "/ai.csv",
            "/api/data",
            "/api/latest"
        ]
    }


@app.get("/data.csv", response_class=PlainTextResponse)
def csv_endpoint():

    with data_lock:
        rows = list(sensor_rows)

    output = [
        "timestamp,uptime_ms,adc_raw,adc_mV,soil_moisture_percent,temperature_C,humidity_percent"
    ]

    for row in rows:
        output.append(
            f"{row['timestamp']},"
            f"{row['uptime_ms']},"
            f"{row['adc_raw']},"
            f"{row['adc_mV']},"
            f"{row['soil_moisture_percent']},"
            f"{row['temperature_C']},"
            f"{row['humidity_percent']}"
        )

    return "\n".join(output)


@app.get("/ai.csv", response_class=PlainTextResponse)
def ai_csv_endpoint():
    """Serve the prerecorded AI demo CSV."""

    file_path = os.path.join(
        os.path.dirname(__file__),
        "greenvision_ai_demo_condensed.csv"
    )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except FileNotFoundError:
        return "error,message\n1,AI CSV file not found"


@app.get("/api/data")
def api_data():

    with data_lock:
        rows = list(sensor_rows)

    return JSONResponse(rows)


@app.get("/api/latest")
def latest():

    with data_lock:
        if sensor_rows:
            return sensor_rows[-1]

    return {"error": "No sensor data yet"}


if __name__ == "__main__":

    serial_thread = threading.Thread(
        target=read_esp32,
        daemon=True
    )

    serial_thread.start()

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )