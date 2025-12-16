from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ==========================
# Configuración InfluxDB
# ==========================

INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG", "inyectora")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "inyectora_data")

if not INFLUX_TOKEN:
    raise RuntimeError("INFLUX_TOKEN no definido en variables de entorno")

client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG,
)

query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)

# ==========================
# API FastAPI
# ==========================
app = FastAPI()

class Reading(BaseModel):
    machine: str
    mold: int
    automatic: bool
    injection_pulse: bool
    timestamp_utc: str
    flows: list[float]

@app.get("/")
def root():
    return {"status": "ok", "message": "API inyectora OPC UA -> InfluxDB"}

@app.post("/readings")
def receive_readings(reading: Reading):
    # Log por consola
    print("=== Lote recibido ===")
    print(f"Máquina: {reading.machine}")
    print(f"Molde: {reading.mold}")
    print(f"Automática: {reading.automatic}")
    print(f"Pulso de inyección: {reading.injection_pulse}")
    print(f"Cantidad de lecturas: {len(reading.flows)}")
    if reading.flows:
        print(f"Primer sensor (Caudalimetro_1): {reading.flows[0]:.2f}")
    print(f"Hora (UTC): {reading.timestamp_utc}")
    print("======================")

    # Parseo de timestamp
    try:
        ts = datetime.fromisoformat(reading.timestamp_utc)
    except Exception:
        ts = datetime.utcnow()

    # Construcción de puntos para InfluxDB
    points = []
    for i, value in enumerate(reading.flows, start=1):
        p = (
            Point("caudalimetros")
            .tag("machine", reading.machine)
            .tag("mold", str(reading.mold))
            .tag("sensor", f"Caudalimetro_{i}")
            .field("value", float(value))
            .field("automatic", reading.automatic)
            .field("injection_pulse", reading.injection_pulse)
            .time(ts)
        )
        points.append(p)

    write_api.write(
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG,
        record=points
    )

    return {"status": "ok", "points_written": len(points)}
