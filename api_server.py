from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ==========================
# Configuración InfluxDB
# ==========================
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "aUAFfIYK049d7k-NSyjVbcBChuZImuU4dBp4SpX89qOVflRDqPoXHXmQyNH9Tamc_5e9-Sr4jzWWqUCgCNpdnw=="
INFLUX_ORG = "inyectora"
INFLUX_BUCKET = "inyectora_data"

client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

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
    # Log por consola (lo que ya venías viendo)
    print("=== Lote recibido ===")
    print(f"Máquina: {reading.machine}")
    print(f"Molde: {reading.mold}")
    print(f"Automática: {reading.automatic}")
    print(f"Pulso de inyección: {reading.injection_pulse}")
    print(f"Cantidad de lecturas: {len(reading.flows)}")
    print(f"Primer sensor: Caudalimetro_1 = {reading.flows[0]:.2f}")
    print(f"Hora (UTC): {reading.timestamp_utc}")
    print("======================")

    # Parseo de timestamp
    try:
        ts = datetime.fromisoformat(reading.timestamp_utc)
    except Exception:
        # Si viene sin microsegundos o algo raro, cae acá
        ts = datetime.utcnow()

    # Construcción de puntos para los 24 caudalímetros
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

    # Escritura en Influx
    write_api.write(
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG,
        record=points
    )

    return {"status": "ok", "points_written": len(points)}
