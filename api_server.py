# ==========================
# api_server.py
# ==========================

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import os

# ==========================
# Configuración InfluxDB
# ==========================
INFLUX_URL   = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "aUAFfIYK049d7k-NSyjVbcBChuZImuU4dBp4SpX89qOVflRDqPoXHXmQyNH9Tamc_5e9-Sr4jzWWqUCgCNpdnw==")
INFLUX_ORG   = os.getenv("INFLUX_ORG", "inyectora")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "inyectora_data")

def get_write_api():
    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )
    return client, client.write_api(write_options=SYNCHRONOUS)

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
        print(f"Primer sensor: Caudalimetro_1 = {reading.flows[0]:.2f}")
    print(f"Hora (UTC): {reading.timestamp_utc}")
    print("======================")

    # Parseo de timestamp
    try:
        ts = datetime.fromisoformat(reading.timestamp_utc)
    except Exception:
        ts = datetime.utcnow()

    # Construcción de puntos
    points = []
    for i, value in enumerate(reading.flows, start=1):
        p = (
            Point("caudalimetros")
            .tag("machine", reading.machine)
            .tag("mold", str(reading.mold))
            .tag("sensor", f"Caudalimetro_{i}")
            .field("value", float(value))
            .field("automatic", bool(reading.automatic))
            .field("injection_pulse", bool(reading.injection_pulse))
            .time(ts)
        )
        points.append(p)

    # Escritura en Influx (robusta): crear cliente/write_api por request + retries cortos
    last_err = None
    for attempt in range(1, 4):  # 3 intentos
        try:
            client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            write_api = client.write_api(write_options=SYNCHRONOUS)

            write_api.write(
                bucket=INFLUX_BUCKET,
                org=INFLUX_ORG,
                record=points
            )

            # limpieza explícita para evitar hilos/estado zombie
            try:
                write_api.close()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass

            return {"status": "ok", "points_written": len(points), "attempt": attempt}

        except Exception as e:
            last_err = e
            # backoff simple
            time.sleep(0.25 * attempt)

    # Si falló todo: devolver 503 (y no 500 genérico)
    raise HTTPException(
        status_code=503,
        detail={
            "error": "InfluxDB write failed",
            "influx_url": INFLUX_URL,
            "bucket": INFLUX_BUCKET,
            "org": INFLUX_ORG,
            "attempts": 3,
            "exception": f"{type(last_err).__name__}: {last_err}",
        },
    )