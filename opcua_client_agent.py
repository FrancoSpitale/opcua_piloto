from opcua import Client
import time
import requests
from datetime import datetime, timezone

print("Iniciando cliente OPC UA...")

# Endpoint OPC UA del servidor simulado
url = "opc.tcp://localhost:4840/inyectora/opcua/"
print(f"URL OPC UA objetivo: {url}")

client = Client(url)

# Endpoint de la API FastAPI
API_URL = "http://127.0.0.1:8000/readings"

try:
    print("Conectando al servidor OPC UA...")
    client.connect()
    print("Conectado al servidor OPC UA")

    # Namespace
    uri = "http://spitale.com/inyectora/opcua/"
    print(f"Buscando namespace index para: {uri}")
    idx = client.get_namespace_index(uri)
    print(f"Namespace index: {idx}")

    # Nodo Objects
    objects = client.get_objects_node()
    print("Nodo Objects obtenido")

    # Nodo Inyectora1
    inyectora = objects.get_child([f"{idx}:Inyectora1"])
    print("Nodo Inyectora1 obtenido")

    # Variables principales
    molde_actual = inyectora.get_child([f"{idx}:MoldeActual"])
    pulso_inyeccion = inyectora.get_child([f"{idx}:PulsoInyeccion"])
    maquina_automatica = inyectora.get_child([f"{idx}:MaquinaAutomatica"])

    # Lista de nodos de los 24 caudalímetros
    caudalimetros = []
    for i in range(1, 25):
        var = inyectora.get_child([f"{idx}:Caudalimetro_{i}"])
        caudalimetros.append((f"Caudalimetro_{i}", var))

    print("Variables OPC UA inicializadas, entrando en bucle de lectura...")

    ciclo = 0
    while True:
        ciclo += 1

        molde = molde_actual.get_value()
        pulso = pulso_inyeccion.get_value()
        auto = maquina_automatica.get_value()

        # Leer los 24 caudales como lista de floats
        flows = []
        for name, node in caudalimetros:
            val = node.get_value()
            flows.append(float(val))

        # Timestamp en UTC ISO 8601
        ts_utc = datetime.now(timezone.utc).isoformat()

        # Payload ALINEADO con el modelo Reading de FastAPI
        payload = {
            "machine": "INY_01",             # str
            "mold": int(molde),             # int
            "automatic": bool(auto),        # bool
            "injection_pulse": bool(pulso), # bool
            "timestamp_utc": ts_utc,        # str (ISO 8601)
            "flows": flows                  # list[float]
        }

        # Enviar a la API
        try:
            resp = requests.post(API_URL, json=payload, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[ciclo {ciclo}] Enviado a API: {data}")
            else:
                print(f"[ciclo {ciclo}] Error API: status {resp.status_code}, body={resp.text}")
        except Exception as e:
            print(f"[ciclo {ciclo}] Error enviando a API: {repr(e)}")

        time.sleep(2.0)

except Exception as e:
    print("ERROR en cliente OPC UA:")
    print(repr(e))

finally:
    try:
        client.disconnect()
        print("Cliente desconectado.")
    except Exception:
        pass
