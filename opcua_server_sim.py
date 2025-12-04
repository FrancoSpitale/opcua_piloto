from opcua import ua, Server
import time
import random

print("Inicializando servidor OPC UA...")

# Crear servidor
server = Server()

# Endpoint
ENDPOINT = "opc.tcp://0.0.0.0:4840/inyectora/opcua/"
server.set_endpoint(ENDPOINT)
print(f"Servidor OPC UA escuchando en: {ENDPOINT}")

# Namespace
uri = "http://spitale.com/inyectora/opcua/"
idx = server.register_namespace(uri)
print(f"Namespace registrado: {uri} (idx = {idx})")

# Obtener nodo Objects
objects = server.get_objects_node()

# Crear objeto principal Inyectora1
inyectora = objects.add_object(idx, "Inyectora1")
print("Objeto Inyectora1 creado.")

# Variables principales
molde_actual = inyectora.add_variable(idx, "MoldeActual", 1)
pulso_inyeccion = inyectora.add_variable(idx, "PulsoInyeccion", False)
maquina_automatica = inyectora.add_variable(idx, "MaquinaAutomatica", True)

# Hacer variables modificables
molde_actual.set_writable()
pulso_inyeccion.set_writable()
maquina_automatica.set_writable()

# Crear 24 caudalímetros
caudalimetros = []
for i in range(1, 25):
    var = inyectora.add_variable(idx, f"Caudalimetro_{i}", 0.0)
    var.set_writable()
    caudalimetros.append(var)

print("Caudalímetros creados correctamente.")

# Iniciar servidor
server.start()
print("Servidor OPC UA iniciado correctamente.")
print("Esperando clientes...\n")

try:
    ciclo = 0
    while True:
        ciclo += 1

        # Simulaciones básicas
        molde_val = random.randint(1, 5)
        pulso_val = random.choice([True, False])
        auto_val = True

        molde_actual.set_value(molde_val)
        pulso_inyeccion.set_value(pulso_val)
        maquina_automatica.set_value(auto_val)

        # Actualizar 24 caudalímetros
        flow_values = []
        for i, cm in enumerate(caudalimetros, start=1):
            v = random.uniform(10.0, 40.0)
            cm.set_value(v)
            flow_values.append(v)

        # Print diagnóstico cada ciclo
        print(f"[Ciclo {ciclo}] Molde={molde_val}, Pulso={pulso_val}")
        print(f"  Caudalimetro_1={flow_values[0]:.2f}  Caudalimetro_24={flow_values[-1]:.2f}")
        print("  (Servidor activo, esperando lecturas del cliente...)")

        time.sleep(1)

except Exception as e:
    print(f"ERROR en servidor OPC UA: {e}")

finally:
    print("Cerrando servidor OPC UA...")
    server.stop()
    print("Servidor detenido.")
