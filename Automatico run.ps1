# ======================
# run.ps1 – Startup total del sistema
# ======================

Write-Host "Activando entorno..."
& C:/Users/spita/opcua_piloto/.venv/Scripts/Activate.ps1
cd C:/Users/spita/opcua_piloto

Write-Host "Iniciando InfluxDB (Docker)..."
docker start influxdb

Start-Sleep -Seconds 2

Write-Host "Matando Streamlit viejo..."
$pid8501 = (netstat -ano | findstr 8501 | Select-String -Pattern "\d+$").Matches.Value
if ($pid8501) {
    taskkill /PID $pid8501 /F
}

Write-Host "Arrancando FastAPI..."
Start-Process powershell "-NoExit -Command `"& C:/Users/spita/opcua_piloto/.venv/Scripts/Activate.ps1; uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload`""

Start-Sleep -Seconds 2

Write-Host "Arrancando OPC UA Server..."
Start-Process powershell "-NoExit -Command `"& C:/Users/spita/opcua_piloto/.venv/Scripts/Activate.ps1; python opcua_server_sim.py`""

Start-Sleep -Seconds 2

Write-Host "Arrancando OPC UA Agent..."
Start-Process powershell "-NoExit -Command `"& C:/Users/spita/opcua_piloto/.venv/Scripts/Activate.ps1; python opcua_client_agent.py`""

Start-Sleep -Seconds 2

Write-Host "Arrancando Dashboard Streamlit..."
Start-Process powershell "-NoExit -Command `"& C:/Users/spita/opcua_piloto/.venv/Scripts/Activate.ps1; streamlit run app.py --server.port 8501`""

Start-Sleep -Seconds 3

Write-Host "Arrancando Ngrok..."
Start-Process powershell "-NoExit -Command `"ngrok http 8501`""

Write-Host "Sistema completo iniciado."
