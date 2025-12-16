# ==========================
# appy.py
# ==========================

import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from textwrap import wrap
import os

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
# Configuración API / Alertas
# ==========================
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://api_server:8000")

# ==========================
# Telegram
# ==========================
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================
# Funciones auxiliares - Datos de proceso
# ==========================

def get_values_df(range_str: str = "-30m", machine: str = "INY_01") -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas: time, sensor, value
    """
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {range_str})
  |> filter(fn: (r) => r._measurement == "caudalimetros")
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => r.machine == "{machine}")
'''
    tables = query_api.query_data_frame(flux)
    if not isinstance(tables, list):
        tables = [tables]
    if len(tables) == 0:
        return pd.DataFrame()

    df = pd.concat(tables)
    df = df[["_time", "sensor", "_value"]].rename(
        columns={"_time": "time", "_value": "value"}
    )
    df = df.sort_values("time")
    return df


def get_last_bool(field: str, machine: str = "INY_01"):
    """
    Devuelve el último valor booleano del campo automatic o injection_pulse
    """
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "caudalimetros")
  |> filter(fn: (r) => r._field == "{field}")
  |> filter(fn: (r) => r.machine == "{machine}")
  |> last()
'''
    tables = query_api.query_data_frame(flux)
    if isinstance(tables, list):
        if len(tables) == 0:
            return None
        df = pd.concat(tables)
    else:
        df = tables

    if df.empty:
        return None

    return bool(df["_value"].iloc[0])


def get_mold(machine: str = "INY_01"):
    """
    Devuelve el molde actual usando el tag 'mold'
    desde el último registro de value.
    """
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "caudalimetros")
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => r.machine == "{machine}")
  |> last()
'''
    tables = query_api.query_data_frame(flux)
    if isinstance(tables, list):
        if len(tables) == 0:
            return None
        df = pd.concat(tables)
    else:
        df = tables

    if df.empty or "mold" not in df.columns:
        return None

    return df["mold"].iloc[0]


def get_global_stats(range_str: str = "-30m", machine: str = "INY_01"):
    """
    Devuelve (mean, stddev) calculados desde Influx.
    Consultas separadas mean/std para evitar conflictos de schema.
    """
    # Mean
    flux_mean = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {range_str})
  |> filter(fn: (r) => r._measurement == "caudalimetros")
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => r.machine == "{machine}")
  |> mean()
'''
    tables_mean = query_api.query_data_frame(flux_mean)
    if isinstance(tables_mean, list):
        tables_mean = pd.concat(tables_mean) if tables_mean else pd.DataFrame()
    mean_val = float(tables_mean["_value"].iloc[0]) if not tables_mean.empty else None

    # Stddev
    flux_std = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {range_str})
  |> filter(fn: (r) => r._measurement == "caudalimetros")
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => r.machine == "{machine}")
  |> stddev()
'''
    tables_std = query_api.query_data_frame(flux_std)
    if isinstance(tables_std, list):
        tables_std = pd.concat(tables_std) if tables_std else pd.DataFrame()
    std_val = float(tables_std["_value"].iloc[0]) if not tables_std.empty else None

    return mean_val, std_val


def build_anomaly_table(
    df_values: pd.DataFrame,
    mean_global: float | None,
    std_global: float | None,
    k_attention: float = 1.0,
    k_alarm: float = 2.0,
) -> pd.DataFrame:
    """
    Calcula mean por sensor y marca estado según desviación respecto al promedio global.

    - |diff| < k_attention * STD      -> OK
    - k_attention * STD ≤ |diff| < k_alarm * STD -> Atención
    - |diff| ≥ k_alarm * STD          -> Alarma
    """
    if (
        df_values.empty
        or mean_global is None
        or std_global is None
        or std_global == 0
        or k_attention <= 0
        or k_alarm <= 0
        or k_alarm <= k_attention
    ):
        return pd.DataFrame()

    df_sensor = (
        df_values.groupby("sensor")["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "mean_sensor"})
    )
    df_sensor["diff"] = df_sensor["mean_sensor"] - mean_global
    df_sensor["abs_diff"] = df_sensor["diff"].abs()

    estado = []
    for d in df_sensor["abs_diff"]:
        if d < std_global * k_attention:
            estado.append("OK")
        elif d < std_global * k_alarm:
            estado.append("Atención")
        else:
            estado.append("Alarma")

    df_sensor["estado"] = estado
    df_sensor = df_sensor.drop(columns=["abs_diff"])
    return df_sensor.sort_values("estado", ascending=False)


def check_api_health():
    """
    Verifica si la API FastAPI responde correctamente en "/"
    """
    try:
        resp = requests.get(f"{FASTAPI_BASE_URL}/", timeout=1.5)
        if resp.status_code == 200:
            return True, "API OK"
        return False, f"API error {resp.status_code}"
    except Exception as e:
        return False, f"API sin respuesta: {e.__class__.__name__}"


def check_influx_health():
    """
    Verifica si InfluxDB responde al ping
    """
    try:
        pong = client.ping()
        if pong:
            return True, "InfluxDB OK"
        return False, "InfluxDB sin respuesta"
    except Exception as e:
        return False, f"InfluxDB error: {e.__class__.__name__}"


def check_data_freshness(last_time: pd.Timestamp | None, max_age_seconds: int = 30):
    """
    Verifica la "frescura" de los datos según la edad del último punto.
    """
    if last_time is None:
        return False, None

    age_sec = (pd.Timestamp.utcnow() - last_time).total_seconds()
    return age_sec <= max_age_seconds, age_sec


# ==========================
# Funciones auxiliares - Alertas / Influx "alerts"
# ==========================

def send_telegram_alert(text: str) -> bool:
    """
    Envía una alerta vía Telegram si está configurado.
    Devuelve True si se envió, False si no.
    """
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def write_alert_event(
    kind: str,
    text: str,
    machine: str,
    severity: str = "info",
    sensors: list[str] | None = None,
    sent_telegram: bool = False,
    extra_fields: dict | None = None,
):
    """
    Guarda una alerta/comentario como evento en InfluxDB (measurement = alerts).

    extra_fields permite guardar campos adicionales:
      p.ej. {"observed_value": 31.2, "mean_sensor": 25.1, "diff_sensor": 6.1}
    """
    try:
        sensors_str = ",".join(sorted(sensors)) if sensors else ""
        p = (
            Point("alerts")
            .tag("machine", machine)
            .tag("kind", kind)
            .tag("severity", severity)
            .field("text", text)
            .field("sent_telegram", bool(sent_telegram))
            .field("sensors", sensors_str)
        )

        if extra_fields:
            for k, v in extra_fields.items():
                p = p.field(k, v)

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
    except Exception:
        pass


from datetime import datetime

def get_alert_history(
    range_str: str | None = "-24h",
    machine: str | None = None,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Devuelve historial de alertas desde Influx.

    - Si se pasa start_dt y end_dt, se usa ese rango absoluto.
    - En caso contrario, se usa range_str (ej: "-24h").
    """
    if start_dt is not None and end_dt is not None:
        start_iso = start_dt.isoformat() + "Z"
        end_iso = end_dt.isoformat() + "Z"
        flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: time(v: "{start_iso}"), stop: time(v: "{end_iso}"))
  |> filter(fn: (r) => r._measurement == "alerts")
'''
    else:
        flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {range_str})
  |> filter(fn: (r) => r._measurement == "alerts")
'''

    if machine:
        flux += f'''  |> filter(fn: (r) => r.machine == "{machine}")\n'''

    flux += '''
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
'''

    tables = query_api.query_data_frame(flux)
    if isinstance(tables, list):
        if not tables:
            return pd.DataFrame()
        df = pd.concat(tables)
    else:
        df = tables

    if df.empty:
        return pd.DataFrame()

    cols = [
        "_time", "machine", "kind", "severity", "text", "sent_telegram", "sensors",
        "mean_sensor", "diff_sensor", "observed_value", "diff_observed"
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].rename(columns={"_time": "time"})
    df = df.sort_values("time", ascending=False)
    return df

def create_excel_report(
    machine: str,
    report_title: str,
    report_subtitle: str,
    time_option: str,
    range_str: str,
    k_attention: float,
    k_alarm: float,
    sensores_sel: list[str],
    df_values: pd.DataFrame,
    df_last: pd.DataFrame,
    tabla_anomalias: pd.DataFrame | None,
    hist_df: pd.DataFrame | None,
    include_anomaly_table: bool = True,
    include_history: bool = True,
) -> bytes:
    """
    Genera un Excel en memoria con varias hojas:
      - Resumen
      - Datos_proceso
      - Ultimos_valores
      - Anomalias (opcional)
      - Historial_alertas (opcional)
    """

    # ---- FUNCIONES PARA REMOVER TIMEZONE ----
    def strip_tz_all(df):
        if df is None or df.empty:
            return df
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # primero intentar quitar tz si está localizado
                try:
                    df[col] = df[col].dt.tz_localize(None)
                    continue
                except:
                    pass
                # si estaba convertido
                try:
                    df[col] = df[col].dt.tz_convert(None)
                except:
                    pass
        return df

    # ---- Forzar todos los DF a ser timezone-unaware ----
    df_values = strip_tz_all(df_values)
    df_last = strip_tz_all(df_last)
    tabla_anomalias = strip_tz_all(tabla_anomalias)
    hist_df = strip_tz_all(hist_df)

    # ---- Comienza generación del Excel ----
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

        # ---------------- Resumen ----------------
        resumen_rows = [
            ["Título", report_title.strip() or "Reporte de Inyectora OPC UA"],
            ["Subtítulo", report_subtitle.strip()],
            ["Machine", machine],
            ["Rango tiempo proceso", f"{time_option} ({range_str})"],
            ["Sensores incluidos", ", ".join(sensores_sel) if sensores_sel else "Todos"],
            ["Umbral Atención", f"{k_attention:.2f} × STD"],
            ["Umbral Alarma", f"{k_alarm:.2f} × STD"],
            ["Cantidad de puntos", len(df_values)],
            ["Sensores activos en datos", df_values["sensor"].nunique() if not df_values.empty else 0],
        ]
        df_resumen = pd.DataFrame(resumen_rows, columns=["Item", "Valor"])
        df_resumen = strip_tz_all(df_resumen)
        df_resumen.to_excel(writer, sheet_name="Resumen", index=False)

        # ---------------- Datos de proceso ----------------
        if not df_values.empty:
            df_values.to_excel(writer, sheet_name="Datos_proceso", index=False)

        # ---------------- Últimos valores ----------------
        if not df_last.empty:
            df_last.to_excel(writer, sheet_name="Ultimos_valores", index=False)

        # ---------------- Anomalías ----------------
        if include_anomaly_table and tabla_anomalias is not None and not tabla_anomalias.empty:
            tabla_anomalias.to_excel(writer, sheet_name="Anomalias", index=False)

        # ---------------- Historial ----------------
        if include_history and hist_df is not None and not hist_df.empty:
            hist_df.to_excel(writer, sheet_name="Historial_alertas", index=False)

        # ---------------- Ajuste de columnas ----------------
        for sheet_name, df_sheet in [
            ("Resumen", df_resumen),
            ("Datos_proceso", df_values),
            ("Ultimos_valores", df_last),
            ("Anomalias", tabla_anomalias if tabla_anomalias is not None else pd.DataFrame()),
            ("Historial_alertas", hist_df if hist_df is not None else pd.DataFrame()),
        ]:
            if sheet_name in writer.sheets and df_sheet is not None and not df_sheet.empty:
                ws = writer.sheets[sheet_name]
                for col_idx, col_name in enumerate(df_sheet.columns):
                    max_len = max(
                        [len(str(col_name))] +
                        [len(str(v)) for v in df_sheet[col_name].head(200)]
                    )
                    ws.set_column(col_idx, col_idx, min(max_len + 2, 60))

    output.seek(0)
    return output.getvalue()



def create_pdf_report(
    machine: str,
    report_title: str,
    report_subtitle: str,
    time_option: str,
    range_str: str,
    k_attention: float,
    k_alarm: float,
    sensores_sel: list[str],
    df_values: pd.DataFrame,
    df_last: pd.DataFrame,
    tabla_anomalias: pd.DataFrame,
    hist_df: pd.DataFrame,
    include_trend: bool = True,
    include_last_bar: bool = True,
    include_per_sensor: bool = True,
    include_anomaly_table: bool = True,
    include_history: bool = True,
    hist_max_rows: int = 20,
) -> bytes:
    """
    Genera un PDF en memoria con:
      - Título / subtítulo
      - filtros actuales
      - resumen
      - tabla de últimos valores
      - tabla de anomalías (si include_anomaly_table)
      - gráfico de tendencia multi-sensor (si include_trend)
      - gráfico de últimos valores (si include_last_bar)
      - gráficos por sensor (si include_per_sensor)
      - historial de alertas (si include_history)
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin_x = 40
    margin_y = 40
    y = height - margin_y

    # Título
    c.setFont("Helvetica-Bold", 16)
    title_to_draw = report_title.strip() if report_title.strip() else "Reporte de Inyectora OPC UA"
    c.drawString(margin_x, y, title_to_draw)
    y -= 20

    # Subtítulo
    c.setFont("Helvetica-Oblique", 10)
    if report_subtitle.strip():
        c.drawString(margin_x, y, report_subtitle.strip())
        y -= 16

    # Filtros
    c.setFont("Helvetica", 10)
    c.drawString(margin_x, y, f"Machine: {machine}")
    y -= 14
    c.drawString(margin_x, y, f"Rango de tiempo (proceso): {time_option}  ({range_str})")
    y -= 14
    c.drawString(margin_x, y, f"Sensores seleccionados: {', '.join(sensores_sel) if sensores_sel else 'Todos'}")
    y -= 14
    c.drawString(
        margin_x,
        y,
        f"Umbrales: Atención={k_attention:.1f}×STD, Alarma={k_alarm:.1f}×STD",
    )
    y -= 20

    # Resumen rápido
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "Resumen de datos")
    y -= 16
    c.setFont("Helvetica", 10)

    n_sensores = df_values["sensor"].nunique()
    n_puntos = len(df_values)
    last_time = df_values["time"].max() if not df_values.empty else None

    c.drawString(margin_x, y, f"Sensores activos (en reporte): {n_sensores}")
    y -= 14
    c.drawString(margin_x, y, f"Puntos en rango (para reporte): {n_puntos}")
    y -= 14
    if last_time is not None:
        c.drawString(margin_x, y, f"Último dato: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
        y -= 14
    y -= 6

    # Tabla: últimos valores
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_x, y, "Último valor por caudalímetro")
    y -= 16
    c.setFont("Helvetica", 9)

    c.drawString(margin_x, y, "Sensor")
    c.drawString(margin_x + 150, y, "Último valor")
    y -= 12

    df_last_print = df_last.sort_values("sensor").copy()
    max_rows_last = 25

    for _, row in df_last_print.head(max_rows_last).iterrows():
        if y < margin_y + 60:
            c.showPage()
            y = height - margin_y
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, y, "Último valor por caudalímetro (cont.)")
            y -= 16
            c.setFont("Helvetica", 9)
            c.drawString(margin_x, y, "Sensor")
            c.drawString(margin_x + 150, y, "Último valor")
            y -= 12

        c.drawString(margin_x, y, str(row["sensor"]))
        c.drawString(margin_x + 150, y, f"{row['value']:.2f}")
        y -= 12

    y -= 20

    # Tabla anomalías
    if include_anomaly_table and not tabla_anomalias.empty:
        if y < margin_y + 100:
            c.showPage()
            y = height - margin_y

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin_x, y, "Anomalías por sensor")
        y -= 16
        c.setFont("Helvetica", 9)

        c.drawString(margin_x, y, "Sensor")
        c.drawString(margin_x + 120, y, "Mean")
        c.drawString(margin_x + 200, y, "Diff")
        c.drawString(margin_x + 280, y, "Estado")
        y -= 12

        max_rows_anom = 25
        for _, row in tabla_anomalias.head(max_rows_anom).iterrows():
            if y < margin_y + 60:
                c.showPage()
                y = height - margin_y
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin_x, y, "Anomalías por sensor (cont.)")
                y -= 16
                c.setFont("Helvetica", 9)
                c.drawString(margin_x, y, "Sensor")
                c.drawString(margin_x + 120, y, "Mean")
                c.drawString(margin_x + 200, y, "Diff")
                c.drawString(margin_x + 280, y, "Estado")
                y -= 12

            c.drawString(margin_x, y, str(row["sensor"]))
            c.drawString(margin_x + 120, y, f"{row['mean_sensor']:.2f}")
            c.drawString(margin_x + 200, y, f"{row['diff']:.2f}")
            c.drawString(margin_x + 280, y, str(row["estado"]))
            y -= 12

        y -= 20

    # Nueva página para gráficos globales
    if include_trend or include_last_bar:
        c.showPage()
        y = height - margin_y

    # Gráfico de tendencia (multi-sensor)
    if include_trend and (not df_values.empty) and sensores_sel:
        df_plot_pdf = df_values[df_values["sensor"].isin(sensores_sel)].copy()

        fig, ax = plt.subplots(figsize=(8, 3))
        for sensor in sorted(df_plot_pdf["sensor"].unique()):
            df_s = df_plot_pdf[df_plot_pdf["sensor"] == sensor]
            ax.plot(df_s["time"], df_s["value"], label=sensor, linewidth=0.8)
        ax.set_title("Tendencia de caudalímetros (value)")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Value")
        ax.legend(fontsize=6, ncol=3)
        fig.tight_layout()

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        plt.close(fig)
        img_buf.seek(0)

        img = ImageReader(img_buf)
        img_width = width - 2 * margin_x
        img_height = img_width * 0.4
        c.drawImage(img, margin_x, y - img_height, width=img_width, height=img_height)
        y -= img_height + 20

    # Gráfico de últimos valores (bar)
    if include_last_bar and not df_last.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        df_last_sorted = df_last.sort_values("sensor")
        ax2.bar(df_last_sorted["sensor"], df_last_sorted["value"])
        ax2.set_title("Último valor por caudalímetro")
        ax2.set_xlabel("Sensor")
        ax2.set_ylabel("Value")
        plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=6)
        fig2.tight_layout()

        img_buf2 = io.BytesIO()
        fig2.savefig(img_buf2, format="png")
        plt.close(fig2)
        img_buf2.seek(0)

        img2 = ImageReader(img_buf2)
        img_width2 = width - 2 * margin_x
        img_height2 = img_width2 * 0.4
        if y - img_height2 < margin_y:
            c.showPage()
            y = height - margin_y
        c.drawImage(img2, margin_x, y - img_height2, width=img_width2, height=img_height2)
        y -= img_height2 + 20

    # Gráficos por sensor (detalle, uno por página)
    if include_per_sensor and (not df_values.empty) and sensores_sel:
        for sensor in sensores_sel:
            df_s = df_values[df_values["sensor"] == sensor]
            if df_s.empty:
                continue

            fig_s, ax_s = plt.subplots(figsize=(8, 3))
            ax_s.plot(df_s["time"], df_s["value"], linewidth=1.0)
            ax_s.set_title(f"Evolución temporal - {sensor}")
            ax_s.set_xlabel("Tiempo")
            ax_s.set_ylabel("Value")
            fig_s.tight_layout()

            img_buf_s = io.BytesIO()
            fig_s.savefig(img_buf_s, format="png")
            plt.close(fig_s)
            img_buf_s.seek(0)

            img_s = ImageReader(img_buf_s)
            img_width_s = width - 2 * margin_x
            img_height_s = img_width_s * 0.4

            c.showPage()
            y = height - margin_y
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, y, f"Detalle de sensor: {sensor}")
            y -= 20
            c.drawImage(img_s, margin_x, y - img_height_s, width=img_width_s, height=img_height_s)

    # -----------------------------------------
    # Página para historial de alertas
    # -----------------------------------------
    if include_history:
        from textwrap import wrap

        c.showPage()
        y = height - margin_y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin_x, y, "Historial de alertas / comentarios")
        y -= 18
        c.setFont("Helvetica", 9)

        if hist_df is None or hist_df.empty:
            c.drawString(margin_x, y, "Sin alertas registradas en el intervalo seleccionado.")
        else:
            used_rows = 0

            for _, row in hist_df.iterrows():
                if used_rows >= hist_max_rows:
                    break

                # Si queda poco espacio → nueva página
                if y < margin_y + 60:
                    c.showPage()
                    y = height - margin_y
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(margin_x, y, "Historial de alertas / comentarios (cont.)")
                    y -= 18
                    c.setFont("Helvetica", 9)

                # ------- Encabezado del evento -------
                fecha = row["time"]
                sev = row.get("severity", "")
                kind = row.get("kind", "")
                sensors = row.get("sensors", "")

                c.drawString(margin_x, y, f"{fecha} - {sev} - {kind} - {sensors}")
                y -= 12

                # ------- Normalizar texto -------
                texto = str(row.get("text", ""))

                # Reemplazar saltos de línea dobles
                texto = texto.replace("\r\n", "\n").replace("\r", "\n")

                # Separar por líneas lógicas
                lineas_logicas = texto.split("\n")

                # Para cada línea lógica: envolver texto a 110 chars
                lineas_finales = []
                for ln in lineas_logicas:
                    if ln.strip():
                        lineas_finales.extend(wrap(ln.strip(), width=110))
                    else:
                        lineas_finales.append("")   # mantener líneas vacías

                # Limitar líneas por evento
                for line in lineas_finales[:6]:
                    if y < margin_y + 40:
                        c.showPage()
                        y = height - margin_y
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(margin_x, y, "Historial de alertas / comentarios (cont.)")
                        y -= 18
                        c.setFont("Helvetica", 9)

                    c.drawString(margin_x + 10, y, line)
                    y -= 10

                y -= 6
                used_rows += 1

    # CIERRE PDF
    c.showPage()
    c.save()


    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def process_state_alerts(
    machine: str,
    mold,
    automatic,
    pulse,
    infra_ok: bool,
    alarm_sensors: set[str],
):
    """
    Detecta cambios en estados clave y genera alertas automáticas.
    """
    if "last_automatic" not in st.session_state:
        st.session_state["last_automatic"] = automatic
    if "last_pulse" not in st.session_state:
        st.session_state["last_pulse"] = pulse
    if "last_mold" not in st.session_state:
        st.session_state["last_mold"] = mold
    if "last_infra_ok" not in st.session_state:
        st.session_state["last_infra_ok"] = infra_ok
    if "last_alarm_sensors" not in st.session_state:
        st.session_state["last_alarm_sensors"] = list(alarm_sensors)

    prev_auto = st.session_state["last_automatic"]
    prev_pulse = st.session_state["last_pulse"]
    prev_mold = st.session_state["last_mold"]
    prev_infra_ok = st.session_state["last_infra_ok"]
    prev_alarm_sensors = set(st.session_state["last_alarm_sensors"])

    alerts = []

    if prev_auto is not None and automatic is not None and automatic != prev_auto:
        alerts.append(
            f"Modo automático cambió de {'Sí' if prev_auto else 'No'} a {'Sí' if automatic else 'No'}."
        )

    if prev_pulse is not None and pulse is not None and pulse != prev_pulse:
        alerts.append(
            f"Pulso de inyección cambió de {'Activo' if prev_pulse else 'Inactivo'} a {'Activo' if pulse else 'Inactivo'}."
        )

    if prev_mold is not None and mold is not None and mold != prev_mold:
        alerts.append(f"Molde cambió de {prev_mold} a {mold}.")

    if prev_infra_ok is not None and infra_ok != prev_infra_ok:
        if infra_ok:
            alerts.append("Infraestructura volvió a estado OK (API + Influx + datos frescos).")
        else:
            alerts.append("Problemas en infraestructura (API/Influx/datos).")

    nuevos_alarmas = alarm_sensors - prev_alarm_sensors
    if nuevos_alarmas:
        lista = ", ".join(sorted(nuevos_alarmas))
        alerts.append(f"Nuevos sensores en ALARMA: {lista}.")

    st.session_state["last_automatic"] = automatic
    st.session_state["last_pulse"] = pulse
    st.session_state["last_mold"] = mold
    st.session_state["last_infra_ok"] = infra_ok
    st.session_state["last_alarm_sensors"] = list(alarm_sensors)

    if alerts:
        texto = "Alertas inyectora:\n" + "\n".join(alerts)

        severity = "info"
        if alarm_sensors or not infra_ok:
            severity = "critical"

        ok = send_telegram_alert(texto)
        write_alert_event(
            kind="auto_state",
            text=texto,
            machine=machine,
            severity=severity,
            sensors=list(alarm_sensors),
            sent_telegram=ok,
        )
        return ok, alerts

    return False, []


# ==========================
# UI STREAMLIT
# ==========================

st.set_page_config(
    page_title="Inyectora OPC UA - Dashboard Python",
    layout="wide",
)

st.title("Inyectora OPC UA - Dashboard en Python")

if st.button("Actualizar ahora", key="btn_refresh"):
    st.rerun()

# ----- SIDEBAR -----
st.sidebar.header("Filtros")

time_option = st.sidebar.selectbox(
    "Rango de tiempo (proceso)",
    options=["Últimos 15 minutos", "Últimos 30 minutos", "Última 1 hora", "Últimas 6 horas"],
    index=1,
)

range_map = {
    "Últimos 15 minutos": "-15m",
    "Últimos 30 minutos": "-30m",
    "Última 1 hora": "-1h",
    "Últimas 6 horas": "-6h",
}
range_str = range_map[time_option]

machine = st.sidebar.text_input("Machine tag", "INY_01")

st.sidebar.markdown("---")
st.sidebar.subheader("Límites de anomalía (en múltiplos de STD)")
k_attention = st.sidebar.number_input(
    "Factor STD para 'Atención'",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
)
k_alarm = st.sidebar.number_input(
    "Factor STD para 'Alarma'",
    min_value=0.2,
    max_value=10.0,
    value=2.0,
    step=0.1,
)

# ---- Cargar datos de proceso ----
df_values = get_values_df(range_str=range_str, machine=machine)

if df_values.empty:
    st.warning("No se encontraron datos para el rango seleccionado.")
    st.stop()

sensores_disponibles = sorted(df_values["sensor"].unique().tolist())

# Modo de selección de sensores
modo_unico = st.sidebar.checkbox("Ver solo un caudalímetro", value=False)

if modo_unico:
    sensor_unico = st.sidebar.selectbox(
        "Caudalímetro",
        options=sensores_disponibles,
        index=0,
    )
    sensores_sel = [sensor_unico]
else:
    sensores_sel = st.sidebar.multiselect(
        "Sensores",
        options=sensores_disponibles,
        default=sensores_disponibles,
    )

df_plot = df_values[df_values["sensor"].isin(sensores_sel)]

# KPIs
mold = get_mold(machine)
automatic = get_last_bool("automatic", machine)
pulse = get_last_bool("injection_pulse", machine)
mean_val, std_val = get_global_stats(range_str=range_str, machine=machine)

n_sensores = df_values["sensor"].nunique()
n_puntos = len(df_values)
last_time = df_values["time"].max() if not df_values.empty else None

tabla_anomalias = build_anomaly_table(
    df_values,
    mean_val,
    std_val,
    k_attention=k_attention,
    k_alarm=k_alarm,
)
current_alarms = set()
if not tabla_anomalias.empty:
    current_alarms = set(
        tabla_anomalias[tabla_anomalias["estado"] == "Alarma"]["sensor"].values
    )

# Salud de infraestructura
api_ok, api_msg = check_api_health()
influx_ok, influx_msg = check_influx_health()
fresh_ok, age_sec = check_data_freshness(last_time)

infra_ok = api_ok and influx_ok and (fresh_ok if fresh_ok is not None else False)

# Alertas automáticas
alert_sent, alert_msgs = process_state_alerts(
    machine=machine,
    mold=mold,
    automatic=automatic,
    pulse=pulse,
    infra_ok=infra_ok,
    alarm_sensors=current_alarms,
)

# ==========================
# Pestañas
# ==========================
tab_overview, tab_trends, tab_detail, tab_anom, tab_history = st.tabs(
    ["Visión general", "Tendencias", "Detalle por sensor", "Anomalías", "Historial"]
)

# --------------------------
# TAB 1 - VISIÓN GENERAL
# --------------------------
with tab_overview:
    st.subheader("Resumen de estado")

    col0, col1, col2 = st.columns(3)
    with col0:
        st.metric("Sensores activos", n_sensores)
        st.metric("Puntos en rango", n_puntos)
    with col1:
        st.metric("Molde actual", mold if mold is not None else "N/A")
        st.metric("Promedio global (value)", f"{mean_val:.2f}" if mean_val is not None else "N/A")
    with col2:
        if last_time is not None:
            st.metric("Último dato", last_time.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            st.metric("Último dato", "N/A")
        st.metric("STD global", f"{std_val:.2f}" if std_val is not None else "N/A")

    st.markdown(
        f"*Límites actuales*: Atención = |diff| < {k_attention:.1f}×STD, "
        f"Alarma = |diff| ≥ {k_alarm:.1f}×STD"
    )

    st.markdown("---")
    st.subheader("Salud de la infraestructura")

    colI1, colI2, colI3 = st.columns(3)

    with colI1:
        if influx_ok:
            st.success(influx_msg)
        else:
            st.error(influx_msg)

    with colI2:
        if api_ok:
            st.success(api_msg)
        else:
            st.error(api_msg)

    with colI3:
        if last_time is None or age_sec is None:
            st.warning("No se puede determinar la frescura de datos.")
        else:
            if fresh_ok:
                st.success(f"Datos recientes (edad ~ {age_sec:.1f}s)")
            else:
                st.error(f"Datos desactualizados (edad ~ {age_sec:.1f}s)")

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Modo de operación")
        st.metric("Modo automático", "Sí" if automatic else "No")
        st.metric("Pulso de inyección", "Activo" if pulse else "Inactivo")
    with colB:
        st.subheader("Comentario rápido")
        if std_val is None:
            st.info("Aún no hay datos suficientes para evaluar estabilidad.")
        else:
            if std_val < 5:
                st.success("Sistema estable: baja dispersión entre caudalímetros.")
            elif std_val < 10:
                st.warning("Sistema con variación moderada. Conviene revisar tendencias.")
            else:
                st.error("Alta dispersión entre caudalímetros. Revisar circuito de refrigeración/inyección.")

    st.markdown("---")

    if alert_msgs:
        st.info("Cambios de estado detectados:")
        for m in alert_msgs:
            st.write(f"- {m}")
        if TELEGRAM_ENABLED:
            if alert_sent:
                st.success("Alertas automáticas enviadas por Telegram.")
            else:
                st.warning("No se pudieron enviar las alertas por Telegram.")
    else:
        st.caption("Sin cambios de estado relevantes desde la última actualización.")

    st.markdown("### Enviar resumen manual")
    if st.button("Enviar resumen ahora (Telegram)", key="btn_manual_telegram"):
        resumen = [
            f"Resumen inyectora {machine}:",
            f"- Molde: {mold if mold is not None else 'N/A'}",
            f"- Automático: {'Sí' if automatic else 'No'}",
            f"- Pulso: {'Activo' if pulse else 'Inactivo'}",
            f"- Sensores activos: {n_sensores}",
            f"- STD global: {std_val:.2f}" if std_val is not None else "- STD global: N/A",
            f"- Umbrales: Atención={k_attention:.1f}×STD, Alarma={k_alarm:.1f}×STD",
        ]
        if current_alarms:
            resumen.append("Sensores en ALARMA: " + ", ".join(sorted(current_alarms)))
        else:
            resumen.append("Sensores en ALARMA: ninguno.")

        txt = "\n".join(resumen)
        ok_manual = send_telegram_alert(txt)
        write_alert_event(
            kind="manual_summary",
            text=txt,
            machine=machine,
            severity="warning" if current_alarms else "info",
            sensors=list(current_alarms),
            sent_telegram=ok_manual,
        )
        if ok_manual:
            st.success("Resumen enviado vía Telegram.")
        else:
            st.warning("No se pudo enviar el resumen. Ver configuración de Telegram.")

    # --------------------------
    # SECCIÓN: REPORTE PDF
    # --------------------------
    st.markdown("---")
    st.subheader("Generar reporte en PDF")

    # Título y subtítulo del reporte
    report_title = st.text_input(
        "Título del reporte",
        value=f"Reporte de Inyectora {machine}",
        key="pdf_title",
    )
    report_subtitle = st.text_input(
        "Subtítulo (cliente, planta, comentario general)",
        value="",
        key="pdf_subtitle",
    )

    st.markdown("**Filtros para el reporte**")

    # 1) Rango de tiempo de proceso para el reporte
    use_dashboard_range = st.checkbox(
        "Usar el mismo rango de tiempo del dashboard",
        value=True,
        key="pdf_use_dash_range",
    )

    pdf_time_option = time_option
    pdf_range_str = range_str

    if not use_dashboard_range:
        pdf_time_option = st.selectbox(
            "Rango de tiempo (proceso) para el reporte",
            options=["Últimos 15 minutos", "Últimos 30 minutos", "Última 1 hora", "Últimas 6 horas"],
            index=1,
            key="pdf_time_option",
        )
        range_map_pdf = {
            "Últimos 15 minutos": "-15m",
            "Últimos 30 minutos": "-30m",
            "Última 1 hora": "-1h",
            "Últimas 6 horas": "-6h",
        }
        pdf_range_str = range_map_pdf[pdf_time_option]

    # 2) Sensores a incluir en el reporte
    sensores_pdf = st.multiselect(
        "Sensores a incluir en el reporte",
        options=sensores_disponibles,
        default=sensores_sel if sensores_sel else sensores_disponibles,
        key="sensores_pdf",
    )

    # 3) Cómo mostrar la tabla de anomalías
    anom_mode = st.selectbox(
        "Tabla de anomalías a incluir",
        options=[
            "Todos los sensores",
            "Solo Atención y Alarma",
            "Solo Alarma",
            "No incluir tabla de anomalías",
        ],
        index=1,
        key="anom_mode_pdf",
    )

    # Datetimes “locales” (sin zona horaria explícita)
    from datetime import datetime, date, time as dtime, timedelta

    st.markdown("**Rango de tiempo para historial de alertas (en el PDF)**")

    col_from, col_to = st.columns(2)

    with col_from:
        hist_date_from = st.date_input(
            "Desde (fecha)",
            value=date.today() - timedelta(days=1),
            key="hist_date_from",
        )
        hist_time_from = st.time_input(
            "Desde (hora)",
            value=dtime(0, 0),
            key="hist_time_from",
        )

    with col_to:
        hist_date_to = st.date_input(
            "Hasta (fecha)",
            value=date.today(),
            key="hist_date_to",
        )
        hist_time_to = st.time_input(
            "Hasta (hora)",
            value=dtime(23, 59),
            key="hist_time_to",
        )

    # Combinar fecha + hora
    hist_start_dt = datetime.combine(hist_date_from, hist_time_from)
    hist_end_dt = datetime.combine(hist_date_to, hist_time_to)

    if hist_end_dt < hist_start_dt:
        st.error("La fecha/hora 'Hasta' debe ser posterior a 'Desde'.")

    hist_sev_filter = st.selectbox(
        "Severidad mínima del historial",
        options=[
            "Todas",
            "Solo Warning y Critical",
            "Solo Critical",
        ],
        index=0,
        key="hist_sev_filter",
    )

    hist_max_rows = st.number_input(
        "Máximo de filas de historial a incluir",
        min_value=5,
        max_value=200,
        value=30,
        step=5,
        key="hist_max_rows",
    )

    # 5) Contenido del reporte
    st.markdown("**Contenido del reporte**")
    include_trend = st.checkbox(
        "Incluir gráfico de tendencia global",
        value=True,
        key="pdf_inc_trend",
    )
    include_last_bar = st.checkbox(
        "Incluir gráfico de últimos valores",
        value=True,
        key="pdf_inc_last_bar",
    )
    include_per_sensor = st.checkbox(
        "Incluir gráficos detallados por sensor (una página por sensor)",
        value=True,
        key="pdf_inc_per_sensor",
    )
    include_anomaly_table = anom_mode != "No incluir tabla de anomalías"
    include_history = st.checkbox(
        "Incluir historial de alertas / comentarios",
        value=True,
        key="pdf_inc_history",
    )

    # -------------------------------
    # BOTÓN: GENERAR REPORTE EN PDF
    # -------------------------------
    if st.button("Generar reporte PDF", key="btn_pdf"):

        # Si el usuario dejó el multiselect vacío, uso todos
        if not sensores_pdf:
            sensores_pdf = sensores_disponibles

        # Datos de proceso específicos para el reporte
        df_values_pdf = get_values_df(range_str=pdf_range_str, machine=machine)
        df_values_pdf = df_values_pdf[df_values_pdf["sensor"].isin(sensores_pdf)]

        if df_values_pdf.empty:
            st.error("No hay datos en el rango y sensores seleccionados para el reporte.")
        else:
            # Último valor por sensor (para ese rango)
            df_last_report = (
                df_values_pdf.sort_values("time")
                .groupby("sensor")
                .tail(1)
            )

            # Recalcular estadísticas globales para ese rango
            mean_report, std_report = get_global_stats(
                range_str=pdf_range_str,
                machine=machine,
            )

            # Tabla de anomalías para el reporte
            tabla_anom_report = build_anomaly_table(
                df_values_pdf,
                mean_report,
                std_report,
                k_attention=k_attention,
                k_alarm=k_alarm,
            )

            if not tabla_anom_report.empty:
                # Filtro de severidad para la tabla (según selección)
                if anom_mode == "Solo Atención y Alarma":
                    tabla_anom_report = tabla_anom_report[
                        tabla_anom_report["estado"].isin(["Atención", "Alarma"])
                    ]
                elif anom_mode == "Solo Alarma":
                    tabla_anom_report = tabla_anom_report[
                        tabla_anom_report["estado"] == "Alarma"
                    ]

                # Filtrar solo sensores seleccionados
                tabla_anom_report = tabla_anom_report[
                    tabla_anom_report["sensor"].isin(sensores_pdf)
                ]

            # Historial de alertas en el rango elegido por calendario/reloj
            hist_df_report = get_alert_history(
                machine=machine,
                start_dt=hist_start_dt,
                end_dt=hist_end_dt,
            )

            if not hist_df_report.empty:
                if hist_sev_filter == "Solo Warning y Critical":
                    hist_df_report = hist_df_report[
                        hist_df_report["severity"].isin(["warning", "critical"])
                    ]
                elif hist_sev_filter == "Solo Critical":
                    hist_df_report = hist_df_report[
                        hist_df_report["severity"] == "critical"]
            
            # Crear PDF
            pdf_bytes = create_pdf_report(
                machine=machine,
                report_title=report_title,
                report_subtitle=report_subtitle,
                time_option=pdf_time_option,
                range_str=pdf_range_str,
                k_attention=k_attention,
                k_alarm=k_alarm,
                sensores_sel=sensores_pdf,
                df_values=df_values_pdf,
                df_last=df_last_report,
                tabla_anomalias=tabla_anom_report,
                hist_df=hist_df_report,
                include_trend=include_trend,
                include_last_bar=include_last_bar,
                include_per_sensor=include_per_sensor,
                include_anomaly_table=include_anomaly_table,
                include_history=include_history,
                hist_max_rows=int(hist_max_rows),
            )

            st.download_button(
                label="Descargar reporte PDF",
                data=pdf_bytes,
                file_name=f"reporte_inyectora_{machine}.pdf",
                mime="application/pdf",
            )

    # -------------------------------
    # BOTÓN: GENERAR REPORTE EN EXCEL
    # -------------------------------
    if st.button("Generar reporte Excel", key="btn_excel"):

        # Si el usuario dejó el multiselect vacío, uso todos
        if not sensores_pdf:
            sensores_pdf = sensores_disponibles

        # Datos de proceso específicos para el reporte
        df_values_excel = get_values_df(range_str=pdf_range_str, machine=machine)
        df_values_excel = df_values_excel[df_values_excel["sensor"].isin(sensores_pdf)]

        if df_values_excel.empty:
            st.error("No hay datos en el rango y sensores seleccionados para el Excel.")
        else:
            # Último valor por sensor (para ese rango)
            df_last_excel = (
                df_values_excel.sort_values("time")
                .groupby("sensor")
                .tail(1)
            )

            # Recalcular estadísticas globales para ese rango
            mean_excel, std_excel = get_global_stats(
                range_str=pdf_range_str,
                machine=machine,
            )

            # Tabla de anomalías para el Excel (misma lógica que para el PDF)
            tabla_anom_excel = build_anomaly_table(
                df_values_excel,
                mean_excel,
                std_excel,
                k_attention=k_attention,
                k_alarm=k_alarm,
            )

            if not tabla_anom_excel.empty:
                if anom_mode == "Solo Atención y Alarma":
                    tabla_anom_excel = tabla_anom_excel[
                        tabla_anom_excel["estado"].isin(["Atención", "Alarma"])
                    ]
                elif anom_mode == "Solo Alarma":
                    tabla_anom_excel = tabla_anom_excel[
                        tabla_anom_excel["estado"] == "Alarma"
                    ]

                tabla_anom_excel = tabla_anom_excel[
                    tabla_anom_excel["sensor"].isin(sensores_pdf)
                ]

            # Historial de alertas usando el mismo rango elegido en el formulario
            hist_df_excel = get_alert_history(
                machine=machine,
                start_dt=hist_start_dt,
                end_dt=hist_end_dt,
            )

            if not hist_df_excel.empty:
                if hist_sev_filter == "Solo Warning y Critical":
                    hist_df_excel = hist_df_excel[
                        hist_df_excel["severity"].isin(["warning", "critical"])
                    ]
                elif hist_sev_filter == "Solo Critical":
                    hist_df_excel = hist_df_excel[
                        hist_df_excel["severity"] == "critical"
                    ]

            # Crear Excel (mismas opciones que el PDF en lo posible)
            excel_bytes = create_excel_report(
                machine=machine,
                report_title=report_title,
                report_subtitle=report_subtitle,
                time_option=pdf_time_option,
                range_str=pdf_range_str,
                k_attention=k_attention,
                k_alarm=k_alarm,
                sensores_sel=sensores_pdf,
                df_values=df_values_excel,
                df_last=df_last_excel,
                tabla_anomalias=tabla_anom_excel,
                hist_df=hist_df_excel,
                include_anomaly_table=include_anomaly_table,
                include_history=include_history,
            )

            st.download_button(
                label="Descargar reporte Excel",
                data=excel_bytes,
                file_name=f"reporte_inyectora_{machine}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# --------------------------
# TAB 2 - TENDENCIAS
# --------------------------
with tab_trends:
    st.subheader("Tendencia de caudalímetros (value)")

    fig_ts = px.line(
        df_plot,
        x="time",
        y="value",
        color="sensor",
        title="Evolución temporal de los caudalímetros"
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Último valor por caudalímetro")
    df_last = df_values.sort_values("time").groupby("sensor").tail(1)
    df_last = df_last[df_last["sensor"].isin(sensores_sel)]

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            df_last.set_index("sensor")[["value"]],
            use_container_width=True
        )
    with col2:
        fig_bar = px.bar(
            df_last,
            x="sensor",
            y="value",
            title="Último valor registrado"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------
# TAB 3 - DETALLE POR SENSOR
# --------------------------
with tab_detail:
    st.subheader("Detalle por caudalímetro")

    sensor_det = st.selectbox(
        "Seleccionar caudalímetro",
        options=sensores_disponibles,
        index=0,
        key="sensor_detalle"
    )

    df_det = df_values[df_values["sensor"] == sensor_det]

    if df_det.empty:
        st.write("Sin datos para el sensor seleccionado.")
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric(
                "Último valor",
                f"{df_det['value'].iloc[-1]:.2f}"
            )
        with col_t2:
            st.metric(
                "Mín / Máx en rango",
                f"{df_det['value'].min():.2f} / {df_det['value'].max():.2f}"
            )

        fig_det = px.line(
            df_det,
            x="time",
            y="value",
            title=f"Evolución temporal - {sensor_det}"
        )
        st.plotly_chart(fig_det, use_container_width=True)

        st.expander("Ver datos crudos").dataframe(df_det, use_container_width=True)

# --------------------------
# TAB 4 - ANOMALÍAS
# --------------------------
with tab_anom:
    st.subheader("Análisis de anomalías por sensor")

    if tabla_anomalias.empty:
        st.info("No se pudo calcular la tabla de anomalías (falta de datos o STD=0 o límites inválidos).")
    else:
        st.markdown(
            "Clasificación basada en la diferencia entre el promedio de cada sensor y el promedio global.\n\n"
            f"- **OK**: |diff| < {k_attention:.1f}×STD\n"
            f"- **Atención**: {k_attention:.1f}×STD ≤ |diff| < {k_alarm:.1f}×STD\n"
            f"- **Alarma**: |diff| ≥ {k_alarm:.1f}×STD"
        )
        st.dataframe(
            tabla_anomalias.set_index("sensor")[["mean_sensor", "diff", "estado"]],
            use_container_width=True
        )

        alarmas = tabla_anomalias[tabla_anomalias["estado"] == "Alarma"]
        if not alarmas.empty:
            st.warning(f"Hay {len(alarmas)} sensores en estado ALARMA.")
            if st.button("Enviar alerta global de anomalías (Telegram)", key="btn_alerta_telegram"):
                texto = "Alertas en caudalímetros (global):\n"
                sensores_alerta = []
                for _, row in alarmas.iterrows():
                    texto += f"- {row['sensor']}: mean={row['mean_sensor']:.2f}, diff={row['diff']:.2f}\n"
                    sensores_alerta.append(row["sensor"])

                ok_alert = send_telegram_alert(texto)
                write_alert_event(
                    kind="anomaly_alert",
                    text=texto,
                    machine=machine,
                    severity="critical",
                    sensors=sensores_alerta,
                    sent_telegram=ok_alert,
                )
                if ok_alert:
                    st.success("Alerta de anomalías enviada vía Telegram.")
                else:
                    st.info("No se pudo enviar la alerta. Ver configuración TELEGRAM_* en el código.")

        st.markdown("---")
        st.subheader("Documentar una anomalía específica")

        # Seleccionar sensor
        sensores_anom = tabla_anomalias["sensor"].tolist()
        sensor_com = st.selectbox(
            "Seleccionar caudalímetro para comentar",
            options=sensores_anom,
            index=0,
            key="sensor_comentario"
        )

        row_sel = tabla_anomalias[tabla_anomalias["sensor"] == sensor_com].iloc[0]

        st.write(f"Estado actual: **{row_sel['estado']}**")
        st.write(f"Mean sensor (ventana): {row_sel['mean_sensor']:.2f}")
        st.write(f"Diferencia vs global (ventana): {row_sel['diff']:.2f}")

        # Valores posibles para ese sensor (para evitar errores de tipeo)
        df_sensor_actual = df_values[df_values["sensor"] == sensor_com]

        if df_sensor_actual.empty:
            st.info("No hay valores registrados para este sensor en el rango actual.")
            valor_observado = None
            diff_observado = None
        else:
            # Tomar últimos valores, redondear e ignorar duplicados
            valores_candidatos = (
                df_sensor_actual.sort_values("time", ascending=False)["value"]
                .round(2)
                .drop_duplicates()
                .head(100)  # límite para no hacer la lista enorme
                .tolist()
            )

            if not valores_candidatos:
                st.info("No hay valores disponibles para listar.")
                valor_observado = None
                diff_observado = None
            else:
                valor_observado = st.selectbox(
                    "Valor observado (tomado de datos reales del sensor)",
                    options=valores_candidatos,
                    index=0,
                    key="sel_valor_observado"
                )

                if mean_val is not None:
                    diff_observado = float(valor_observado) - mean_val
                    st.write(f"Diferencia del valor observado vs global: {diff_observado:.2f}")
                else:
                    diff_observado = None
                    st.info("No se puede calcular la diferencia del valor observado (mean global = None).")

        comentario = st.text_area(
            "Descripción / diagnóstico técnico",
            value="",
            help="Ejemplo: posible obstrucción parcial, revisar válvula, diferencia de caudal entre circuitos, etc.",
            key="txt_anom_desc"
        )

        enviar_tele = st.checkbox(
            "Enviar también este comentario por Telegram",
            value=False,
            key="chk_send_tele_comment"
        )

        if st.button("Guardar descripción de anomalía", key="btn_save_anom_desc"):
            if valor_observado is None:
                st.warning("No hay valor observado disponible para este sensor.")
            elif not comentario.strip():
                st.warning("Ingresá una descripción antes de guardar.")
            else:
                texto = (
                    "[Comentario anomalía]\n"
                    f"Machine: {machine}\n"
                    f"Sensor: {sensor_com}\n"
                    f"Estado: {row_sel['estado']}\n"
                    f"Mean sensor (ventana): {row_sel['mean_sensor']:.2f}\n"
                    f"Diferencia vs global (ventana): {row_sel['diff']:.2f}\n"
                    f"Valor observado (seleccionado): {float(valor_observado):.2f}\n"
                )

                if diff_observado is not None:
                    texto += f"Diferencia valor observado vs global: {diff_observado:.2f}\n"

                texto += (
                    f"Umbrales: Atención={k_attention:.1f}×STD, Alarma={k_alarm:.1f}×STD\n"
                    f"Comentario técnico: {comentario}"
                )

                sev_map = {
                    "OK": "info",
                    "Atención": "warning",
                    "Alarma": "critical",
                }
                severity = sev_map.get(row_sel["estado"], "info")

                extra_fields = {
                    "mean_sensor": float(row_sel["mean_sensor"]),
                    "diff_sensor": float(row_sel["diff"]),
                    "observed_value": float(valor_observado),
                }
                if diff_observado is not None:
                    extra_fields["diff_observed"] = float(diff_observado)

                sent = False
                if enviar_tele:
                    sent = send_telegram_alert(texto)

                write_alert_event(
                    kind="anomaly_comment",
                    text=texto,
                    machine=machine,
                    severity=severity,
                    sensors=[sensor_com],
                    sent_telegram=sent,
                    extra_fields=extra_fields,
                )

                if enviar_tele:
                    if sent:
                        st.success("Comentario guardado y enviado por Telegram.")
                    else:
                        st.warning("Comentario guardado, pero no se pudo enviar por Telegram.")
                else:
                    st.success("Comentario de anomalía guardado en InfluxDB.")

# --------------------------
# TAB 5 - HISTORIAL
# --------------------------
with tab_history:
    st.subheader("Historial de alertas / comentarios (InfluxDB)")

    hist_range_option = st.selectbox(
        "Rango de tiempo (historial)",
        options=[
            "Última 1 hora",
            "Últimas 6 horas",
            "Últimas 24 horas",
            "Últimos 7 días",
            "Últimos 30 días",
        ],
        index=2,
    )

    hist_range_map = {
        "Última 1 hora": "-1h",
        "Últimas 6 horas": "-6h",
        "Últimas 24 horas": "-24h",
        "Últimos 7 días": "-7d",
        "Últimos 30 días": "-30d",
    }

    hist_df = get_alert_history(
        range_str=hist_range_map[hist_range_option],
        machine=machine,
    )

    if hist_df.empty:
        st.info("No hay alertas/comentarios registrados en el intervalo seleccionado.")
    else:
        st.dataframe(
            hist_df,
            use_container_width=True,
        )

        st.markdown("### Detalle de una alerta/comentario")
        idx = st.number_input(
            "Índice (0 = más reciente)",
            min_value=0,
            max_value=len(hist_df) - 1,
            value=0,
            step=1,
        )
        row = hist_df.iloc[int(idx)]
        st.write(f"Fecha/hora: {row['time']}")
        if "severity" in row:
            st.write(f"Severidad: {row['severity']}")
        if "kind" in row:
            st.write(f"Tipo: {row['kind']}")
        if "sensors" in row and isinstance(row["sensors"], str) and row["sensors"]:
            st.write(f"Sensores asociados: {row['sensors']}")
        if "observed_value" in row:
            st.write(f"Valor observado: {row['observed_value']}")
        if "diff_observed" in row:
            st.write(f"Diferencia valor observado vs global: {row['diff_observed']}")
        st.markdown("Texto:")
        st.code(row["text"])
