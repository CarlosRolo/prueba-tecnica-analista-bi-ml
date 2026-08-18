"""
Construcción de Base de Datos - Prueba Técnica Analista BI & ML
==================================================================
Parte D (paso 1): convierte el dataset limpio de la Parte A en una base
de datos SQLite local, que es la que consultará el Agente de IA.

Fuente: ventas_limpio.csv (generado por etl_ventas.py, Parte A)
Salida: ventas_retail.db
"""

import sqlite3
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("ventas_limpio.csv")
DB_FILE = Path("ventas_retail.db")
TABLE_NAME = "ventas"


def main() -> None:
    print(f"Leyendo {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, sep=";", decimal=",")
    df["fecha_factura"] = pd.to_datetime(df["fecha_factura"])

    print(f"Filas a cargar: {df.shape[0]}")

    conn = sqlite3.connect(DB_FILE)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    # Verificación rápida: cuenta filas y muestra el esquema resultante
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    total = cursor.fetchone()[0]
    print(f"Filas cargadas en '{TABLE_NAME}': {total}")

    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    print("\nEsquema de la tabla:")
    for col in cursor.fetchall():
        print(f"  {col[1]:20s} {col[2]}")

    conn.close()
    print(f"\nBase de datos generada: {DB_FILE.resolve()}")


if __name__ == "__main__":
    main()
