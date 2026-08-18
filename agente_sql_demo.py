"""
Agente de IA (Texto -> SQL) - MODO DEMO / RESPALDO
=====================================================
Parte D: Inteligencia Artificial y Agentes de Automatización

Por qué existe este archivo:
    agente_sql.py es el agente real: recibe la pregunta en texto libre y
    un LLM (OpenAI) genera el SQL dinámicamente. Ese es el entregable
    principal de la Parte D.

    Este archivo es un respaldo para la exposición en vivo, por si en
    ese momento no hay cuota/crédito de API disponible. Ejecuta el mismo
    tipo de preguntas, sobre la misma base de datos real (ventas_retail.db),
    pero con el SQL ya definido en vez de generado por el LLM en el
    momento. Los datos que devuelve son 100% reales, no inventados —
    solo se omite el paso de "traducción automática lenguaje -> SQL"
    que en producción hace el modelo de lenguaje.

    Ser transparente sobre esto en la presentación es parte de la buena
    práctica profesional: un ingeniero serio siempre tiene un plan B para
    una demo en vivo que depende de un servicio externo de pago.
"""

import sqlite3

DB_FILE = "ventas_retail.db"


PREGUNTAS_Y_SQL = [
    {
        "pregunta": "¿Cuáles fueron las ventas netas totales de la marca La Favorita?",
        "sql": "SELECT ROUND(SUM(venta_neta), 2) FROM ventas WHERE dsca_marca = 'La Favorita'",
        "plantilla_respuesta": "Las ventas netas totales de la marca La Favorita fueron de ${valor}.",
    },
    {
        "pregunta": "¿Cuál es el cliente que más compró en kilos durante todo el periodo?",
        "sql": """
            SELECT dsca_cliente, ROUND(SUM(qty_kilos), 2) as kilos
            FROM ventas
            GROUP BY dsca_cliente
            ORDER BY kilos DESC
            LIMIT 1
        """,
        "plantilla_respuesta": None,  # se arma distinto por ser 2 columnas
    },
    {
        "pregunta": "¿Cuántos kilos se vendieron en total en el mes de julio de 2026?",
        "sql": "SELECT ROUND(SUM(qty_kilos), 2) FROM ventas WHERE mes = 7 AND anio = 2026",
        "plantilla_respuesta": "En julio de 2026 se vendieron un total de {valor} kg.",
    },
]


def ejecutar_demo() -> None:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("MODO DEMO — respuestas generadas sobre datos reales de "
          f"'{DB_FILE}', con SQL predefinido (respaldo sin dependencia de API).\n")

    for item in PREGUNTAS_Y_SQL:
        print(f"{'=' * 70}\nPREGUNTA: {item['pregunta']}\n{'=' * 70}")
        print(f"SQL ejecutado:\n{item['sql'].strip()}\n")

        cursor.execute(item["sql"])
        resultado = cursor.fetchone()

        if item["plantilla_respuesta"]:
            respuesta = item["plantilla_respuesta"].format(valor=resultado[0])
        else:
            # Caso especial: pregunta de cliente top (2 columnas)
            respuesta = (f"El cliente que más compró en kilos fue "
                         f"{resultado[0]}, con {resultado[1]} kg en total.")

        print(f"RESPUESTA: {respuesta}\n")

    conn.close()


if __name__ == "__main__":
    ejecutar_demo()
