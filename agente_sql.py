"""
Agente de IA (Texto -> SQL) - Prueba Técnica Analista BI & ML
=================================================================
Parte D: Inteligencia Artificial y Agentes de Automatización

Objetivo:
    Recibir una consulta en lenguaje natural, traducirla a SQL, consultar
    la base de datos local (ventas_retail.db) y devolver una respuesta
    resumida en español.

Requisitos previos:
    1. Haber corrido crear_db.py (genera ventas_retail.db)
    2. pip install langchain langchain-community langchain-openai
    3. Una API key de OpenAI en la variable de entorno OPENAI_API_KEY
"""

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

DB_FILE = "ventas_retail.db"

# ------------------------------------------------------------------
# Conexión a la base de datos local generada en la Parte A/D
# ------------------------------------------------------------------
db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")

# ------------------------------------------------------------------
# Modelo de lenguaje que traduce la pregunta a SQL
# ------------------------------------------------------------------
# Requiere la variable de entorno OPENAI_API_KEY configurada.
# temperature=0 -> respuestas deterministas, ideal para generar SQL.
# gpt-4o-mini es el modelo económico de OpenAI, suficiente para esquemas
# pequeños como este (una sola tabla, 20 columnas).
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------------------------------------------------------
# Creación del agente ejecutor de consultas SQL
# ------------------------------------------------------------------
agent_executor = create_sql_agent(llm, db=db, verbose=True)


def preguntar(pregunta: str) -> str:
    """Envía una pregunta en lenguaje natural al agente y devuelve su respuesta."""
    print(f"\n{'=' * 70}\nPREGUNTA: {pregunta}\n{'=' * 70}")
    respuesta = agent_executor.invoke({"input": pregunta})
    print(f"\nRESPUESTA: {respuesta['output']}")
    return respuesta["output"]


if __name__ == "__main__":
    # Preguntas de prueba, tal como pide el enunciado del caso de estudio
    preguntas_demo = [
        "¿Cuáles fueron las ventas netas totales de la marca La Favorita?",
        "¿Cuál es el cliente que más compró en kilos durante todo el periodo?",
        "¿Cuántos kilos se vendieron en total en el mes de julio de 2026?",
    ]

    for p in preguntas_demo:
        preguntar(p)


# ------------------------------------------------------------------
# ALTERNATIVAS (si más adelante quieres cambiar de proveedor)
# ------------------------------------------------------------------
#   Modelo local gratuito con Ollama (sin API key, sin costo):
#      pip install langchain-ollama
#      from langchain_ollama import ChatOllama
#      llm = ChatOllama(model="llama3.1", temperature=0)
#      (requiere tener Ollama instalado y el modelo descargado)
#
#   Anthropic / Claude:
#      pip install langchain-anthropic
#      from langchain_anthropic import ChatAnthropic
#      llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
#      (requiere variable de entorno ANTHROPIC_API_KEY)
#
# El resto del script (SQLDatabase, create_sql_agent, preguntar) funciona
# exactamente igual, sin cambios.
