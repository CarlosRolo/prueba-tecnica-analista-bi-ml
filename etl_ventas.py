"""
ETL - Prueba Técnica Analista BI & ML
=======================================
Parte A: Ingeniería de Datos (ETL/ELT) y SQL

Objetivo:
    Leer las 3 fuentes crudas (Clientes, Material, Ventas), aplicar limpieza
    de datos, resolver nulos, estandarizar formatos y calcular la métrica de
    kilos, dejando un dataset analítico único listo para BI y ML.

Fuente: Data_Test_1.xlsx (hojas: Clientes, Material, Ventas)
Autor: Carlos [Analista BI & ML] - Prueba técnica La Fabril
"""

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 0. CONFIGURACIÓN
# ------------------------------------------------------------------
INPUT_FILE = Path("Data_Test_1.xlsx")
OUTPUT_FILE = Path("ventas_limpio.csv")


def log_step(mensaje: str) -> None:
    """Imprime un separador legible para trazar cada etapa del ETL."""
    print(f"\n{'=' * 70}\n{mensaje}\n{'=' * 70}")


# ------------------------------------------------------------------
# 1. EXTRACCIÓN (EXTRACT)
# ------------------------------------------------------------------
def extraer_datos(path: Path) -> dict[str, pd.DataFrame]:
    """Lee las 3 hojas del Excel origen tal como llegan, sin transformar."""
    log_step("1. EXTRACCIÓN: leyendo hojas crudas del Excel")
    xl = pd.ExcelFile(path)
    clientes = xl.parse("Clientes")
    material = xl.parse("Material")
    ventas = xl.parse("Ventas")

    print(f"Clientes : {clientes.shape[0]} filas, {clientes.shape[1]} columnas")
    print(f"Material : {material.shape[0]} filas, {material.shape[1]} columnas")
    print(f"Ventas   : {ventas.shape[0]} filas, {ventas.shape[1]} columnas")

    return {"clientes": clientes, "material": material, "ventas": ventas}


# ------------------------------------------------------------------
# 2. TRANSFORMACIÓN (TRANSFORM)
# ------------------------------------------------------------------
def limpiar_material(material: pd.DataFrame) -> pd.DataFrame:
    """
    Hallazgo de calidad de datos: el maestro Material trae 368 de 550 filas
    totalmente duplicadas (solo 182 materiales únicos reales). Se eliminan
    duplicados exactos para no inflar las ventas al hacer el JOIN.
    """
    log_step("2.1 Limpieza de Material")
    duplicados = material.duplicated().sum()
    print(f"Filas totalmente duplicadas detectadas: {duplicados}")

    material_limpio = material.drop_duplicates(subset="id_material").copy()
    print(f"Material tras deduplicar: {material_limpio.shape[0]} filas "
          f"(antes: {material.shape[0]})")

    # Estandarización de texto: quitar espacios extra, formato consistente
    cols_texto = material_limpio.select_dtypes(include=["object", "str"]).columns
    for col in cols_texto:
        material_limpio[col] = material_limpio[col].astype(str).str.strip()

    return material_limpio


def limpiar_clientes(clientes: pd.DataFrame) -> pd.DataFrame:
    """El maestro de Clientes no presenta nulos ni duplicados (validado)."""
    log_step("2.2 Limpieza de Clientes")
    print(f"Nulos por columna:\n{clientes.isnull().sum().to_string()}")
    print(f"Duplicados por id_cliente: {clientes['id_cliente'].duplicated().sum()}")

    clientes_limpio = clientes.copy()
    clientes_limpio["dsca_cliente"] = clientes_limpio["dsca_cliente"].str.strip()
    return clientes_limpio


def validar_integridad(ventas: pd.DataFrame, clientes: pd.DataFrame,
                        material: pd.DataFrame) -> None:
    """Verifica que no existan ventas con clientes o materiales huérfanos."""
    log_step("2.3 Validación de integridad referencial")
    huerfanos_cliente = (~ventas["id_cliente"].isin(clientes["id_cliente"])).sum()
    huerfanos_material = (~ventas["id_material"].isin(material["id_material"])).sum()
    print(f"Ventas con id_cliente sin maestro: {huerfanos_cliente}")
    print(f"Ventas con id_material sin maestro: {huerfanos_material}")

    if huerfanos_cliente or huerfanos_material:
        print("ADVERTENCIA: existen registros huérfanos; se conservarán con "
              "un LEFT JOIN, pero quedarán con nulos en las columnas del "
              "maestro correspondiente. Revisar antes de producción.")
    else:
        print("OK: integridad referencial validada, 0 registros huérfanos.")


def enriquecer_ventas(ventas: pd.DataFrame, clientes: pd.DataFrame,
                       material: pd.DataFrame) -> pd.DataFrame:
    """
    Hallazgo de calidad de datos: dsca_cliente, dsca_material y qty_kilos
    llegan 100% vacías en Ventas (500/500 filas). Se completan mediante
    LEFT JOIN contra los maestros ya limpios.
    """
    log_step("2.4 Enriquecimiento de Ventas (JOIN con maestros)")

    ventas_enriquecido = ventas.drop(columns=["dsca_cliente", "dsca_material"]).merge(
        clientes[["id_cliente", "dsca_cliente"]],
        on="id_cliente",
        how="left",
    ).merge(
        material[["id_material", "dsca_material", "dsca_marca", "dsca_categoria",
                   "dsca_linea", "factor_kg"]],
        on="id_material",
        how="left",
        suffixes=("_ventas", "_material"),
    )

    print(f"Filas tras JOIN: {ventas_enriquecido.shape[0]} "
          f"(se mantiene el conteo original de Ventas: {ventas.shape[0]})")

    nulos_post_join = ventas_enriquecido[["dsca_cliente", "dsca_material"]].isnull().sum()
    print(f"Nulos tras el JOIN:\n{nulos_post_join.to_string()}")

    return ventas_enriquecido


def calcular_metricas(ventas: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la métrica pedida en el enunciado: conversión a kilos.

    Importante: el factor_kg que trae la hoja Ventas viene fijo en 1 para
    todas las filas (no es útil). El factor_kg correcto es el del maestro
    Material, traído en el JOIN anterior como 'factor_kg_material'.
        qty_kilos = qty_entregada * factor_kg_material

    También se calcula el margen, útil luego para los KPIs de BI.
    """
    log_step("2.5 Cálculo de métricas derivadas")

    df = ventas.copy()
    df["qty_kilos"] = df["qty_entregada"] * df["factor_kg_material"]
    df["margen"] = df["venta_neta"] - df["costos"]
    df["margen_pct"] = (df["margen"] / df["venta_neta"]).round(4)

    # Columnas de fecha estandarizadas, útiles para BI y ML
    df["anio"] = df["fecha_factura"].dt.year
    df["mes"] = df["fecha_factura"].dt.month
    df["anio_mes"] = df["fecha_factura"].dt.to_period("M").astype(str)

    print("Métricas calculadas: qty_kilos, margen, margen_pct, anio, mes, anio_mes")
    print(df[["qty_entregada", "factor_kg_material", "qty_kilos"]].head(3).to_string())

    return df


def seleccionar_columnas_finales(df: pd.DataFrame) -> pd.DataFrame:
    """Deja un dataset analítico limpio, ordenado y con nombres consistentes."""
    log_step("2.6 Selección y orden de columnas finales")

    columnas_finales = [
        "fecha_factura", "anio", "mes", "anio_mes",
        "nro_factura", "oid",
        "id_cliente", "dsca_cliente",
        "id_material", "dsca_material", "dsca_marca", "dsca_categoria", "dsca_linea",
        "qty_entregada", "factor_kg_material", "qty_kilos",
        "costos", "venta_neta", "margen", "margen_pct",
    ]
    df_final = df[columnas_finales].rename(columns={"factor_kg_material": "factor_kg"})
    return df_final


# ------------------------------------------------------------------
# 3. CARGA (LOAD)
# ------------------------------------------------------------------
def cargar_datos(df: pd.DataFrame, path: Path) -> None:
    log_step("3. CARGA: guardando dataset analítico final")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Archivo generado: {path.resolve()}")
    print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")


# ------------------------------------------------------------------
# 4. RESUMEN DE VALIDACIÓN FINAL
# ------------------------------------------------------------------
def resumen_calidad(df: pd.DataFrame) -> None:
    log_step("4. Resumen de calidad del dataset final")
    print(f"Nulos por columna:\n{df.isnull().sum().to_string()}")
    print(f"\nRango de fechas: {df['fecha_factura'].min()} -> {df['fecha_factura'].max()}")
    print(f"Marcas distintas: {df['dsca_marca'].nunique()}")
    print(f"Clientes distintos: {df['id_cliente'].nunique()}")
    print(f"Venta neta total: {df['venta_neta'].sum():,.2f}")
    print(f"Kilos totales: {df['qty_kilos'].sum():,.2f}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main() -> None:
    datos = extraer_datos(INPUT_FILE)

    material_limpio = limpiar_material(datos["material"])
    clientes_limpio = limpiar_clientes(datos["clientes"])
    validar_integridad(datos["ventas"], clientes_limpio, material_limpio)

    ventas_enriquecido = enriquecer_ventas(datos["ventas"], clientes_limpio, material_limpio)
    ventas_con_metricas = calcular_metricas(ventas_enriquecido)
    ventas_final = seleccionar_columnas_finales(ventas_con_metricas)

    cargar_datos(ventas_final, OUTPUT_FILE)
    resumen_calidad(ventas_final)


if __name__ == "__main__":
    main()
