"""
Machine Learning - Prueba Técnica Analista BI & ML
=====================================================
Parte C: Modelado Predictivo y MLOps

Objetivo:
    Entrenar un modelo que prediga el volumen de ventas (qty_kilos) de la
    semana siguiente, a partir de la serie histórica ya limpia en la Parte A.
    No se busca precisión perfecta: se documenta la metodología completa
    (features, separación temporal train/test, métricas de evaluación).

Fuente: ventas_limpio.csv (generado por etl_ventas.py, Parte A)
Autor: Carlos [Analista BI & ML] - Prueba técnica La Fabril
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------
INPUT_FILE = Path("ventas_limpio.csv")
TEST_SIZE_WEEKS = 6  # últimas 6 semanas como conjunto de prueba
RANDOM_STATE = 42


def log_step(mensaje: str) -> None:
    print(f"\n{'=' * 70}\n{mensaje}\n{'=' * 70}")


# ------------------------------------------------------------------
# 1. CARGA Y AGREGACIÓN SEMANAL
# ------------------------------------------------------------------
def cargar_y_agregar(path: Path) -> pd.DataFrame:
    """
    Carga el dataset limpio de la Parte A y lo agrega a nivel semanal.
    Se elige semana (y no día) porque a nivel diario la serie es demasiado
    ruidosa para un dataset de 500 filas / ~7 meses; la agregación semanal
    da una señal más estable para el modelo.
    """
    log_step("1. Carga y agregación semanal de la serie de ventas")

    df = pd.read_csv(path, sep=";", decimal=",")
    df["fecha_factura"] = pd.to_datetime(df["fecha_factura"])

    semanal = (
        df.set_index("fecha_factura")
        .resample("W-MON", label="left", closed="left")
        .agg(qty_kilos=("qty_kilos", "sum"),
             venta_neta=("venta_neta", "sum"),
             nro_transacciones=("nro_factura", "count"))
        .reset_index()
        .rename(columns={"fecha_factura": "semana"})
    )

    print(f"Serie semanal generada: {semanal.shape[0]} semanas")
    print(semanal.head())

    return semanal


# ------------------------------------------------------------------
# 2. INGENIERÍA DE VARIABLES (FEATURE ENGINEERING)
# ------------------------------------------------------------------
def crear_features(semanal: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables de tipo 'lag' (rezago) y promedio móvil, estándar en
    modelado de series de tiempo: el objetivo es predecir qty_kilos de la
    semana actual usando información de semanas anteriores (nunca del
    futuro, para evitar fuga de datos / data leakage).
    """
    log_step("2. Ingeniería de variables (lags y rolling mean)")

    df = semanal.copy()
    df["lag_1"] = df["qty_kilos"].shift(1)
    df["lag_2"] = df["qty_kilos"].shift(2)
    df["media_movil_3"] = df["qty_kilos"].shift(1).rolling(window=3).mean()
    df["semana_del_anio"] = df["semana"].dt.isocalendar().week.astype(int)

    filas_antes = df.shape[0]
    df = df.dropna().reset_index(drop=True)
    print(f"Filas descartadas por no tener suficiente historia (lags/rolling): "
          f"{filas_antes - df.shape[0]}")
    print(f"Dataset final para modelar: {df.shape[0]} semanas")

    return df


# ------------------------------------------------------------------
# 3. SEPARACIÓN TRAIN / TEST (TEMPORAL, NO ALEATORIA)
# ------------------------------------------------------------------
def separar_train_test(df: pd.DataFrame, semanas_test: int):
    """
    Importante: en series de tiempo NUNCA se separa train/test de forma
    aleatoria (eso filtraría información del futuro hacia el pasado). Se
    respeta el orden cronológico: las últimas N semanas son el test.
    """
    log_step("3. Separación temporal train / test")

    features = ["lag_1", "lag_2", "media_movil_3", "semana_del_anio"]
    target = "qty_kilos"

    train = df.iloc[:-semanas_test]
    test = df.iloc[-semanas_test:]

    print(f"Train: {train.shape[0]} semanas ({train['semana'].min().date()} "
          f"a {train['semana'].max().date()})")
    print(f"Test : {test.shape[0]} semanas ({test['semana'].min().date()} "
          f"a {test['semana'].max().date()})")

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    return X_train, X_test, y_train, y_test, test["semana"]


# ------------------------------------------------------------------
# 4. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ------------------------------------------------------------------
def entrenar_y_evaluar(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Se entrenan 2 modelos simples (Regresión Lineal como baseline y
    Random Forest como modelo no lineal) y se comparan con MAE y RMSE.
    No se busca el mejor accuracy posible, sino demostrar la metodología:
    baseline interpretable + modelo más flexible + métricas comparables.
    """
    log_step("4. Entrenamiento y evaluación de modelos")

    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=4, random_state=RANDOM_STATE
        ),
    }

    resultados = []
    predicciones = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        predicciones[nombre] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        resultados.append({"modelo": nombre, "MAE": round(mae, 2), "RMSE": round(rmse, 2)})
        print(f"{nombre:20s} -> MAE: {mae:8.2f} kg | RMSE: {rmse:8.2f} kg")

    return pd.DataFrame(resultados), predicciones


# ------------------------------------------------------------------
# 5. TABLA COMPARATIVA REAL VS PREDICHO
# ------------------------------------------------------------------
def tabla_comparativa(semanas_test, y_test, predicciones: dict) -> pd.DataFrame:
    log_step("5. Comparación real vs. predicho (conjunto de prueba)")

    comparativa = pd.DataFrame({
        "semana": semanas_test.dt.date.values,
        "real_kg": y_test.values.round(2),
    })
    for nombre, y_pred in predicciones.items():
        comparativa[f"pred_{nombre.lower().replace(' ', '_')}"] = np.round(y_pred, 2)

    print(comparativa.to_string(index=False))
    return comparativa


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    semanal = cargar_y_agregar(INPUT_FILE)
    df_features = crear_features(semanal)
    X_train, X_test, y_train, y_test, semanas_test = separar_train_test(
        df_features, TEST_SIZE_WEEKS
    )
    resultados, predicciones = entrenar_y_evaluar(X_train, X_test, y_train, y_test)
    comparativa = tabla_comparativa(semanas_test, y_test, predicciones)

    log_step("RESUMEN FINAL")
    print(resultados.to_string(index=False))

    resultados.to_csv("ml_resultados_metricas.csv", index=False)
    comparativa.to_csv("ml_comparativo_real_vs_predicho.csv", index=False)
    print("\nArchivos generados: ml_resultados_metricas.csv, "
          "ml_comparativo_real_vs_predicho.csv")


if __name__ == "__main__":
    main()
