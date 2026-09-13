import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# Configuracion de rutas
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "xlsx_limpios"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "resenas_flat_clusterizado.csv"

RATING_MIN = 1.0
RATING_MAX = 5.0


# --------------------------------------------------------------------------
# Carga de datos fuente
# --------------------------------------------------------------------------
print("Cargando archivos fuente...")
fact = pd.read_excel(INPUT_DIR / "FactReviews.xlsx")
dim_tiempo = pd.read_excel(INPUT_DIR / "DimTiempo.xlsx")
dim_cliente = pd.read_excel(INPUT_DIR / "DimCliente.xlsx")
dim_restaurante = pd.read_excel(INPUT_DIR / "DimRestaurante.xlsx")

print(f"  FactReviews:    {len(fact):,} filas")
print(f"  DimTiempo:      {len(dim_tiempo):,} filas")
print(f"  DimCliente:     {len(dim_cliente):,} filas")
print(f"  DimRestaurante: {len(dim_restaurante):,} filas")


# --------------------------------------------------------------------------
# Correccion de calidad de datos: recorte de Rating fuera de rango
# --------------------------------------------------------------------------
fuera_de_rango = ((fact["Rating"] < RATING_MIN) | (fact["Rating"] > RATING_MAX)).sum()
print(f"\nFilas con Rating fuera de [{RATING_MIN}, {RATING_MAX}] detectadas: {fuera_de_rango}")

fact = fact.copy()
fact["Rating"] = fact["Rating"].clip(lower=RATING_MIN, upper=RATING_MAX)

fuera_de_rango_post = ((fact["Rating"] < RATING_MIN) | (fact["Rating"] > RATING_MAX)).sum()
print(f"Filas con Rating fuera de rango tras la correccion: {fuera_de_rango_post} (debe ser 0)")


# --------------------------------------------------------------------------
# Renombrado de columnas por dimension
# --------------------------------------------------------------------------
dim_tiempo_r = dim_tiempo.rename(columns={
    "DateKey": "fecha_key",
    "FullDate": "fecha",
    "DayOfMonth": "dia",
    "DayName": "nombre_dia",
    "IsWeekend": "es_fin_de_semana",
    "WeekOfYear": "semana_anio",
    "MonthNumber": "mes",
    "MonthName": "nombre_mes",
    "Quarter": "trimestre",
    "Year": "anio",
    "YearMonth": "anio_mes",
})[["fecha_key", "fecha", "dia", "nombre_dia", "es_fin_de_semana",
    "semana_anio", "mes", "nombre_mes", "trimestre", "anio", "anio_mes"]]

dim_cliente_r = dim_cliente.rename(columns={
    "ClienteKey": "cliente_key",
    "CustomerID": "cliente_id",
    "CustomerAlias": "alias_cliente",
    "CityName": "ciudad_cliente",
    "CountryName": "pais_cliente",
    "CustomerSegment": "segmento_cliente",
})[["cliente_key", "cliente_id", "alias_cliente", "ciudad_cliente",
    "pais_cliente", "segmento_cliente"]]

dim_restaurante_r = dim_restaurante.rename(columns={
    "RestauranteKey": "restaurante_key",
    "RestaurantID": "restaurante_id",
    "RestaurantName": "nombre_restaurante",
    "CuisineName": "cocina_restaurante",
    "CityName": "ciudad_restaurante",
    "CountryName": "pais_restaurante",
    "RatingTier": "nivel_rating_restaurante",
})[["restaurante_key", "restaurante_id", "nombre_restaurante",
    "cocina_restaurante", "ciudad_restaurante", "pais_restaurante",
    "nivel_rating_restaurante"]]

fact_r = fact.rename(columns={
    "ReviewKey": "resena_key",
    "OrderID": "pedido_id",
    "DateKey": "fecha_key",
    "ClienteKey": "cliente_key",
    "RestauranteKey": "restaurante_key",
    "Rating": "calificacion",
    "HasReviewText": "tiene_comentario",
    "ReviewText": "comentario",
})


# --------------------------------------------------------------------------
# Aplanamiento
# --------------------------------------------------------------------------
print("\nAplanando (uniendo dimensiones al hecho)...")
plano = fact_r.merge(dim_tiempo_r, on="fecha_key", how="left")
plano = plano.merge(dim_cliente_r, on="cliente_key", how="left")
plano = plano.merge(dim_restaurante_r, on="restaurante_key", how="left")

orden_columnas = [
    "resena_key", "pedido_id",
    "fecha_key", "fecha", "dia", "nombre_dia", "es_fin_de_semana",
    "semana_anio", "mes", "nombre_mes", "trimestre", "anio", "anio_mes",
    "cliente_key", "cliente_id", "alias_cliente", "ciudad_cliente",
    "pais_cliente", "segmento_cliente",
    "restaurante_key", "restaurante_id", "nombre_restaurante",
    "cocina_restaurante", "ciudad_restaurante", "pais_restaurante",
    "nivel_rating_restaurante",
    "calificacion", "tiene_comentario", "comentario",
]
plano = plano[orden_columnas]

print(f"  {len(plano):,} filas x {len(plano.columns)} columnas")


# --------------------------------------------------------------------------
# Validacion rapida
# --------------------------------------------------------------------------
huerfanos_fecha = plano["fecha"].isnull().sum()
huerfanos_cliente = plano["cliente_id"].isnull().sum()
huerfanos_restaurante = plano["restaurante_id"].isnull().sum()
inconsistentes = (
    ((plano["tiene_comentario"] == 1) & (plano["comentario"].isnull()))
    | ((plano["tiene_comentario"] == 0) & (plano["comentario"].notnull()))
).sum()

print("\nValidacion de integridad referencial y reglas de negocio:")
print(f"  Resenas sin fecha resuelta:                {huerfanos_fecha}")
print(f"  Resenas sin cliente resuelto:               {huerfanos_cliente}")
print(f"  Resenas sin restaurante resuelto:           {huerfanos_restaurante}")
print(f"  Inconsistencias tiene_comentario/comentario: {inconsistentes}")

if huerfanos_fecha or huerfanos_cliente or huerfanos_restaurante or inconsistentes:
    print("  ADVERTENCIA: revisar los casos listados arriba antes de usar el CSV.")
else:
    print("  OK: todas las resenas resolvieron sus dimensiones y pasan la regla cruzada.")


# --------------------------------------------------------------------------
# Exportacion a CSV
# --------------------------------------------------------------------------
plano.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nArchivo generado: {OUTPUT_CSV}")
