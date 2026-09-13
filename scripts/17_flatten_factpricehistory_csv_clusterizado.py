

import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# Configuracion de rutas
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "xlsx_limpios"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "historial_precios_flat_clusterizado.csv"


# --------------------------------------------------------------------------
# Carga de datos fuente
# --------------------------------------------------------------------------
print("Cargando archivos fuente...")
fact = pd.read_excel(INPUT_DIR / "FactPriceHistory.xlsx")
dim_tiempo = pd.read_excel(INPUT_DIR / "DimTiempo.xlsx")
dim_menuitem = pd.read_excel(INPUT_DIR / "DimMenuItem.xlsx")
dim_restaurante = pd.read_excel(INPUT_DIR / "DimRestaurante.xlsx")

print(f"  FactPriceHistory: {len(fact):,} filas")
print(f"  DimTiempo:        {len(dim_tiempo):,} filas")
print(f"  DimMenuItem:      {len(dim_menuitem):,} filas")
print(f"  DimRestaurante:   {len(dim_restaurante):,} filas")


# --------------------------------------------------------------------------
# Correccion de calidad de datos: recalculo de PriceChangePercent
# --------------------------------------------------------------------------
fact = fact.copy()
recalculado = (
    (fact["NewPrice"] - fact["PreviousPrice"]) / fact["PreviousPrice"] * 100
).round(2)

inconsistentes = (recalculado != fact["PriceChangePercent"].round(2)).sum()
print(f"\nFilas con PriceChangePercent inconsistente detectadas: {inconsistentes}")

fact["PriceChangePercent"] = recalculado
print(f"Filas inconsistentes tras la correccion: 0 (recalculado para el 100% de las filas)")


# --------------------------------------------------------------------------
# Renombrado de columnas por dimension
# --------------------------------------------------------------------------
dim_tiempo_r = dim_tiempo.rename(columns={
    "DateKey": "fecha_key",
    "FullDate": "fecha",
    "MonthNumber": "mes",
    "MonthName": "nombre_mes",
    "Quarter": "trimestre",
    "Year": "anio",
    "YearMonth": "anio_mes",
})[["fecha_key", "fecha", "mes", "nombre_mes", "trimestre", "anio", "anio_mes"]]

dim_menuitem_r = dim_menuitem.rename(columns={
    "MenuItemKey": "item_menu_key",
    "MenuID": "item_menu_id",
    "ItemName": "nombre_item",
    "Category": "categoria_item",
    "CuisineName": "cocina_item",
    "IsAvailable": "disponible_item",
})[["item_menu_key", "item_menu_id", "nombre_item", "categoria_item",
    "cocina_item", "disponible_item"]]

dim_restaurante_r = dim_restaurante.rename(columns={
    "RestauranteKey": "restaurante_key",
    "RestaurantID": "restaurante_id",
    "RestaurantName": "nombre_restaurante",
    "CuisineName": "cocina_restaurante",
    "CityName": "ciudad_restaurante",
    "CountryName": "pais_restaurante",
})[["restaurante_key", "restaurante_id", "nombre_restaurante",
    "cocina_restaurante", "ciudad_restaurante", "pais_restaurante"]]

fact_r = fact.rename(columns={
    "PriceChangeKey": "cambio_precio_key",
    "DateKey": "fecha_key",
    "MenuItemKey": "item_menu_key",
    "RestauranteKey": "restaurante_key",
    "PreviousPrice": "precio_anterior",
    "NewPrice": "precio_nuevo",
    "PriceChangeAmount": "monto_cambio",
    "PriceChangePercent": "porcentaje_cambio",
})


# --------------------------------------------------------------------------
# Aplanamiento
# --------------------------------------------------------------------------
print("\nAplanando (uniendo dimensiones al hecho)...")
plano = fact_r.merge(dim_tiempo_r, on="fecha_key", how="left")
plano = plano.merge(dim_menuitem_r, on="item_menu_key", how="left")
plano = plano.merge(dim_restaurante_r, on="restaurante_key", how="left")

orden_columnas = [
    "cambio_precio_key",
    "fecha_key", "fecha", "mes", "nombre_mes", "trimestre", "anio", "anio_mes",
    "item_menu_key", "item_menu_id", "nombre_item", "categoria_item",
    "cocina_item", "disponible_item",
    "restaurante_key", "restaurante_id", "nombre_restaurante",
    "cocina_restaurante", "ciudad_restaurante", "pais_restaurante",
    "precio_anterior", "precio_nuevo", "monto_cambio", "porcentaje_cambio",
]
plano = plano[orden_columnas]

print(f"  {len(plano):,} filas x {len(plano.columns)} columnas")


# --------------------------------------------------------------------------
# Validacion rapida
# --------------------------------------------------------------------------
huerfanos_fecha = plano["fecha"].isnull().sum()
huerfanos_item = plano["item_menu_id"].isnull().sum()
huerfanos_restaurante = plano["restaurante_id"].isnull().sum()

print("\nValidacion de integridad referencial:")
print(f"  Eventos sin fecha resuelta:        {huerfanos_fecha}")
print(f"  Eventos sin item de menu resuelto: {huerfanos_item}")
print(f"  Eventos sin restaurante resuelto:  {huerfanos_restaurante}")

if huerfanos_fecha or huerfanos_item or huerfanos_restaurante:
    print("  ADVERTENCIA: revisar los casos listados arriba antes de usar el CSV.")
else:
    print("  OK: todos los eventos de precio resolvieron sus dimensiones correctamente.")


# --------------------------------------------------------------------------
# Exportacion a CSV
# --------------------------------------------------------------------------
plano.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nArchivo generado: {OUTPUT_CSV}")
