import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# Configuracion de rutas
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "xlsx_limpios"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "pedidos_flat_clusterizado.csv"


# --------------------------------------------------------------------------
# Carga de datos fuente
# --------------------------------------------------------------------------
print("Cargando archivos fuente...")
fact = pd.read_excel(INPUT_DIR / "FactOrders.xlsx")
dim_tiempo = pd.read_excel(INPUT_DIR / "DimTiempo.xlsx")
dim_cliente = pd.read_excel(INPUT_DIR / "DimCliente.xlsx")
dim_restaurante = pd.read_excel(INPUT_DIR / "DimRestaurante.xlsx")
dim_metodopago = pd.read_excel(INPUT_DIR / "DimMetodoPago.xlsx")
dim_estadoorden = pd.read_excel(INPUT_DIR / "DimEstadoOrden.xlsx")

print(f"  FactOrders:     {len(fact):,} filas")
print(f"  DimTiempo:      {len(dim_tiempo):,} filas")
print(f"  DimCliente:     {len(dim_cliente):,} filas")
print(f"  DimRestaurante: {len(dim_restaurante):,} filas")
print(f"  DimMetodoPago:  {len(dim_metodopago):,} filas")
print(f"  DimEstadoOrden: {len(dim_estadoorden):,} filas")


# --------------------------------------------------------------------------
# Renombrado de columnas por dimension (nomenclatura plana en español)
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
    "CustomerHash": "hash_cliente",
    "CityName": "ciudad_cliente",
    "CountryName": "pais_cliente",
    "CountryCode": "codigo_pais_cliente",
    "SignupDate": "fecha_registro_cliente",
    "CustomerSegment": "segmento_cliente",
    "PreferredPaymentMethod": "metodo_pago_preferido_cliente",
    "IsActive": "cliente_activo",
})[["cliente_key", "cliente_id", "alias_cliente", "hash_cliente",
    "ciudad_cliente", "pais_cliente", "codigo_pais_cliente",
    "fecha_registro_cliente", "segmento_cliente",
    "metodo_pago_preferido_cliente", "cliente_activo"]]

dim_restaurante_r = dim_restaurante.rename(columns={
    "RestauranteKey": "restaurante_key",
    "RestaurantID": "restaurante_id",
    "RestaurantName": "nombre_restaurante",
    "CuisineName": "cocina_restaurante",
    "CityName": "ciudad_restaurante",
    "CountryName": "pais_restaurante",
    "CountryCode": "codigo_pais_restaurante",
    "AverageCostForTwo": "costo_promedio_dos_personas",
    "RatingTier": "nivel_rating_restaurante",
    "HasDelivery": "tiene_delivery",
    "HasDineIn": "tiene_comer_en_local",
    "HasWifi": "tiene_wifi",
    "AcceptsCreditCard": "acepta_tarjeta",
    "IsActive": "restaurante_activo",
})[["restaurante_key", "restaurante_id", "nombre_restaurante",
    "cocina_restaurante", "ciudad_restaurante", "pais_restaurante",
    "codigo_pais_restaurante", "costo_promedio_dos_personas",
    "nivel_rating_restaurante", "tiene_delivery", "tiene_comer_en_local",
    "tiene_wifi", "acepta_tarjeta", "restaurante_activo"]]

dim_metodopago_r = dim_metodopago.rename(columns={
    "MetodoPagoKey": "metodo_pago_key",
    "PaymentMethod": "metodo_pago",
})[["metodo_pago_key", "metodo_pago"]]

dim_estadoorden_r = dim_estadoorden.rename(columns={
    "EstadoKey": "estado_key",
    "OrderStatus": "estado_pedido",
    "EsCompletado": "estado_completado",
    "EsIncidencia": "estado_es_incidencia",
})[["estado_key", "estado_pedido", "estado_completado", "estado_es_incidencia"]]

fact_r = fact.rename(columns={
    "OrderKey": "pedido_key",
    "DateKey": "fecha_key",
    "ClienteKey": "cliente_key",
    "RestauranteKey": "restaurante_key",
    "MetodoPagoKey": "metodo_pago_key",
    "EstadoKey": "estado_key",
    "TotalAmount": "monto_total",
    "DeliveryFee": "tarifa_entrega",
    "DistanceKm": "distancia_km",
    "DeliveryTimeMinutes": "tiempo_entrega_minutos",
    "PeakHourMultiplier": "multiplicador_hora_pico",
    "CustomerRating": "calificacion_operativa",
    "WasDelayed": "hubo_demora",
})

fact_r["pedido_id"] = fact_r["pedido_key"]  # OrderKey == OrderID en este modelo


# --------------------------------------------------------------------------
# Aplanamiento (JOIN de las 5 dimensiones contra el hecho)
# --------------------------------------------------------------------------
print("\nAplanando (uniendo dimensiones al hecho)...")
plano = fact_r.merge(dim_tiempo_r, on="fecha_key", how="left")
plano = plano.merge(dim_cliente_r, on="cliente_key", how="left")
plano = plano.merge(dim_restaurante_r, on="restaurante_key", how="left")
plano = plano.merge(dim_metodopago_r, on="metodo_pago_key", how="left")
plano = plano.merge(dim_estadoorden_r, on="estado_key", how="left")

# --------------------------------------------------------------------------
# Orden final de columnas (identificadores primero, tal como se solicito)
# --------------------------------------------------------------------------
orden_columnas = [
    "pedido_key", "pedido_id",
    "fecha_key", "fecha", "dia", "nombre_dia", "es_fin_de_semana",
    "semana_anio", "mes", "nombre_mes", "trimestre", "anio", "anio_mes",
    "cliente_key", "cliente_id", "alias_cliente", "hash_cliente",
    "ciudad_cliente", "pais_cliente", "codigo_pais_cliente",
    "fecha_registro_cliente", "segmento_cliente",
    "metodo_pago_preferido_cliente", "cliente_activo",
    "restaurante_key", "restaurante_id", "nombre_restaurante",
    "cocina_restaurante", "ciudad_restaurante", "pais_restaurante",
    "codigo_pais_restaurante", "costo_promedio_dos_personas",
    "nivel_rating_restaurante", "tiene_delivery", "tiene_comer_en_local",
    "tiene_wifi", "acepta_tarjeta", "restaurante_activo",
    "metodo_pago_key", "metodo_pago",
    "estado_key", "estado_pedido", "estado_completado", "estado_es_incidencia",
    "monto_total", "tarifa_entrega", "distancia_km", "tiempo_entrega_minutos",
    "multiplicador_hora_pico", "calificacion_operativa", "hubo_demora",
]
plano = plano[orden_columnas]

print(f"  {len(plano):,} filas x {len(plano.columns)} columnas")


# --------------------------------------------------------------------------
# Validacion rapida de integridad tras el aplanamiento
# --------------------------------------------------------------------------
huerfanos_fecha = plano["fecha"].isnull().sum()
huerfanos_cliente = plano["cliente_id"].isnull().sum()
huerfanos_restaurante = plano["restaurante_id"].isnull().sum()

print("\nValidacion de integridad referencial:")
print(f"  Pedidos sin fecha resuelta:       {huerfanos_fecha}")
print(f"  Pedidos sin cliente resuelto:     {huerfanos_cliente}")
print(f"  Pedidos sin restaurante resuelto: {huerfanos_restaurante}")

if huerfanos_fecha or huerfanos_cliente or huerfanos_restaurante:
    print("  ADVERTENCIA: hay filas con dimensiones sin resolver. Revisar antes de usar el CSV.")
else:
    print("  OK: todos los pedidos resolvieron sus dimensiones correctamente.")


# --------------------------------------------------------------------------
# Exportacion a CSV
# --------------------------------------------------------------------------
plano.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nArchivo generado: {OUTPUT_CSV}")
