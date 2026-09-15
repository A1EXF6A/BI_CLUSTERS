import pandas as pd
from pathlib import Path

from cluster_utils import (
    limpiar_columna_por_cluster,
    deduplicar_catalogo_por_cluster,
    clusterizar_kmeans_df,
    reiniciar_reportes,
    imputar_texto_por_grupo,
    imputar_numerico_por_grupo,
    graficar_top_clusters,
    graficar_reduccion_unicos,
    graficar_imputacion_numerica,
    graficar_codo_silhouette,
    graficar_clusters_kmeans_2d,
    graficar_distribucion_clusters,
)

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "datos_sucios"
OUTPUT_DIR = BASE_DIR / "xlsx_limpios"
GRAPHS_DIR = BASE_DIR / "graficos_clusters"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("Limpiando DimCliente...")
print("=" * 60)
reiniciar_reportes()
dim_cliente = pd.read_excel(INPUT_DIR / "DimCliente.xlsx")
dim_cliente = limpiar_columna_por_cluster(dim_cliente, "CountryName", tabla="DimCliente")
dim_cliente = limpiar_columna_por_cluster(dim_cliente, "CityName", tabla="DimCliente")
dim_cliente = imputar_texto_por_grupo(dim_cliente, "CountryName", "CountryCode", tabla="DimCliente")
dim_cliente.to_excel(OUTPUT_DIR / "DimCliente.xlsx", index=False)
graficar_top_clusters(GRAPHS_DIR)
graficar_reduccion_unicos(GRAPHS_DIR)

print("\n" + "=" * 60)
print("Limpiando DimRestaurante...")
print("=" * 60)
reiniciar_reportes()
dim_restaurante = pd.read_excel(INPUT_DIR / "DimRestaurante.xlsx")
dim_restaurante = limpiar_columna_por_cluster(dim_restaurante, "CountryName", tabla="DimRestaurante")
dim_restaurante = limpiar_columna_por_cluster(dim_restaurante, "CityName", tabla="DimRestaurante")
dim_restaurante = limpiar_columna_por_cluster(dim_restaurante, "CuisineName", tabla="DimRestaurante")
dim_restaurante = limpiar_columna_por_cluster(dim_restaurante, "RatingTier", tabla="DimRestaurante")
dim_restaurante = imputar_texto_por_grupo(dim_restaurante, "CountryName", "CountryCode", tabla="DimRestaurante")
dim_restaurante.to_excel(OUTPUT_DIR / "DimRestaurante.xlsx", index=False)
graficar_top_clusters(GRAPHS_DIR)
graficar_reduccion_unicos(GRAPHS_DIR)

print("\n" + "=" * 60)
print("Limpiando DimMenuItem...")
print("=" * 60)
reiniciar_reportes()
dim_menuitem = pd.read_excel(INPUT_DIR / "DimMenuItem.xlsx")
dim_menuitem = limpiar_columna_por_cluster(dim_menuitem, "Category", tabla="DimMenuItem")
dim_menuitem = limpiar_columna_por_cluster(dim_menuitem, "CuisineName", tabla="DimMenuItem")
dim_menuitem.to_excel(OUTPUT_DIR / "DimMenuItem.xlsx", index=False)
graficar_top_clusters(GRAPHS_DIR)
graficar_reduccion_unicos(GRAPHS_DIR)

print("\n" + "=" * 60)
print("Limpiando DimMetodoPago (deduplicando catalogo)...")
print("=" * 60)
dim_metodopago = pd.read_excel(INPUT_DIR / "DimMetodoPago.xlsx")
dim_metodopago = deduplicar_catalogo_por_cluster(dim_metodopago, "PaymentMethod", "MetodoPagoKey", tabla="DimMetodoPago")
dim_metodopago.to_excel(OUTPUT_DIR / "DimMetodoPago.xlsx", index=False)

print("\n" + "=" * 60)
print("Limpiando DimEstadoOrden (deduplicando catalogo)...")
print("=" * 60)
dim_estadoorden = pd.read_excel(INPUT_DIR / "DimEstadoOrden.xlsx")
dim_estadoorden = deduplicar_catalogo_por_cluster(dim_estadoorden, "OrderStatus", "EstadoKey", tabla="DimEstadoOrden")
dim_estadoorden.to_excel(OUTPUT_DIR / "DimEstadoOrden.xlsx", index=False)

print("\n" + "=" * 60)
print("Limpiando FactOrders (imputacion numerica + K-means)...")
print("=" * 60)
reiniciar_reportes()
fact_orders = pd.read_excel(INPUT_DIR / "FactOrders.xlsx")
fact_orders = imputar_numerico_por_grupo(fact_orders, "DeliveryTimeMinutes", "RestauranteKey", tabla="FactOrders")
fact_orders = imputar_numerico_por_grupo(fact_orders, "CustomerRating", "RestauranteKey", tabla="FactOrders")
fact_orders.to_excel(OUTPUT_DIR / "FactOrders.xlsx", index=False)
graficar_imputacion_numerica(GRAPHS_DIR)

reiniciar_reportes()
fact_orders = clusterizar_kmeans_df(
    fact_orders,
    columnas=["TotalAmount", "DeliveryFee", "DistanceKm", "DeliveryTimeMinutes", "CustomerRating"],
    tabla="FactOrders",
    k_min=2, k_max=6,
)
fact_orders.to_excel(OUTPUT_DIR / "FactOrders.xlsx", index=False)
graficar_codo_silhouette(GRAPHS_DIR)
graficar_clusters_kmeans_2d(GRAPHS_DIR)
graficar_distribucion_clusters(GRAPHS_DIR)

print("\n" + "=" * 60)
print("Procesando FactReviews (K-means)...")
print("=" * 60)
reiniciar_reportes()
fact_reviews = pd.read_excel(INPUT_DIR / "FactReviews.xlsx")
fact_reviews = clusterizar_kmeans_df(
    fact_reviews,
    columnas=["Rating", "HasReviewText"],
    tabla="FactReviews",
    k_min=2, k_max=5,
)
fact_reviews.to_excel(OUTPUT_DIR / "FactReviews.xlsx", index=False)
graficar_codo_silhouette(GRAPHS_DIR)
graficar_distribucion_clusters(GRAPHS_DIR)

print("\n" + "=" * 60)
print("Procesando FactPriceHistory (K-means)...")
print("=" * 60)
reiniciar_reportes()
fact_pricehistory = pd.read_excel(INPUT_DIR / "FactPriceHistory.xlsx")
fact_pricehistory = clusterizar_kmeans_df(
    fact_pricehistory,
    columnas=["PreviousPrice", "NewPrice", "PriceChangePercent"],
    tabla="FactPriceHistory",
    k_min=2, k_max=6,
)
fact_pricehistory.to_excel(OUTPUT_DIR / "FactPriceHistory.xlsx", index=False)
graficar_codo_silhouette(GRAPHS_DIR)
graficar_clusters_kmeans_2d(GRAPHS_DIR)
graficar_distribucion_clusters(GRAPHS_DIR)

print("\n" + "=" * 60)
print("Copiando DimTiempo (sin cambios)...")
print("=" * 60)
pd.read_excel(INPUT_DIR / "DimTiempo.xlsx").to_excel(OUTPUT_DIR / "DimTiempo.xlsx", index=False)

print("\n" + "=" * 60)
print("PROCESO COMPLETADO")
print("=" * 60)
print(f"Archivos limpios en: {OUTPUT_DIR}")
print(f"Graficos de auditoria en: {GRAPHS_DIR}")
print(f"Archivos generados: {len(list(OUTPUT_DIR.glob('*.xlsx')))} xlsx")
print(f"Graficos generados: {len(list(GRAPHS_DIR.glob('*.png')))} png")
