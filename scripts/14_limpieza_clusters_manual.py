import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "datos_sucios"
OUTPUT_DIR = BASE_DIR / "xlsx_limpios"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TECNICA 1: CLUSTERING MANUAL POR FINGERPRINT (colision de claves)
# ============================================================================
def generar_fingerprint(texto: str) -> str:
    """
    Genera la 'huella' de un texto: minusculas, sin puntuacion, sin
    espacios sobrantes, palabras unicas ordenadas alfabeticamente.
    Dos textos que representan el mismo valor real deberian producir
    la MISMA huella sin importar mayusculas, espacios o el orden de
    escritura de las palabras.
    """
    if not isinstance(texto, str):
        return texto
    t = texto.strip().lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)     # quitar puntuacion
    t = re.sub(r"\s+", " ", t).strip()    # colapsar espacios multiples
    tokens = sorted(set(t.split()))
    return " ".join(tokens)


def construir_clusters(valores: pd.Series) -> dict:
    """
    Agrupa manualmente los valores no nulos de una columna segun su
    fingerprint. Devuelve un diccionario:
        { fingerprint: Counter({valor_original: frecuencia, ...}) }
    Esto ES el clustering: cada fingerprint es un cluster que contiene
    todas las variantes de escritura observadas para ese valor real.
    """
    clusters = {}
    for valor in valores.dropna():
        huella = generar_fingerprint(valor)
        if huella not in clusters:
            clusters[huella] = Counter()
        clusters[huella][valor] += 1
    return clusters


def elegir_valor_canonico(contador_variantes: Counter) -> str:
    """
    Dentro de un cluster, elige el valor canonico: la variante mas
    frecuente. En caso de empate, se prefiere una variante en formato
    'mixto' (con mayusculas Y minusculas, ej. 'CreditCard' o 'India')
    por sobre TODO-MAYUSCULAS o todo-minusculas, ya que ese es
    tipicamente el formato humano original antes de la captura
    defectuosa. Si sigue habiendo empate, se usa orden alfabetico
    (desempate deterministico y reproducible).
    """
    variantes_mas_frecuentes = contador_variantes.most_common()
    max_frecuencia = variantes_mas_frecuentes[0][1]
    empatadas = [v for v, f in variantes_mas_frecuentes if f == max_frecuencia]

    if len(empatadas) == 1:
        return empatadas[0]

    # Preferir variantes con mayusculas Y minusculas mezcladas (ni todo
    # mayuscula ni todo minuscula), sin depender de que tengan espacios
    # entre palabras (cubre tanto 'United States' como 'CreditCard').
    con_formato_mixto = [
        v for v in empatadas
        if not v.strip().isupper() and not v.strip().islower()
    ]
    if len(con_formato_mixto) == 1:
        return con_formato_mixto[0]
    if len(con_formato_mixto) > 1:
        empatadas = con_formato_mixto  # sigue desempatando solo entre estas

    return sorted(empatadas)[0]


def limpiar_columna_por_cluster(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Aplica el clustering manual por fingerprint a una columna de texto:
    agrupa variantes, elige un valor canonico por cluster, y reemplaza
    cada valor original por el canonico de su cluster. Imprime un
    reporte de auditoria (cuantos clusters, cuantas filas cambiaron).
    """
    df = df.copy()
    clusters = construir_clusters(df[columna])

    mapa_canonico = {}
    clusters_con_variantes = 0
    for huella, variantes in clusters.items():
        canonico = elegir_valor_canonico(variantes)
        if len(variantes) > 1:
            clusters_con_variantes += 1
        for variante_original in variantes:
            mapa_canonico[variante_original] = canonico

    filas_afectadas = df[columna].map(
        lambda v: v in mapa_canonico and mapa_canonico[v] != v
    ).sum()

    df[columna] = df[columna].map(lambda v: mapa_canonico.get(v, v))

    print(f"    {columna}: {len(clusters)} clusters encontrados "
          f"({clusters_con_variantes} con mas de una variante), "
          f"{filas_afectadas} filas normalizadas")

    return df


# ============================================================================
# TECNICA 2: IMPUTACION MANUAL POR AGRUPAMIENTO (cluster -> valor tipico)
# ============================================================================
def imputar_texto_por_grupo(df: pd.DataFrame, columna_objetivo: str, columna_grupo: str) -> pd.DataFrame:
    """
    Completa valores vacios de 'columna_objetivo' usando el valor MAS
    FRECUENTE observado en su mismo grupo (definido por 'columna_grupo'),
    entre las filas donde 'columna_objetivo' SI esta presente. Cada valor
    distinto de columna_grupo actua como un cluster de filas relacionadas.
    """
    df = df.copy()

    moda_por_grupo = (
        df.dropna(subset=[columna_objetivo])
        .groupby(columna_grupo)[columna_objetivo]
        .agg(lambda serie: Counter(serie).most_common(1)[0][0])
        .to_dict()
    )

    antes = df[columna_objetivo].isna().sum()
    mascara_vacio = df[columna_objetivo].isna()
    df.loc[mascara_vacio, columna_objetivo] = df.loc[mascara_vacio, columna_grupo].map(moda_por_grupo)
    despues = df[columna_objetivo].isna().sum()

    print(f"    {columna_objetivo}: {antes} valores vacios -> {despues} tras imputar "
          f"por grupo de '{columna_grupo}' ({antes - despues} completados)")

    return df


def imputar_numerico_por_grupo(df: pd.DataFrame, columna_objetivo: str, columna_grupo: str) -> pd.DataFrame:
    """
    Completa valores numericos vacios usando el PROMEDIO del mismo grupo
    (ej. el mismo restaurante). Si un grupo no tiene ningun valor valido
    para calcular su propio promedio, se usa el promedio general como
    respaldo (para no dejar nulos residuales).
    """
    df = df.copy()

    promedio_por_grupo = df.groupby(columna_grupo)[columna_objetivo].transform("mean")
    promedio_general = df[columna_objetivo].mean()

    antes = df[columna_objetivo].isna().sum()
    df[columna_objetivo] = df[columna_objetivo].fillna(promedio_por_grupo)
    df[columna_objetivo] = df[columna_objetivo].fillna(promedio_general)
    df[columna_objetivo] = df[columna_objetivo].round(2)
    despues = df[columna_objetivo].isna().sum()

    print(f"    {columna_objetivo}: {antes} valores vacios -> {despues} tras imputar "
          f"por promedio de '{columna_grupo}' ({antes - despues} completados)")

    return df


def deduplicar_catalogo_por_cluster(df: pd.DataFrame, columna_texto: str, columna_clave: str) -> pd.DataFrame:
    """
    Para catalogos pequeños (DimMetodoPago, DimEstadoOrden) donde el
    'ensuciado' agrego FILAS duplicadas con variantes de formato: se
    agrupan las filas por fingerprint del texto, se conserva solo la
    fila cuyo texto sea el valor canonico del cluster, y se descartan
    las filas duplicadas sucias.
    """
    df = df.copy()
    df["_fingerprint"] = df[columna_texto].map(generar_fingerprint)

    filas_finales = []
    for huella, grupo in df.groupby("_fingerprint"):
        canonico = elegir_valor_canonico(Counter(grupo[columna_texto]))
        fila_canonica = grupo[grupo[columna_texto] == canonico].iloc[0]
        filas_finales.append(fila_canonica)

    resultado = pd.DataFrame(filas_finales).drop(columns=["_fingerprint"]).reset_index(drop=True)
    print(f"    Catalogo: {len(df)} filas -> {len(resultado)} filas tras deduplicar por cluster "
          f"({len(df) - len(resultado)} filas sucias eliminadas)")
    return resultado


# ============================================================================
# APLICACION A LAS 9 TABLAS
# ============================================================================

print("Limpiando DimCliente...")
dim_cliente = pd.read_excel(INPUT_DIR / "DimCliente.xlsx")
dim_cliente = limpiar_columna_por_cluster(dim_cliente, "CountryName")
dim_cliente = limpiar_columna_por_cluster(dim_cliente, "CityName")
dim_cliente = imputar_texto_por_grupo(dim_cliente, "CountryName", "CountryCode")
dim_cliente.to_excel(OUTPUT_DIR / "DimCliente.xlsx", index=False)


print("\nLimpiando DimRestaurante...")
dim_restaurante = pd.read_excel(INPUT_DIR / "DimRestaurante.xlsx")
dim_restaurante = limpiar_columna_por_cluster(dim_restaurante, "CountryName")
dim_restaurante = limpiar_columna_por_cluster(dim_restaurante, "CityName")
dim_restaurante = limpiar_columna_por_cluster(dim_restaurante, "CuisineName")
dim_restaurante = limpiar_columna_por_cluster(dim_restaurante, "RatingTier")
dim_restaurante = imputar_texto_por_grupo(dim_restaurante, "CountryName", "CountryCode")
dim_restaurante.to_excel(OUTPUT_DIR / "DimRestaurante.xlsx", index=False)


print("\nLimpiando DimMenuItem...")
dim_menuitem = pd.read_excel(INPUT_DIR / "DimMenuItem.xlsx")
dim_menuitem = limpiar_columna_por_cluster(dim_menuitem, "Category")
dim_menuitem = limpiar_columna_por_cluster(dim_menuitem, "CuisineName")
dim_menuitem.to_excel(OUTPUT_DIR / "DimMenuItem.xlsx", index=False)


print("\nLimpiando DimMetodoPago (deduplicando catalogo)...")
dim_metodopago = pd.read_excel(INPUT_DIR / "DimMetodoPago.xlsx")
dim_metodopago = deduplicar_catalogo_por_cluster(dim_metodopago, "PaymentMethod", "MetodoPagoKey")
dim_metodopago.to_excel(OUTPUT_DIR / "DimMetodoPago.xlsx", index=False)


print("\nLimpiando DimEstadoOrden (deduplicando catalogo)...")
dim_estadoorden = pd.read_excel(INPUT_DIR / "DimEstadoOrden.xlsx")
dim_estadoorden = deduplicar_catalogo_por_cluster(dim_estadoorden, "OrderStatus", "EstadoKey")
dim_estadoorden.to_excel(OUTPUT_DIR / "DimEstadoOrden.xlsx", index=False)


print("\nLimpiando FactOrders (imputacion numerica agrupada por restaurante)...")
fact_orders = pd.read_excel(INPUT_DIR / "FactOrders.xlsx")
fact_orders = imputar_numerico_por_grupo(fact_orders, "DeliveryTimeMinutes", "RestauranteKey")
fact_orders = imputar_numerico_por_grupo(fact_orders, "CustomerRating", "RestauranteKey")
fact_orders.to_excel(OUTPUT_DIR / "FactOrders.xlsx", index=False)


print("\nCopiando sin cambios: DimTiempo, FactReviews, FactPriceHistory...")
for nombre in ["DimTiempo.xlsx", "FactReviews.xlsx", "FactPriceHistory.xlsx"]:
    pd.read_excel(INPUT_DIR / nombre).to_excel(OUTPUT_DIR / nombre, index=False)

print(f"\nListo. Archivos limpios en: {OUTPUT_DIR}")
