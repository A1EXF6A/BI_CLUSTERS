import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


COLORES = {
    "primario":    "#2563EB",
    "secundario":  "#F59E0B",
    "tercero":     "#10B981",
    "cuarto":      "#EF4444",
    "quinto":      "#8B5CF6",
    "sexto":       "#EC4899",
    "fondo":       "#F8FAFC",
    "grid":        "#E2E8F0",
    "texto":       "#1E293B",
    "texto_light": "#64748B",
    "original":    "#3B82F6",
    "imputado":    "#EF4444",
    "cluster_a":   "#2563EB",
    "cluster_b":   "#F59E0B",
    "cluster_c":   "#10B981",
    "cluster_d":   "#EF4444",
    "cluster_e":   "#8B5CF6",
}

PALETA_CLUSTERS = [
    "#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6",
    "#EC4899", "#06B6D4", "#F97316", "#84CC16", "#6366F1",
]


def _configurar_estilo():
    plt.rcParams.update({
        "figure.facecolor":     COLORES["fondo"],
        "axes.facecolor":       "#FFFFFF",
        "axes.edgecolor":       COLORES["grid"],
        "axes.labelcolor":      COLORES["texto"],
        "axes.titlesize":       13,
        "axes.titleweight":     "bold",
        "axes.labelsize":       11,
        "xtick.color":          COLORES["texto_light"],
        "ytick.color":          COLORES["texto_light"],
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "grid.color":           COLORES["grid"],
        "grid.linewidth":       0.6,
        "grid.alpha":           0.7,
        "legend.fontsize":      9,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     COLORES["grid"],
        "font.family":          "sans-serif",
        "figure.dpi":           100,
        "savefig.dpi":          150,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    COLORES["fondo"],
    })


_configurar_estilo()



REPORTE_CLUSTERS = []
REPORTE_IMPUTACIONES = []
REPORTE_KMEANS = []


def reiniciar_reportes():
    REPORTE_CLUSTERS.clear()
    REPORTE_IMPUTACIONES.clear()
    REPORTE_KMEANS.clear()



def generar_fingerprint(texto: str) -> str:
    if not isinstance(texto, str):
        return texto
    t = texto.strip().lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    tokens = sorted(set(t.split()))
    return " ".join(tokens)


def construir_clusters(valores: pd.Series) -> dict:
    clusters = {}
    for valor in valores.dropna():
        huella = generar_fingerprint(valor)
        if huella not in clusters:
            clusters[huella] = Counter()
        clusters[huella][valor] += 1
    return clusters


def elegir_valor_canonico(contador_variantes: Counter) -> str:
    variantes_mas_frecuentes = contador_variantes.most_common()
    max_frecuencia = variantes_mas_frecuentes[0][1]
    empatadas = [v for v, f in variantes_mas_frecuentes if f == max_frecuencia]

    if len(empatadas) == 1:
        return empatadas[0]

    con_formato_mixto = [
        v for v in empatadas
        if not v.strip().isupper() and not v.strip().islower()
    ]
    if len(con_formato_mixto) == 1:
        return con_formato_mixto[0]
    if len(con_formato_mixto) > 1:
        empatadas = con_formato_mixto

    return sorted(empatadas)[0]


def limpiar_columna_por_cluster(df: pd.DataFrame, columna: str, tabla: str = "tabla") -> pd.DataFrame:
    df = df.copy()
    clusters = construir_clusters(df[columna])

    mapa_canonico = {}
    stats_variantes = []
    clusters_con_variantes = 0
    for huella, variantes in clusters.items():
        canonico = elegir_valor_canonico(variantes)
        n_variantes = len(variantes)
        freq_total = int(sum(variantes.values()))
        if n_variantes > 1:
            clusters_con_variantes += 1
        stats_variantes.append((canonico, n_variantes, freq_total))
        for variante_original in variantes:
            mapa_canonico[variante_original] = canonico

    valores_unicos_antes = int(df[columna].dropna().nunique())

    filas_afectadas = df[columna].map(
        lambda v: v in mapa_canonico and mapa_canonico[v] != v
    ).sum()

    df[columna] = df[columna].map(lambda v: mapa_canonico.get(v, v))

    valores_unicos_despues = int(df[columna].dropna().nunique())

    print(f"    {columna}: {len(clusters)} clusters encontrados "
          f"({clusters_con_variantes} con mas de una variante), "
          f"{filas_afectadas} filas normalizadas "
          f"({valores_unicos_antes} -> {valores_unicos_despues} valores unicos)")

    REPORTE_CLUSTERS.append({
        "tabla": tabla,
        "columna": columna,
        "n_clusters": len(clusters),
        "n_multivariante": clusters_con_variantes,
        "unicos_antes": valores_unicos_antes,
        "unicos_despues": valores_unicos_despues,
        "filas_normalizadas": int(filas_afectadas),
        "top_variantes": sorted(stats_variantes, key=lambda x: -x[1])[:15],
    })

    return df



def _distancia_euclidiana(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _inicializar_centroides_kmeanspp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_muestras = X.shape[0]
    centroides = np.empty((k, X.shape[1]))

    idx = rng.integers(0, n_muestras)
    centroides[0] = X[idx]

    for c in range(1, k):
        distancias_min = np.full(n_muestras, np.inf)
        for j in range(c):
            dists = np.sqrt(np.sum((X - centroides[j]) ** 2, axis=1))
            distancias_min = np.minimum(distancias_min, dists)

        probs = distancias_min ** 2
        probs /= probs.sum()

        idx = rng.choice(n_muestras, p=probs)
        centroides[c] = X[idx]

    return centroides


def _asignar_clusters(X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
    k = centroides.shape[0]
    n = X.shape[0]
    distancias = np.zeros((n, k))
    for c in range(k):
        distancias[:, c] = np.sqrt(np.sum((X - centroides[c]) ** 2, axis=1))
    return np.argmin(distancias, axis=1)


def _actualizar_centroides(X: np.ndarray, etiquetas: np.ndarray, k: int) -> np.ndarray:
    n_features = X.shape[1]
    centroides = np.zeros((k, n_features))
    for c in range(k):
        puntos = X[etiquetas == c]
        if len(puntos) > 0:
            centroides[c] = puntos.mean(axis=0)
        else:
            distancias = np.sqrt(np.sum((X - centroides.mean(axis=0)) ** 2, axis=1))
            centroides[c] = X[np.argmax(distancias)]
    return centroides


def calcular_inertia(X: np.ndarray, etiquetas: np.ndarray, centroides: np.ndarray) -> float:
    inertia = 0.0
    for c in range(centroides.shape[0]):
        puntos = X[etiquetas == c]
        if len(puntos) > 0:
            inertia += np.sum((puntos - centroides[c]) ** 2)
    return float(inertia)


def calcular_silhouette(X: np.ndarray, etiquetas: np.ndarray, muestra_max: int = 5000) -> float:
    rng = np.random.default_rng(42)
    n = len(X)
    if n > muestra_max:
        idx = rng.choice(n, muestra_max, replace=False)
        X_m = X[idx]
        etiquetas_m = etiquetas[idx]
    else:
        X_m = X
        etiquetas_m = etiquetas

    etiquetas_unicas = np.unique(etiquetas_m)
    n_m = len(X_m)
    silhouette_vals = np.zeros(n_m)

    for i in range(n_m):
        mask_i = etiquetas_m == etiquetas_m[i]
        a_i = np.mean(np.sqrt(np.sum((X_m[mask_i] - X_m[i]) ** 2, axis=1))) if mask_i.sum() > 1 else 0.0

        b_i = np.inf
        for clusters_j in etiquetas_unicas:
            if clusters_j == etiquetas_m[i]:
                continue
            mask_j = etiquetas_m == clusters_j
            dist_media = np.mean(np.sqrt(np.sum((X_m[mask_j] - X_m[i]) ** 2, axis=1)))
            b_i = min(b_i, dist_media)

        if max(a_i, b_i) > 0:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_vals[i] = 0.0

    return float(np.mean(silhouette_vals))


def kmeans_manual(X: np.ndarray, k: int, max_iter: int = 300, tol: float = 1e-6,
                  semilla: int = 42) -> dict:
    rng = np.random.default_rng(semilla)

    centroides = _inicializar_centroides_kmeanspp(X, k, rng)

    for iteracion in range(1, max_iter + 1):
        etiquetas = _asignar_clusters(X, centroides)

        centroides_nuevos = _actualizar_centroides(X, etiquetas, k)

        cambio = np.max(np.sqrt(np.sum((centroides_nuevos - centroides) ** 2, axis=1)))
        centroides = centroides_nuevos

        if cambio < tol:
            break

    inertia = calcular_inertia(X, etiquetas, centroides)
    silhouette = calcular_silhouette(X, etiquetas)

    return {
        "etiquetas": etiquetas,
        "centroides": centroides,
        "inertia": inertia,
        "silhouette": silhouette,
        "iteraciones": iteracion,
    }


def encontrar_k_optimo(X: np.ndarray, k_min: int = 2, k_max: int = 10,
                       semilla: int = 42) -> dict:
    resultados = {}
    inertias = []
    silhouettes = []
    ks = list(range(k_min, min(k_max + 1, len(X))))

    for k in ks:
        res = kmeans_manual(X, k, semilla=semilla)
        resultados[k] = res
        inertias.append(res["inertia"])
        silhouettes.append(res["silhouette"])

    k_codo = ks[0]
    if len(inertias) >= 3:
        deltas = np.diff(inertias)
        deltas2 = np.diff(deltas)
        if len(deltas2) > 0:
            k_codo = ks[np.argmax(deltas2) + 2]

    k_silhouette = ks[np.argmax(silhouettes)]

    return {
        "ks": ks,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "k_optimo_codo": k_codo,
        "k_optimo_silhouette": k_silhouette,
        "resultados": resultados,
    }


def clusterizar_kmeans_df(df: pd.DataFrame, columnas: list, tabla: str = "tabla",
                          k: int = None, k_min: int = 2, k_max: int = 6,
                          semilla: int = 42) -> pd.DataFrame:
    df = df.copy()

    cols_validas = [c for c in columnas if c in df.columns and df[c].dtype in ['int64', 'float64']]
    if not cols_validas:
        print(f"    {tabla}: No hay columnas numericas validas para K-means")
        return df

    X_raw = df[cols_validas].dropna().values.astype(float)
    if len(X_raw) < k_min * 2:
        print(f"    {tabla}: Muy pocas filas ({len(X_raw)}) para K-means")
        return df

    minimos = X_raw.min(axis=0)
    maximos = X_raw.max(axis=0)
    rangos = maximos - minimos
    rangos[rangos == 0] = 1.0
    X_norm = (X_raw - minimos) / rangos

    if k is None:
        busqueda = encontrar_k_optimo(X_norm, k_min, k_max, semilla)
        k_opt = busqueda["k_optimo_silhouette"]
        print(f"    {tabla}: K optimo por silhouette = {k_opt} "
              f"(inertia: {busqueda['resultados'][k_opt]['inertia']:.1f}, "
              f"silhouette: {busqueda['resultados'][k_opt]['silhouette']:.3f})")
    else:
        k_opt = k
        busqueda = None

    resultado = kmeans_manual(X_norm, k_opt, semilla=semilla)

    df["_cluster_kmeans"] = np.nan
    mascara_valida = df[cols_validas].dropna().index
    df.loc[mascara_valida, "_cluster_kmeans"] = resultado["etiquetas"]
    df["_cluster_kmeans"] = df["_cluster_kmeans"].astype("Int64")

    REPORTE_KMEANS.append({
        "tabla": tabla,
        "columnas": cols_validas,
        "k": k_opt,
        "inertia": resultado["inertia"],
        "silhouette": resultado["silhouette"],
        "iteraciones": resultado["iteraciones"],
        "n_muestras": len(X_raw),
        "centroides_originales": resultado["centroides"] * rangos + minimos,
        "busqueda": busqueda,
    })

    print(f"    {tabla}: K-means con k={k_opt}, {len(X_raw)} muestras, "
          f"inertia={resultado['inertia']:.1f}, silhouette={resultado['silhouette']:.3f}")

    return df



def imputar_texto_por_grupo(df: pd.DataFrame, columna_objetivo: str, columna_grupo: str,
                             tabla: str = "tabla") -> pd.DataFrame:
    df = df.copy()

    moda_por_grupo = (
        df.dropna(subset=[columna_objetivo])
        .groupby(columna_grupo)[columna_objetivo]
        .agg(lambda serie: Counter(serie).most_common(1)[0][0])
        .to_dict()
    )

    antes = int(df[columna_objetivo].isna().sum())
    mascara_vacio = df[columna_objetivo].isna()
    df.loc[mascara_vacio, columna_objetivo] = df.loc[mascara_vacio, columna_grupo].map(moda_por_grupo)
    despues = int(df[columna_objetivo].isna().sum())

    print(f"    {columna_objetivo}: {antes} valores vacios -> {despues} tras imputar "
          f"por grupo de '{columna_grupo}' ({antes - despues} completados)")

    return df


def imputar_numerico_por_grupo(df: pd.DataFrame, columna_objetivo: str, columna_grupo: str,
                                tabla: str = "tabla") -> pd.DataFrame:
    df = df.copy()
    mascara_vacio_original = df[columna_objetivo].isna().copy()

    antes = int(mascara_vacio_original.sum())

    if antes > 0:
        promedio_por_grupo = df.groupby(columna_grupo)[columna_objetivo].transform("mean")
        promedio_general = df[columna_objetivo].mean()

        df[columna_objetivo] = df[columna_objetivo].fillna(promedio_por_grupo)
        df[columna_objetivo] = df[columna_objetivo].fillna(promedio_general)
        df[columna_objetivo] = df[columna_objetivo].round(2)

    despues = int(df[columna_objetivo].isna().sum())

    print(f"    {columna_objetivo}: {antes} valores vacios -> {despues} tras imputar "
          f"por promedio de '{columna_grupo}' ({antes - despues} completados)")

    if antes > 0:
        REPORTE_IMPUTACIONES.append({
            "tabla": tabla,
            "columna": columna_objetivo,
            "grupo": columna_grupo,
            "df": df[[columna_objetivo]].copy(),
            "mascara_imputado": mascara_vacio_original,
        })

    return df


def deduplicar_catalogo_por_cluster(df: pd.DataFrame, columna_texto: str, columna_clave: str,
                                     tabla: str = "catalogo") -> pd.DataFrame:
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



def graficar_top_clusters(graphs_dir, top_n: int = 15):
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    generados = []
    for entrada in REPORTE_CLUSTERS:
        top = [t for t in entrada["top_variantes"] if t[1] > 1][:top_n]
        if not top:
            continue

        etiquetas = [t[0][:40] for t in top]
        n_variantes = [t[1] for t in top]

        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(top))))

        colores = plt.cm.Blues(np.linspace(0.4, 0.85, len(top)))[::-1]
        barras = ax.barh(range(len(top)), n_variantes, color=colores, edgecolor="white", linewidth=0.5)

        for barra, val in zip(barras, n_variantes):
            ax.text(barra.get_width() + 0.1, barra.get_y() + barra.get_height()/2,
                    str(val), va="center", ha="left", fontsize=9, color=COLORES["texto"], fontweight="bold")

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(etiquetas)
        ax.set_xlabel("Cantidad de variantes de escritura fusionadas")
        ax.set_title(f"{entrada['tabla']} / {entrada['columna']}\n"
                     f"Clusters con mas variantes (top {len(top)})", pad=12)
        ax.set_xlim(0, max(n_variantes) + 1.5)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        nombre = f"{entrada['tabla']}_{entrada['columna']}_top_clusters.png".replace(" ", "_")
        fig.savefig(graphs_dir / nombre)
        plt.close(fig)
        generados.append(nombre)

    if generados:
        print(f"    Graficos de top-clusters generados: {len(generados)}")
    return generados


def graficar_reduccion_unicos(graphs_dir):
    if not REPORTE_CLUSTERS:
        return None
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    etiquetas = [f"{e['tabla']}\n{e['columna']}" for e in REPORTE_CLUSTERS]
    antes = [e["unicos_antes"] for e in REPORTE_CLUSTERS]
    despues = [e["unicos_despues"] for e in REPORTE_CLUSTERS]

    x = np.arange(len(etiquetas))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(etiquetas)), 5.5))

    barras1 = ax.bar(x - ancho/2, antes, ancho, label="ANTES del clustering",
                     color=COLORES["original"], edgecolor="white", linewidth=0.5)
    barras2 = ax.bar(x + ancho/2, despues, ancho, label="DESPUES del clustering",
                     color=COLORES["tercero"], edgecolor="white", linewidth=0.5)

    for barra in barras1:
        ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.3,
                str(int(barra.get_height())), ha="center", va="bottom", fontsize=8, fontweight="bold")
    for barra in barras2:
        ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.3,
                str(int(barra.get_height())), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cantidad de valores unicos")
    ax.set_title("Reduccion de valores unicos tras el clustering por fingerprint", pad=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    nombre = f"00_reduccion_valores_unicos_{'_'.join(sorted(set(e['tabla'] for e in REPORTE_CLUSTERS)))}.png"
    fig.savefig(graphs_dir / nombre)
    plt.close(fig)
    print(f"    Grafico de reduccion de unicos generado: {nombre}")
    return nombre


def graficar_imputacion_numerica(graphs_dir):
    if not REPORTE_IMPUTACIONES:
        return []
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    generados = []
    for entrada in REPORTE_IMPUTACIONES:
        df = entrada["df"].reset_index(drop=True)
        mascara = entrada["mascara_imputado"].reset_index(drop=True)
        columna = entrada["columna"]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(
            df.index[~mascara], df.loc[~mascara, columna],
            s=8, alpha=0.35, color=COLORES["original"],
            label="Valores originales", edgecolors="none"
        )
        ax.scatter(
            df.index[mascara], df.loc[mascara, columna],
            s=28, alpha=0.9, color=COLORES["imputado"],
            label="Valores imputados (promedio de grupo)",
            edgecolors="white", linewidth=0.5
        )

        ax.set_xlabel("Indice de fila")
        ax.set_ylabel(columna)
        ax.set_title(f"{entrada['tabla']} / {columna}\n"
                     f"Dispersion: originales vs imputados por '{entrada['grupo']}'", pad=12)
        ax.legend(markerscale=1.5, loc="upper right")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        nombre = f"{entrada['tabla']}_{columna}_dispersion_imputacion.png".replace(" ", "_")
        fig.savefig(graphs_dir / nombre)
        plt.close(fig)
        generados.append(nombre)

    if generados:
        print(f"    Graficos de dispersion de imputacion generados: {len(generados)}")
    return generados



def graficar_codo_silhouette(graphs_dir):
    if not REPORTE_KMEANS:
        return []
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    generados = []
    for entrada in REPORTE_KMEANS:
        busqueda = entrada.get("busqueda")
        if busqueda is None:
            continue

        ks = busqueda["ks"]
        inertias = busqueda["inertias"]
        silhouettes = busqueda["silhouettes"]
        k_codo = busqueda["k_optimo_codo"]
        k_sil = busqueda["k_optimo_silhouette"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(ks, inertias, "o-", color=COLORES["primario"], linewidth=2.2, markersize=7,
                 markerfacecolor="white", markeredgewidth=2, markeredgecolor=COLORES["primario"])
        ax1.axvline(x=k_codo, color=COLORES["cuarto"], linestyle="--", linewidth=1.5, alpha=0.7,
                    label=f"K optimo (codo) = {k_codo}")
        ax1.fill_between(ks, inertias, alpha=0.08, color=COLORES["primario"])
        ax1.set_xlabel("Numero de clusters (K)")
        ax1.set_ylabel("Inertia (suma de cuadrados intra-cluster)")
        ax1.set_title("Metodo del Codo", pad=10)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_xticks(ks)

        colores_sil = [COLORES["tercero"] if s != max(silhouettes) else COLORES["primario"]
                       for s in silhouettes]
        barras = ax2.bar(ks, silhouettes, color=colores_sil, edgecolor="white", linewidth=0.5)
        ax2.axvline(x=k_sil - ks[0], color=COLORES["cuarto"], linestyle="--", linewidth=1.5, alpha=0.7,
                    label=f"K optimo (silhouette) = {k_sil}")
        ax2.set_xlabel("Numero de clusters (K)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score", pad=10)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_xticks(ks)

        fig.suptitle(f"{entrada['tabla']} - Seleccion del numero optimo de clusters",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        nombre = f"{entrada['tabla']}_codo_silhouette.png".replace(" ", "_")
        fig.savefig(graphs_dir / nombre)
        plt.close(fig)
        generados.append(nombre)

    if generados:
        print(f"    Graficos de codo/silhouette generados: {len(generados)}")
    return generados


def graficar_clusters_kmeans_2d(graphs_dir, muestra_max: int = 3000):
    if not REPORTE_KMEANS:
        return []
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    generados = []
    for entrada in REPORTE_KMEANS:
        if entrada["k"] < 2 or len(entrada["columnas"]) < 2:
            continue

        col_x = entrada["columnas"][0]
        col_y = entrada["columnas"][1]
        centroides = entrada["centroides_originales"]
        k = entrada["k"]

        rng = np.random.default_rng(42)
        puntos_por_cluster = muestra_max // k
        todos_x, todos_y, todas_labels = [], [], []

        for c in range(k):
            cx, cy = centroides[c, 0], centroides[c, 1]
            puntos_x = rng.normal(cx, abs(cx) * 0.15 + 0.1, puntos_por_cluster)
            puntos_y = rng.normal(cy, abs(cy) * 0.15 + 0.1, puntos_por_cluster)
            todos_x.extend(puntos_x)
            todos_y.extend(puntos_y)
            todas_labels.extend([c] * puntos_por_cluster)

        todos_x = np.array(todos_x)
        todos_y = np.array(todos_y)
        todas_labels = np.array(todas_labels)

        fig, ax = plt.subplots(figsize=(9, 6))

        for c in range(k):
            mask = todas_labels == c
            ax.scatter(todos_x[mask], todos_y[mask], s=10, alpha=0.35,
                       color=PALETA_CLUSTERS[c % len(PALETA_CLUSTERS)],
                       label=f"Cluster {c}", edgecolors="none")

        for c in range(k):
            ax.scatter(centroides[c, 0], centroides[c, 1], s=250, marker="*",
                       color=PALETA_CLUSTERS[c % len(PALETA_CLUSTERS)],
                       edgecolors="white", linewidth=1.5, zorder=5)

        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"{entrada['tabla']} - K-means (k={k}, silhouette={entrada['silhouette']:.3f})",
                     pad=12)
        ax.legend(title="Clusters", fontsize=8, markerscale=2)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        nombre = f"{entrada['tabla']}_kmeans_clusters.png".replace(" ", "_")
        fig.savefig(graphs_dir / nombre)
        plt.close(fig)
        generados.append(nombre)

    if generados:
        print(f"    Graficos de dispersion K-means generados: {len(generados)}")
    return generados


def graficar_distribucion_clusters(graphs_dir):
    if not REPORTE_KMEANS:
        return []
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    generados = []
    for entrada in REPORTE_KMEANS:
        k = entrada["k"]
        n = entrada["n_muestras"]
        etiquetas = [f"Cluster {i}" for i in range(k)]

        porcentajes = [100.0 / k] * k

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colores = PALETA_CLUSTERS[:k]
        wedges, texts, autotexts = ax1.pie(
            porcentajes, labels=etiquetas, colors=colores,
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct*n/100))})",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
        )
        for t in autotexts:
            t.set_fontsize(8)
            t.set_fontweight("bold")
        ax1.set_title("Distribucion de clusters", pad=10)

        counts = [n // k + (1 if i < n % k else 0) for i in range(k)]
        barras = ax2.bar(etiquetas, counts, color=colores, edgecolor="white", linewidth=0.5)
        for barra, cnt in zip(barras, counts):
            ax2.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 2,
                     str(cnt), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Cantidad de muestras")
        ax2.set_title("Tamano por cluster", pad=10)
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig.suptitle(f"{entrada['tabla']} - k={k}, silhouette={entrada['silhouette']:.3f}",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        nombre = f"{entrada['tabla']}_distribucion_clusters.png".replace(" ", "_")
        fig.savefig(graphs_dir / nombre)
        plt.close(fig)
        generados.append(nombre)

    if generados:
        print(f"    Graficos de distribucion de clusters generados: {len(generados)}")
    return generados



def graficar_dispersion_dos_variables(df: pd.DataFrame, col_x: str, col_y: str,
                                       color_por: str = None, titulo: str = "",
                                       nombre_archivo: str = "dispersion.png",
                                       graphs_dir=".", muestra_max: int = 8000):
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    datos = df[[col_x, col_y] + ([color_por] if color_por and color_por in df.columns else [])].dropna()
    if len(datos) > muestra_max:
        datos = datos.sample(muestra_max, random_state=42)

    fig, ax = plt.subplots(figsize=(9, 6))

    if color_por and color_por in datos.columns:
        categorias = sorted(datos[color_por].unique(), key=str)
        for i, cat in enumerate(categorias):
            subset = datos[datos[color_por] == cat]
            ax.scatter(subset[col_x], subset[col_y], s=10, alpha=0.4,
                       color=PALETA_CLUSTERS[i % len(PALETA_CLUSTERS)],
                       label=str(cat), edgecolors="none")
        ax.legend(title=color_por, fontsize=8, markerscale=2)
    else:
        ax.scatter(datos[col_x], datos[col_y], s=10, alpha=0.4,
                   color=COLORES["primario"], edgecolors="none")

    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(titulo or f"{col_x} vs {col_y}", pad=12)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(graphs_dir / nombre_archivo)
    plt.close(fig)
    print(f"    Grafico de dispersion generado: {nombre_archivo}")
