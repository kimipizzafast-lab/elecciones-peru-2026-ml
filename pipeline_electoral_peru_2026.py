"""
Pipeline Científico Completo - Machine Learning No Supervisado para Paper LACCEI 2026
Estudio: Clusterización territorial de las preferencias electorales en la primera vuelta presidencial del Perú 2026
Universo: 1,890 circunscripciones distritales oficiales de la ONPE
Autores: Luis Miguel Tirabantes Córdova, Omar Santillán Aching
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances
)
from scipy import stats

RS = 42
FIG_DIR = 'figuras_laccei'
os.makedirs(FIG_DIR, exist_ok=True)

print("="*70, flush=True)
print("1. CARGA Y AUDITORÍA DE DATOS OFICIALES DE LA ONPE (N = 1,890 DISTRITOS)", flush=True)
print("="*70, flush=True)

df = pd.read_excel('dataset_onpe_CENSAL_1891.xlsx')
n_distritos = len(df)
print(f"[*] Registros distritales cargados: {n_distritos}", flush=True)

cols_all = [c for c in df.columns if c.startswith('PCT_') and c not in ('PCT_POBREZA_TOTAL','PCT_VOTOS_EN_BLANCO','PCT_VOTOS_NULOS') and 'TRABAJADORES' not in c.upper()]
n_formulas = len(cols_all)
print(f"[*] Fórmulas presidenciales válidas: {n_formulas}", flush=True)

X_raw = df[cols_all].fillna(0.0).values

# Conteo de ganadores por idxmax
win_col = df[cols_all].idxmax(axis=1).str.replace('PCT_', '')
conteo_ganadores = win_col.value_counts().to_dict()

# Identificar empates exactos (margen = 0.000 pp)
srt = np.sort(X_raw, axis=1)[:, ::-1]
margen_raw = srt[:, 0] - srt[:, 1]
empates_mask = (margen_raw == 0.0)
df_empates = df[empates_mask][['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'UBIGEO_RENIEC']].copy()
print(f"[*] Distritos con empate exacto a tres decimales (0.000 pp): {len(df_empates)}", flush=True)

# 2. INGENIERÍA DE CARACTERÍSTICAS
print("\n" + "="*70, flush=True)
print("2. INGENIERÍA DE CARACTERÍSTICAS Y ESQUEMAS DE REPRESENTACIÓN", flush=True)
print("="*70, flush=True)

# Esquema A: 35 dimensiones crudas
Xa = StandardScaler().fit_transform(X_raw)

# Esquema B: Filtro de umbral electoral + OTROS + Variables estructurales (NEP, Margen de Victoria)
top_parties = ['PCT_JUNTOS_POR_EL_PERÚ', 'PCT_FUERZA_POPULAR', 'PCT_PARTIDO_CÍVICO_OBRAS', 'PCT_AHORA_NACIÓN_-_AN', 'PCT_RENOVACIÓN_POPULAR', 'PCT_PARTIDO_DEL_BUEN_GOBIERNO']
df['PCT_OTROS'] = df[[c for c in cols_all if c not in top_parties]].sum(axis=1)

# Número Efectivo de Partidos (Laakso-Taagepera)
prop = X_raw / 100.0
df['NEP'] = 1.0 / np.sum(prop**2, axis=1)

# Margen de Victoria distrital
df['MARGEN_VICTORIA'] = margen_raw

vars_b = top_parties + ['PCT_OTROS', 'NEP', 'MARGEN_VICTORIA']
Xb_raw = df[vars_b].values
Xb = StandardScaler().fit_transform(Xb_raw)

# Esquema C: CLR + PCA (85% varianza)
delta = 0.005
Z = X_raw.copy()
for i in range(Z.shape[0]):
    zeros = Z[i] == 0
    nz = zeros.sum()
    Z[i, zeros] = delta
    Z[i, ~zeros] = Z[i, ~zeros] * (1.0 - nz * delta / 100.0)
L = np.log(Z)
CLR = L - L.mean(axis=1, keepdims=True)

pca_clr = PCA(n_components=0.85, random_state=RS)
Xc = pca_clr.fit_transform(CLR)
print(f"[*] Esquema C: CLR reducido con PCA a {Xc.shape[1]} dimensiones ({pca_clr.explained_variance_ratio_.sum()*100:.2f}% var)", flush=True)

# Función K-Medoids PAM
def kmedoids_pam(X, k, max_iter=100, random_state=RS):
    np.random.seed(random_state)
    D = pairwise_distances(X, metric='euclidean')
    n = len(X)
    medoids = np.random.choice(n, k, replace=False)
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = np.zeros(k, dtype=int)
        for i in range(k):
            members = np.where(labels == i)[0]
            if len(members) == 0:
                new_medoids[i] = medoids[i]
            else:
                sub_D = D[np.ix_(members, members)]
                new_medoids[i] = members[np.argmin(sub_D.sum(axis=1))]
        if np.array_equal(medoids, new_medoids):
            break
        medoids = new_medoids
    labels = np.argmin(D[:, medoids], axis=1)
    return medoids, labels

# 3. EXPERIMENTACIÓN MULTIALGORÍTMICA (TABLA 3)
print("\n" + "="*70, flush=True)
print("3. MODELAMIENTO Y COMPARACIÓN DE RENDIMIENTO (TABLA 3)", flush=True)
print("="*70, flush=True)

tabla_modelos = []
dict_etiquetas = {}

# Esquema A (Línea Base: 35 vars)
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=RS).fit(Xa)
    s = silhouette_score(Xa, km.labels_)
    ch = calinski_harabasz_score(Xa, km.labels_)
    db = davies_bouldin_score(Xa, km.labels_)
    sizes = [int(x) for x in np.bincount(km.labels_)]
    tabla_modelos.append({
        'esquema': 'Esquema A (35 vars crudas)',
        'algoritmo': 'K-Means',
        'k': k,
        'silhouette': round(float(s), 4),
        'calinski_harabasz': round(float(ch), 2),
        'davies_bouldin': round(float(db), 4),
        'tamanos': sizes
    })

# Esquema B (Ingeniería de Características: Fuerzas clave + NEP + MV)
modelos_b = ['K-Means', 'K-Medoids (PAM)', 'HAC (Ward)', 'GMM (Full)', 'Spectral Clustering']
for model_name in modelos_b:
    for k in [2, 3, 4, 5]:
        if model_name == 'K-Means':
            m = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=RS).fit(Xb)
            labs = m.labels_
        elif model_name == 'K-Medoids (PAM)':
            _, labs = kmedoids_pam(Xb, k, random_state=RS)
        elif model_name == 'HAC (Ward)':
            labs = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xb)
        elif model_name == 'GMM (Full)':
            labs = GaussianMixture(n_components=k, covariance_type='full', n_init=5, random_state=RS).fit_predict(Xb)
        elif model_name == 'Spectral Clustering':
            labs = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=15, random_state=RS).fit_predict(Xb)
        
        dict_etiquetas[(model_name, k)] = labs
        s = silhouette_score(Xb, labs)
        ch = calinski_harabasz_score(Xb, labs)
        db = davies_bouldin_score(Xb, labs)
        sizes = [int(x) for x in np.bincount(labs)]
        tabla_modelos.append({
            'esquema': 'Esquema B (Ingeniería Características)',
            'algoritmo': model_name,
            'k': k,
            'silhouette': round(float(s), 4),
            'calinski_harabasz': round(float(ch), 2),
            'davies_bouldin': round(float(db), 4),
            'tamanos': sizes
        })

# Esquema C (CLR + PCA)
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=RS).fit(Xc)
    s = silhouette_score(Xc, km.labels_)
    ch = calinski_harabasz_score(Xc, km.labels_)
    db = davies_bouldin_score(Xc, km.labels_)
    sizes = [int(x) for x in np.bincount(km.labels_)]
    tabla_modelos.append({
        'esquema': 'Esquema C (CLR + PCA 85%)',
        'algoritmo': 'K-Means',
        'k': k,
        'silhouette': round(float(s), 4),
        'calinski_harabasz': round(float(ch), 2),
        'davies_bouldin': round(float(db), 4),
        'tamanos': sizes
    })

print(f"[*] Total experimentos ejecutados en Tabla 3: {len(tabla_modelos)}", flush=True)

# 4. MATRIZ DE ESTABILIDAD Y CONCORDANCIA (TABLA 4: ARI y NMI)
print("\n" + "="*70, flush=True)
print("4. CONCORDANCIA ENTRE MODELOS (TABLA 4 - ARI / NMI)", flush=True)
print("="*70, flush=True)

tabla_concordancia = []
pares_comparacion = [
    ('K-Means', 'K-Medoids (PAM)'),
    ('K-Means', 'HAC (Ward)'),
    ('K-Means', 'Spectral Clustering'),
    ('K-Means', 'GMM (Full)'),
    ('HAC (Ward)', 'Spectral Clustering'),
    ('HAC (Ward)', 'GMM (Full)')
]

for m1, m2 in pares_comparacion:
    for k_val in [2, 3]:
        l1 = dict_etiquetas[(m1, k_val)]
        l2 = dict_etiquetas[(m2, k_val)]
        ari = adjusted_rand_score(l1, l2)
        nmi = normalized_mutual_info_score(l1, l2)
        tabla_concordancia.append({
            'par_modelos': f"{m1} vs. {m2}",
            'k': k_val,
            'ari': round(float(ari), 4),
            'nmi': round(float(nmi), 4)
        })
        print(f"{m1} vs {m2} (k={k_val}): ARI = {ari:.4f}, NMI = {nmi:.4f}", flush=True)

# 5. CARACTERIZACIÓN DE LOS CLÚSTERES ÓPTIMOS (TABLA 5 - PERFILES TERRITORIALES)
print("\n" + "="*70, flush=True)
print("5. CARACTERIZACIÓN DE PERFILES TERRITORIALES (TABLA 5)", flush=True)
print("="*70, flush=True)

# Modelo principal seleccionado: Esquema B con K-Means (k=3) y (k=2)
labels_k3 = dict_etiquetas[('K-Means', 3)]
# Reordenar etiquetas k=3 para que:
# Clúster 0 = Hegemónico JPP (sierra/sur)
# Clúster 1 = Competitivo / Disperso (centro/ciudades)
# Clúster 2 = Competitivo FP (costa/norte)
means_jpp = [df.loc[labels_k3 == c, 'PCT_JUNTOS_POR_EL_PERÚ'].mean() for c in range(3)]
order_k3 = np.argsort(means_jpp)[::-1] # 0 = mayor JPP
remap_k3 = {old: new for new, old in enumerate(order_k3)}
final_k3 = np.array([remap_k3[l] for l in labels_k3])
df['CLUSTER_K3'] = final_k3

perfiles_k3 = {}
for c in range(3):
    sub = df[df['CLUSTER_K3'] == c]
    perfiles_k3[f'Cluster_{c}'] = {
        'n': len(sub),
        'pct_nacional': round(len(sub) / n_distritos * 100, 2),
        'pct_jpp_media': round(sub['PCT_JUNTOS_POR_EL_PERÚ'].mean(), 2),
        'pct_fp_media': round(sub['PCT_FUERZA_POPULAR'].mean(), 2),
        'pct_pco_media': round(sub['PCT_PARTIDO_CÍVICO_OBRAS'].mean(), 2),
        'pct_an_media': round(sub['PCT_AHORA_NACIÓN_-_AN'].mean(), 2),
        'pct_rp_media': round(sub['PCT_RENOVACIÓN_POPULAR'].mean(), 2),
        'pct_pbg_media': round(sub['PCT_PARTIDO_DEL_BUEN_GOBIERNO'].mean(), 2),
        'pct_otros_media': round(sub['PCT_OTROS'].mean(), 2),
        'nep_media': round(sub['NEP'].mean(), 2),
        'margen_media': round(sub['MARGEN_VICTORIA'].mean(), 2),
        'deptos_top': sub['DEPARTAMENTO'].value_counts().head(5).to_dict()
    }
    print(f"Clúster {c}: n={len(sub)} ({len(sub)/n_distritos*100:.1f}%) | JPP={perfiles_k3[f'Cluster_{c}']['pct_jpp_media']}% | FP={perfiles_k3[f'Cluster_{c}']['pct_fp_media']}% | NEP={perfiles_k3[f'Cluster_{c}']['nep_media']} | MV={perfiles_k3[f'Cluster_{c}']['margen_media']} pp")

# Guardar asignación en Excel para trazabilidad
df.to_excel('resultados_clusters_laccei_2026.xlsx', index=False)
print("[*] Guardado resultados_clusters_laccei_2026.xlsx", flush=True)

# 6. GENERACIÓN DE FIGURAS EN ALTA RESOLUCIÓN (300 DPI)
print("\n" + "="*70, flush=True)
print("6. GENERACIÓN DE FIGURAS (FIGURAS 1 A 4)", flush=True)
print("="*70, flush=True)

# FIGURA 1: Arquitectura CRISP-DM
fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=300)
ax.axis('off')
fases = [
    ('Fase 1: Comprensión\ndel Dominio', 'Hiperfragmentación 2026\n35 fórmulas | N=1,890 dist.'),
    ('Fase 2: Comprensión\nde los Datos', 'API ONPE (JSON)\nHomologación UBIGEO'),
    ('Fase 3: Preparación\ny Características', 'Filtro umbral >=3%\nNEP | Margen Victoria | CLR'),
    ('Fase 4: Modelamiento\nMultialgorítmico', 'K-Means · PAM · HAC\n· GMM · Spectral (k=2..5)'),
    ('Fase 5: Evaluación\ny Validación', 'Silhouette · Calinski-H.\n· Davies-Bouldin · ARI/NMI'),
    ('Fase 6: Despliegue\ny Reproducibilidad', 'Mapas coropléticos, PCA\nSoftware abierto en GitHub')
]
pos = [(0.02, 0.55), (0.35, 0.55), (0.68, 0.55), (0.68, 0.08), (0.35, 0.08), (0.02, 0.08)]
for (t, d), (x, y) in zip(fases, pos):
    ax.add_patch(FancyBboxPatch((x, y), 0.29, 0.36, boxstyle='round,pad=0.015', fc='#eef4fb', ec='#1f4e79', lw=1.3))
    ax.text(x + 0.145, y + 0.25, t, ha='center', va='center', fontsize=8.5, weight='bold', color='#1f4e79')
    ax.text(x + 0.145, y + 0.09, d, ha='center', va='center', fontsize=7.2, color='#2c3e50')
arrows = [((0.31, 0.73), (0.35, 0.73)), ((0.64, 0.73), (0.68, 0.73)), ((0.825, 0.55), (0.825, 0.44)),
          ((0.68, 0.26), (0.64, 0.26)), ((0.35, 0.26), (0.31, 0.26)), ((0.165, 0.44), (0.165, 0.55))]
for a, b in arrows:
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='#1f4e79', lw=1.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig1_crispdm.png', dpi=300)
plt.close(fig)
print("[*] Figura 1 (CRISP-DM) generada exitosamente.", flush=True)

# FIGURA 2: Curvas de optimización (Silhouette, Calinski-Harabasz, Davies-Bouldin)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=300)

ks = [2, 3, 4, 5]
colors = {'K-Means': '#1f77b4', 'K-Medoids (PAM)': '#ff7f0e', 'HAC (Ward)': '#2ca02c', 'GMM (Full)': '#d62728', 'Spectral Clustering': '#9467bd'}

for model_name in modelos_b:
    sil_vals = [t['silhouette'] for t in tabla_modelos if t['esquema'] == 'Esquema B (Ingeniería Características)' and t['algoritmo'] == model_name]
    ax1.plot(ks, sil_vals, marker='o', lw=1.8, label=model_name, color=colors[model_name])

# Agregar línea base Esquema A K-Means
sil_a = [t['silhouette'] for t in tabla_modelos if t['esquema'] == 'Esquema A (35 vars crudas)' and t['algoritmo'] == 'K-Means']
ax1.plot(ks, sil_a, marker='s', lw=1.5, ls='--', label='Línea Base (35 vars crudas)', color='#7f7f7f')

ax1.axvline(2, ls=':', color='#333333', alpha=0.7)
ax1.axvline(3, ls=':', color='#333333', alpha=0.7)
ax1.set_xlabel('Número de Clústeres (k)', fontsize=8.5)
ax1.set_ylabel('Coeficiente de Silhouette (mayor es mejor)', fontsize=8.5)
ax1.set_title('(a) Coeficiente de Silhouette vs. k', fontsize=9.5, weight='bold')
ax1.set_xticks(ks)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(fontsize=7.2, frameon=True, loc='best')

# Panel (b): Davies-Bouldin en Esquema B
for model_name in modelos_b:
    db_vals = [t['davies_bouldin'] for t in tabla_modelos if t['esquema'] == 'Esquema B (Ingeniería Características)' and t['algoritmo'] == model_name]
    ax2.plot(ks, db_vals, marker='s', lw=1.8, label=model_name, color=colors[model_name])

db_a = [t['davies_bouldin'] for t in tabla_modelos if t['esquema'] == 'Esquema A (35 vars crudas)' and t['algoritmo'] == 'K-Means']
ax2.plot(ks, db_a, marker='o', lw=1.5, ls='--', label='Línea Base (35 vars crudas)', color='#7f7f7f')

ax2.axvline(2, ls=':', color='#333333', alpha=0.7)
ax2.axvline(3, ls=':', color='#333333', alpha=0.7)
ax2.set_xlabel('Número de Clústeres (k)', fontsize=8.5)
ax2.set_ylabel('Índice Davies-Bouldin (menor es mejor)', fontsize=8.5)
ax2.set_title('(b) Índice Davies-Bouldin vs. k', fontsize=9.5, weight='bold')
ax2.set_xticks(ks)
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(fontsize=7.2, frameon=True, loc='best')

fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig2_curvas_optimizacion.png', dpi=300)
plt.close(fig)
print("[*] Figura 2 (Curvas de optimización) generada exitosamente.", flush=True)

# FIGURA 3: Espacio Latente PCA (Panel a) y Firma de Centroides (Panel b)
pca_b = PCA(n_components=2, random_state=RS)
P_b = pca_b.fit_transform(Xb)
var_pca = pca_b.explained_variance_ratio_ * 100

fig, (ax_pca, ax_bar) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

cluster_colors = ['#1f77b4', '#2ca02c', '#d62728']
cluster_names = [
    'Clúster 0: Hegemonía JPP (Andino/Rural)',
    'Clúster 1: Hiperfragmentación (Centro/Urbano)',
    'Clúster 2: Competencia FP (Costa/Norte)'
]

for c in range(3):
    mask = (final_k3 == c)
    ax_pca.scatter(P_b[mask, 0], P_b[mask, 1], s=8, alpha=0.6, color=cluster_colors[c], label=cluster_names[c])

ax_pca.set_xlabel(f'Componente Principal 1 ({var_pca[0]:.1f} % var)', fontsize=8.5)
ax_pca.set_ylabel(f'Componente Principal 2 ({var_pca[1]:.1f} % var)', fontsize=8.5)
ax_pca.set_title(f'(a) Proyección PCA de los 1,890 distritos (k = 3)\nVarianza acumulada: {var_pca.sum():.1f} %', fontsize=9.5, weight='bold')
ax_pca.grid(True, linestyle=':', alpha=0.4)
ax_pca.legend(fontsize=7.0, loc='best', frameon=True)

# Panel (b): Perfil promedio de variables clave en barras horizontales
vars_plot = ['PCT_JUNTOS_POR_EL_PERÚ', 'PCT_FUERZA_POPULAR', 'PCT_PARTIDO_CÍVICO_OBRAS', 'PCT_AHORA_NACIÓN_-_AN', 'PCT_OTROS', 'MARGEN_VICTORIA']
names_plot = ['Juntos por el Perú', 'Fuerza Popular', 'Cívico Obras', 'Ahora Nación', 'Otros Partidos', 'Margen Victoria']
y_indices = np.arange(len(vars_plot))
height = 0.25

for c in range(3):
    sub = df[df['CLUSTER_K3'] == c]
    vals = [sub[v].mean() for v in vars_plot]
    ax_bar.barh(y_indices + (c - 1) * height, vals, height=height, color=cluster_colors[c], label=f'Clúster {c}')

ax_bar.set_yticks(y_indices)
ax_bar.set_yticklabels(names_plot, fontsize=8)
ax_bar.set_xlabel('Porcentaje Promedio (%)', fontsize=8.5)
ax_bar.set_title('(b) Firma electoral comparativa de los centroides', fontsize=9.5, weight='bold')
ax_bar.grid(True, linestyle=':', alpha=0.4, axis='x')
ax_bar.legend(fontsize=7.5, loc='lower right', frameon=True)

fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig3_espacio_latente.png', dpi=300)
plt.close(fig)
print("[*] Figura 3 (Espacio Latente y Firmas) generada exitosamente.", flush=True)

# FIGURA 4: Mapa Coroplético Distrital Nacional (k = 3)
with open('peru_distrital_simple.geojson', 'r', encoding='utf-8') as f:
    geo = json.load(f)

cluster_map_k3 = dict(zip(df['UBIGEO_INEI'].astype(str).str.zfill(6), df['CLUSTER_K3']))
patches_c = {0: [], 1: [], 2: [], 'nodata': []}

def add_geom_to_list(geom, t_list):
    if not geom or not isinstance(geom, dict):
        return
    t = geom.get('type')
    coords = geom.get('coordinates')
    if not coords:
        return
    polys = [coords] if t == 'Polygon' else coords if t == 'MultiPolygon' else []
    for poly in polys:
        if poly:
            pts = [(x, y) for x, y in poly[0]]
            t_list.append(Polygon(pts, closed=True))

emparejados = 0
for feat in geo['features']:
    geom = feat.get('geometry')
    props = feat.get('properties', {})
    iddist = str(props.get('IDDIST', '')).zfill(6)
    c = cluster_map_k3.get(iddist, None)
    if c in [0, 1, 2]:
        add_geom_to_list(geom, patches_c[c])
        emparejados += 1
    else:
        add_geom_to_list(geom, patches_c['nodata'])

print(f"[*] Polígonos de mapa emparejados con éxito: {emparejados} de {len(geo['features'])}", flush=True)

fig, ax = plt.subplots(figsize=(6, 8.5), dpi=300)
ax.set_aspect('equal')

coll_nd = PatchCollection(patches_c['nodata'], facecolor='#e6e6e6', edgecolor='#bbbbbb', linewidth=0.15)
coll_c0 = PatchCollection(patches_c[0], facecolor='#1f77b4', edgecolor='#0d3b66', linewidth=0.22)
coll_c1 = PatchCollection(patches_c[1], facecolor='#2ca02c', edgecolor='#134f13', linewidth=0.22)
coll_c2 = PatchCollection(patches_c[2], facecolor='#d62728', edgecolor='#7a1212', linewidth=0.22)

ax.add_collection(coll_nd)
ax.add_collection(coll_c0)
ax.add_collection(coll_c1)
ax.add_collection(coll_c2)

ax.set_xlim(-81.5, -68.5)
ax.set_ylim(-18.5, 0.2)
ax.set_xlabel('Longitud (°O)', fontsize=8)
ax.set_ylabel('Latitud (°S)', fontsize=8)
ax.set_title(f'Distribución territorial de clústeres electorales (K-Means, k=3)\nPrimera Vuelta Presidencial 2026 ({emparejados:,} de {n_distritos:,} distritos)',
             fontsize=9, weight='bold', pad=10)

legend_elements = [
    Line2D([0], [0], marker='s', color='w', label=f'Clúster 0: Hegemonía JPP (n = {perfiles_k3["Cluster_0"]["n"]})',
           markerfacecolor='#1f77b4', markersize=9, markeredgecolor='#0d3b66'),
    Line2D([0], [0], marker='s', color='w', label=f'Clúster 1: Hiperfragmentación (n = {perfiles_k3["Cluster_1"]["n"]})',
           markerfacecolor='#2ca02c', markersize=9, markeredgecolor='#134f13'),
    Line2D([0], [0], marker='s', color='w', label=f'Clúster 2: Competencia FP (n = {perfiles_k3["Cluster_2"]["n"]})',
           markerfacecolor='#d62728', markersize=9, markeredgecolor='#7a1212'),
    Line2D([0], [0], marker='s', color='w', label=f'Polígono sin correspondencia (n = {len(geo["features"]) - emparejados})',
           markerfacecolor='#e6e6e6', markersize=9, markeredgecolor='#bbbbbb')
]
ax.legend(handles=legend_elements, loc='lower left', frameon=True, fontsize=7.2,
          framealpha=0.92, facecolor='white', edgecolor='#cccccc')

ax.grid(True, linestyle=':', alpha=0.4, color='#999999')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig4_mapa_clusters.png', dpi=300)
plt.close(fig)
print("[*] Figura 4 (Mapa Coroplético Nacional) generada exitosamente.", flush=True)

# 7. GUARDAR RESULTADOS EN JSON PARA COMPILACIÓN
res_json = {
    'n_distritos': n_distritos,
    'n_formulas': n_formulas,
    'conteo_ganadores': conteo_ganadores,
    'empates': [dict(r) for _, r in df_empates.iterrows()],
    'tabla_modelos': tabla_modelos,
    'tabla_concordancia': tabla_concordancia,
    'perfiles_k3': perfiles_k3,
    'pca_var_acumulada': round(float(var_pca.sum()), 2),
    'pca_var_cp1': round(float(var_pca[0]), 2),
    'pca_var_cp2': round(float(var_pca[1]), 2),
    'poligonos_emparejados': emparejados
}

with open('resultados_paper_laccei_2026.json', 'w', encoding='utf-8') as f:
    json.dump(res_json, f, ensure_ascii=False, indent=2)

print("\n[+] PIPELINE CIENTÍFICO FINALIZADO CON ÉXITO TOTAL. Resultados exportados a resultados_paper_laccei_2026.json.", flush=True)
