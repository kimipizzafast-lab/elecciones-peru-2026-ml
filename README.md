# Clusterización Territorial de Preferencias Electorales en el Perú 2026

Repositorio oficial del artículo científico:  
**«Clusterización territorial de las preferencias electorales en la primera vuelta presidencial del Perú 2026 aplicando algoritmos de Machine Learning no supervisado sobre datos de la ONPE»**

- **Autores:** Luis Miguel Tirabantes Córdova¹, Omar Santillán Aching²  
- **Institución:** Universidad Tecnológica del Perú (UTP), Lima, Perú  
- **Conferencia:** *24th LACCEI International Multi-Conference for Engineering, Education, and Technology (LACCEI 2026)*  
- **Área:** Ciencia de Datos, Inteligencia Artificial, Geocomputación Electoral e Ingeniería de Software

---

## 📌 Descripción del Proyecto

Este proyecto implementa un pipeline computacional automatizado, modular y reproducible bajo el marco **CRISP-DM** para analizar los patrones espaciales y electorales de la primera vuelta presidencial de las Elecciones Generales 2026 en el Perú. 

Abarca la totalidad de las **1,890 circunscripciones distritales** del país a partir de microdatos abiertos de la Oficina Nacional de Procesos Electorales (ONPE). El sistema resuelve de manera determinista las discrepancias históricas entre el estándar UBIGEO del INEI y la codificación electoral de RENIEC/ONPE, evaluando comparativamente cinco familias algorítmicas no supervisadas (**K-Means**, **K-Medoids / PAM**, **Clustering Jerárquico Aglomerativo de Ward**, **Modelos de Mezclas Gaussianas / GMM** y **Spectral Clustering**).

---

## 📁 Estructura del Repositorio

```text
├── dataset_onpe_1890_distritos.xlsx    # Matriz oficial homologada de 1,890 distritos
├── peru_distrital_simple.geojson       # Capa vectorial cartográfica de distritos del Perú
├── pipeline_electoral_peru_2026.py     # Script maestro del pipeline de clustering y evaluación
├── resultados_paper_laccei_2026.json   # Métricas y resultados numéricos del benchmarking
├── figuras/                            # Figuras en alta resolución generadas por el pipeline
├── requirements.txt                    # Dependencias de Python
├── LICENSE                             # Licencia de código abierto MIT
└── README.md                           # Documentación del proyecto
```

---

## ⚙️ Instalación y Requisitos

Se recomienda utilizar Python 3.10 o superior en un entorno virtual aislado:

```bash
# 1. Clonar el repositorio
git clone https://github.com/kimipizzafast-lab/elecciones-peru-2026-ml.git
cd elecciones-peru-2026-ml

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate   # En Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales (`requirements.txt`)
- `scikit-learn>=1.4.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scipy>=1.10.0`
- `matplotlib>=3.7.0`
- `openpyxl>=3.1.0`

---

## 🚀 Ejecución del Pipeline

Para reproducir el análisis completo, calcular las métricas de validación (Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI) y regenerar las figuras y tablas:

```bash
python pipeline_electoral_peru_2026.py
```

---

## 📊 Principales Resultados Cuantitativos

- **Universo evaluado:** 1,890 circunscripciones distritales (100.00 % de cobertura).
- **Esquema B (Ingeniería de Características):**
  - **Silhouette Score:** 0.2841 ($k=2$) / 0.2650 ($k=3$)
  - **Calinski-Harabasz:** 824.39 ($k=2$) / 702.34 ($k=3$)
  - **Davies-Bouldin:** 1.3214 ($k=2$) / 1.4868 ($k=3$)
  - **Varianza Explicada en PCA 2D:** 65.90 %
- **Tipología Tripartita Seleccionada ($k=3$):**
  1. *Clúster 0 (Hegemonía Andino-Rural, n = 672):* JPP 47.70 %, margen 32.70 pp, NEP 3.89.
  2. *Clúster 1 (Hiperfragmentación de Centro y Ciudades Intermedias, n = 604):* NEP 8.07, margen 6.80 pp.
  3. *Clúster 2 (Competitividad Urbano-Costera, n = 614):* FP 27.99 %, margen 13.69 pp, NEP 6.81.

---

## 📜 Licencia

Este proyecto se distribuye bajo la Licencia **MIT**. Consulta el archivo `LICENSE` para mayores detalles.
