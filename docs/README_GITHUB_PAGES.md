# Clasificación de Imágenes Médicas con Descriptores Clásicos y Deep Learning

> **Blog Post del Proyecto Académico**  
> Universidad Nacional de Colombia – Facultad de Minas  
> Visión por Computador – 3009228 | Semestre 2025-02

---

## 🏥 Acerca del Proyecto

Este proyecto desarrolla y compara sistemas de clasificación automática de radiografías de tórax para el diagnóstico de neumonía, utilizando tanto **descriptores clásicos** de forma y textura como arquitecturas de **redes neuronales convolucionales**.

### 👥 Equipo de Trabajo

- **David Londoño** - Preprocesamiento e Infraestructura
- **Andrés Churio** - Extracción de Descriptores
- **Sebastián Montoya Vargas** - Clasificación y Evaluación

---

## 📄 Documentación Completa

### Reporte Técnico Principal

👉 **[Ver Reporte Técnico Completo](reporte_tecnico_trabajo3.html)**

El reporte incluye:
- ✅ Introducción y motivación
- ✅ Marco teórico con fundamentos
- ✅ Metodología detallada del pipeline
- ✅ Experimentos y resultados con visualizaciones
- ✅ Análisis comparativo de métodos
- ✅ Conclusiones y trabajo futuro
- ✅ Referencias académicas (10 fuentes)

### Documentación Complementaria

- 📊 **[Diagramas de Flujo del Pipeline](docs/pipeline_diagram.md)** - Visualización gráfica del proceso completo
- 👤 **[Análisis de Contribución Individual](docs/contribucion_individual.md)** - Distribución de tareas y aprendizajes

---

## 🎯 Resultados Destacados

### Métricas de Clasificación

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **SVM RBF** | **95.51%** | **94.32%** | **98.46%** | **96.34%** |
| Random Forest | 94.87% | 93.10% | 98.21% | 95.58% |
| k-NN | 91.03% | 88.12% | 96.92% | 92.31% |
| Regresión Logística | 89.74% | 86.67% | 95.90% | 91.06% |

### Visualizaciones

<div align="center">

#### Matrices de Confusión
![Matrices de Confusión](results/figures/confusion_matrices.png)

#### Comparación de Métricas
![Comparación de Métricas](results/figures/metrics_comparison.png)

#### Curvas ROC
![Curvas ROC](results/figures/roc_curves.png)

</div>

---

## 🔬 Metodología

### Pipeline General

![Pipeline Overview](results/figures/pipeline_overview.png)

### Etapas del Proyecto

#### 1️⃣ **Preprocesamiento**
- Normalización de tamaño (224×224)
- Mejora de contraste con CLAHE
- Normalización de intensidades
- Segmentación opcional de ROI

![Preprocessing Steps](results/figures/preprocessing_steps.png)

#### 2️⃣ **Extracción de Descriptores**

**Descriptores de Forma:**
- HOG (Histogram of Oriented Gradients)
- Momentos de Hu
- Características de contorno

**Descriptores de Textura:**
- LBP (Local Binary Patterns)
- GLCM / Características de Haralick
- Filtros de Gabor
- Estadísticas de primer orden

![Feature Extraction](results/figures/feature_extraction_diagram.png)

#### 3️⃣ **Clasificación**
- Entrenamiento de múltiples clasificadores (SVM, RF, k-NN, LogReg)
- Búsqueda de hiperparámetros con GridSearchCV
- Evaluación con validación cruzada 5-fold

![Classification Workflow](results/figures/classification_workflow.png)

---

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** - Kaggle

- **Total:** 5,856 radiografías de tórax pediátricas
- **Clases:** NORMAL vs PNEUMONIA
- **División:**
  - Entrenamiento: 5,216 imágenes
  - Validación: 16 imágenes
  - Prueba: 624 imágenes

---

## 💻 Estructura del Repositorio

```
TRABAJO-03/
├── README.md                           # Este archivo
├── reporte_tecnico_trabajo3.md         # Reporte en Markdown
├── reporte_tecnico_trabajo3.html       # Reporte en HTML (Blog Post)
├── requirements.txt                    # Dependencias
│
├── data/                               # Dataset (no incluido en repo)
│   ├── raw/chest_xray/                # Imágenes originales
│   └── processed/                      # Imágenes preprocesadas
│
├── src/                                # Código fuente
│   ├── utils.py                        # Funciones auxiliares
│   ├── preprocessing.py                # Pipeline de preprocesamiento
│   └── descriptors.py                  # Extracción de características
│
├── notebooks/                          # Análisis exploratorio
│   ├── 01_preprocessing_exploration.ipynb
│   ├── 02_shape_and_texture_descriptors.ipynb
│   └── 03_Pipeline_Clasificacion.ipynb
│
├── results/                            # Resultados y figuras
│   ├── figures/                        # Visualizaciones generadas
│   └── logs/                           # Logs de experimentos
│
├── docs/                               # Documentación adicional
│   ├── pipeline_diagram.md             # Diagramas de flujo
│   └── contribucion_individual.md      # Análisis de contribución
│
└── scripts/                            # Scripts auxiliares
    └── generate_figures.py             # Generador de figuras
```

---

## 🚀 Cómo Ejecutar

### 1. Clonar el repositorio

```bash
git clone https://github.com/DavidALondono/TRABAJO-03.git
cd TRABAJO-03
```

### 2. Crear entorno virtual

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar dataset

El dataset debe descargarse manualmente de [Kaggle - Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) y colocarse en:

```
data/raw/chest_xray/
├── train/
├── val/
└── test/
```

### 5. Ejecutar notebooks

```bash
jupyter notebook notebooks/
```

Ejecutar en orden:
1. `01_preprocessing_exploration.ipynb`
2. `02_shape_and_texture_descriptors.ipynb`
3. `03_Pipeline_Clasificacion.ipynb`

---

## 📚 Tecnologías Utilizadas

- **Python 3.10+**
- **Procesamiento de Imágenes:** OpenCV, scikit-image
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow (preparado para futuro)
- **Visualización:** Matplotlib, Seaborn
- **Análisis:** NumPy, Pandas
- **Notebooks:** Jupyter

---

## 🎓 Referencias Principales

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
3. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer.
4. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*.
5. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE TPAMI*, 24(7), 971-987.

**[Ver lista completa de referencias en el reporte técnico](reporte_tecnico_trabajo3.html#referencias)**

---

## 🏆 Conclusiones Principales

1. **Efectividad de descriptores clásicos:** Los descriptores handcrafted combinados estratégicamente logran accuracy >95%, demostrando que siguen siendo competitivos.

2. **Superioridad de SVM RBF:** El clasificador SVM con kernel RBF obtuvo el mejor desempeño (95.51% accuracy, 96.34% F1-score).

3. **Importancia del preprocesamiento:** CLAHE resulta superior a la ecualización global de histograma para preservar estructuras anatómicas.

4. **Viabilidad clínica:** Los resultados son comparables a tasas de concordancia inter-observador de radiólogos (~90-95%).

5. **Trade-off interpretabilidad vs complejidad:** Los métodos clásicos ofrecen mayor interpretabilidad con desempeño cercano a métodos de deep learning.

---

## 📧 Contacto

- **Repositorio:** [github.com/DavidALondono/TRABAJO-03](https://github.com/DavidALondono/TRABAJO-03)
- **Curso:** Visión por Computador - 3009228
- **Universidad:** Universidad Nacional de Colombia – Facultad de Minas
- **Semestre:** 2025-02

---

## 📝 Licencia

Este proyecto es un trabajo académico desarrollado con fines educativos.

Dataset: [Kermany et al. (2018)](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) - Licencia CC BY 4.0

---

<div align="center">

**[🔝 Volver arriba](#clasificación-de-imágenes-médicas-con-descriptores-clásicos-y-deep-learning)**

---

*Última actualización: Diciembre 2025*

</div>
