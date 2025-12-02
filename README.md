Clasificación de Imágenes Médicas con Descriptores Clásicos y Deep Learning

Universidad Nacional de Colombia – Facultad de Minas
Visión por Computador – 3009228
Semestre 2025-02
Autores: David Londoño, Andrés Churio, Sebastián Montoya

Introducción

El diagnóstico de neumonía mediante radiografías de tórax es una tarea fundamental en la práctica clínica, especialmente en poblaciones pediátricas donde la enfermedad presenta una alta incidencia y puede derivar en complicaciones severas si no se identifica de manera oportuna. La interpretación de estas imágenes requiere conocimiento especializado y, en escenarios con alta demanda asistencial o escasez de radiólogos, puede generar retrasos, inconsistencias y sobrecarga operativa.

En las últimas décadas, las técnicas de Visión por Computador han demostrado ser herramientas valiosas para apoyar procesos diagnósticos, al ofrecer métodos capaces de analizar imágenes médicas de forma objetiva, reproducible y eficiente. Este trabajo tiene como objetivo comparar dos aproximaciones para la clasificación de radiografías de tórax:

Métodos clásicos basados en descriptores manuales (handcrafted features).

Técnicas modernas de Deep Learning, particularmente arquitecturas convolucionales.

A partir de este análisis se busca evaluar el potencial, limitaciones y aplicabilidad práctica de cada enfoque en el contexto del análisis automatizado de imágenes médicas.

Marco Teórico
1. Clasificación de Imágenes Médicas

La clasificación automática de imágenes médicas consiste en asignar una etiqueta diagnóstica a una imagen a partir de patrones visuales. Este proceso puede apoyarse en enfoques tradicionales basados en extracción explícita de características, o en modelos de aprendizaje profundo capaces de aprender representaciones jerárquicas directamente desde los datos.

2. Descriptores Handcrafted

Los métodos clásicos se fundamentan en la representación explícita de características relevantes de la imagen. Entre los más utilizados se encuentran:

**Descriptores de Forma:**

a. HOG (Histogram of Oriented Gradients):
Descriptor basado en la distribución de gradientes locales, útil para capturar bordes y estructuras anatómicas.

b. Momentos de Hu:
Siete momentos invariantes a traslación, rotación y escala para describir la forma.

c. Descriptores de Contorno:
Área, perímetro y circularidad de las regiones detectadas, útiles para caracterizar la morfología pulmonar.

**Descriptores de Textura:**

d. LBP (Local Binary Patterns):
Codifica relaciones espaciales locales entre intensidades vecinas, robusto frente a cambios de iluminación.

e. GLCM (Gray Level Co-occurrence Matrix):
Calcula propiedades de coocurrencia: contraste, correlación, energía y homogeneidad.

f. Filtros de Gabor:
Banco de filtros con diferentes frecuencias y orientaciones para capturar texturas direccionales.

g. Estadísticas de Primer Orden:
Media, desviación estándar, percentiles y otras estadísticas básicas de intensidad.

3. Métodos Clásicos de Clasificación

Entre los clasificadores más empleados se encuentran:

SVM (Support Vector Machines): Modelos robustos para problemas de alta dimensión.

k-NN: Clasificador basado en la proximidad entre vectores de características.

Random Forest: Ensamble de árboles que ofrece interpretabilidad y buena capacidad de generalización.

4. Deep Learning y Redes Convolucionales

Las Convolutional Neural Networks (CNN) aprenden representaciones complejas directamente desde los píxeles, construyendo jerarquías de filtros que permiten capturar bordes, texturas y patrones semánticos. Su uso ha transformado el análisis de imágenes médicas gracias a su capacidad para extraer características discriminativas sin requerir ingeniería manual.

En este proyecto también se emplean técnicas de Transfer Learning, aprovechando pesos preentrenados en grandes bases de datos (p. ej., ImageNet) para acelerar la convergencia y mejorar el rendimiento con datasets médicos limitados.

Metodología

La metodología desarrollada se estructura en cinco fases principales, orientadas a reproducibilidad, claridad y separación modular del proceso.

1. Exploración del Dataset

Se realizó una inspección inicial del conjunto de radiografías mediante:

Conteo de muestras por clase en train, val y test.

Visualización de ejemplos representativos para entender variaciones en contraste, tamaño, ruido y posicionamiento.

Análisis preliminar del desbalanceo de clases.

Esta etapa permitió identificar la necesidad de normalización de imágenes y de técnicas que mitigaran diferencias de iluminación.

2. Preprocesamiento

Se implementó un pipeline que incluye:

Normalización de tamaño: Se seleccionó una resolución fija para asegurar uniformidad en los modelos clásicos y en las CNN.

Conversión a escala de grises (cuando aplica): Aunque el dataset ya está en un solo canal, se garantizó consistencia en el procesamiento.

CLAHE: Se aplicó Contrast Limited Adaptive Histogram Equalization, técnica recomendada para radiografías debido a su capacidad de mejorar el contraste sin amplificar excesivamente el ruido.

Segmentación: Se evaluó la pertinencia de aislar regiones anatómicamente relevantes, aunque dada la estructura del dataset se utiliza principalmente el preprocesamiento global.

El conjunto preprocesado se almacena en data/processed/ para su reutilización en las fases posteriores.

3. Extracción de Características Handcrafted

Se implementaron múltiples descriptores clásicos de forma y textura. Cada imagen preprocesada se transforma en un vector numérico mediante funciones desarrolladas en el módulo feature_extraction.py.
Luego, estos vectores conforman la matriz de características usada por los clasificadores tradicionales.

4. Modelos de Clasificación

A partir de las características extraídas, se entrenaron múltiples clasificadores clásicos:

- **SVM** con kernels lineales y RBF
- **Random Forest** (100 estimadores)
- **k-NN** (k=5 vecinos)
- **Logistic Regression** con regularización

En paralelo, se entrenó un modelo de Deep Learning:

- **CNN mejorada** con BatchNormalization, Dropout y callbacks (EarlyStopping, ReduceLROnPlateau)
- Arquitectura ligera pero efectiva con 3 bloques convolucionales
- Data augmentation opcional mediante ImageDataGenerator

La evaluación se llevó a cabo utilizando métricas estandarizadas y validación cruzada cuando aplica.

5. Comparación y Análisis

Los resultados de los enfoques clásicos y de Deep Learning se compararon considerando:

Rendimiento global

Sensibilidad y especificidad (relevantes en contextos médicos)

Robustez frente a variaciones visuales

Interpretabilidad

Esta etapa permitió discutir el impacto de los descriptores manuales versus el aprendizaje de características automáticas.

---

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (para clonar el repositorio)

---

## Instalación

### Configuración en Windows

Ejecute los siguientes comandos en el terminal (CMD o PowerShell):

```bash
# 1. Navegar al directorio del proyecto
cd TRABAJO-03

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
.venv\Scripts\activate

# 4. Actualizar pip
python -m pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt
```

### Configuración en macOS y Linux

Ejecute los siguientes comandos en el terminal:

```bash
# 1. Navegar al directorio del proyecto
cd TRABAJO-03

# 2. Crear entorno virtual
python3 -m venv .venv

# 3. Activar entorno virtual
source .venv/bin/activate

# 4. Actualizar pip
pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales

- `opencv-python>=4.8.0` - Procesamiento de imágenes
- `numpy>=1.24.0` - Cálculos numéricos
- `pandas>=2.0.0` - Manipulación de datos
- `matplotlib>=3.7.0` - Visualización
- `seaborn>=0.12.0` - Visualización estadística
- `scikit-image>=0.21.0` - Procesamiento de imágenes
- `scikit-learn>=1.3.0` - Machine Learning
- `tensorflow>=2.13.0` - Deep Learning
- `jupyter>=1.0.0` - Notebooks interactivos

---

## Descarga del Dataset

El dataset debe descargarse manualmente desde Kaggle:

1. Visitar: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Descargar el archivo ZIP
3. Extraer el contenido en la carpeta `data/raw/`
4. La estructura debe quedar: `data/raw/chest_xray/train/`, `data/raw/chest_xray/val/`, `data/raw/chest_xray/test/`

Para más detalles, consultar el archivo `GUIA_DESCARGA_DATASET.md`

---

## Estructura del Proyecto

```
TRABAJO-03/
├── data/
│   └── raw/
│       └── chest_xray/        # Dataset de Kaggle
│           ├── train/
│           │   ├── NORMAL/
│           │   └── PNEUMONIA/
│           ├── val/
│           │   ├── NORMAL/
│           │   └── PNEUMONIA/
│           └── test/
│               ├── NORMAL/
│               └── PNEUMONIA/
├── src/
│   ├── __init__.py
│   ├── descriptors.py         # Descriptores de forma y textura
│   ├── preprocessing.py       # Pipeline de preprocesamiento
│   └── utils.py               # Funciones auxiliares
├── notebooks/
│   ├── 01_preprocessing_exploration.ipynb   # Parte 1: Exploración
│   ├── 02_shape_and_texture_descriptors.ipynb  # Parte 2: Descriptores
│   └── 03_Pipeline_Clasificacion.ipynb      # Parte 3: Clasificación completa
├── results/
│   ├── figures/               # Gráficos generados (matrices de confusión, ROC)
│   ├── logs/                  # Logs de ejecución
│   └── comparison_summary.csv # Resultados de comparación de modelos
├── tests/                     # Tests unitarios
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias del proyecto
└── reporte_tecnico_trabajo3.md  # Reporte técnico detallado
```

---

## Ejecución del Proyecto

Los notebooks deben ejecutarse en orden:

### Parte 1: Preprocesamiento y Exploración

```bash
# Activar entorno virtual (si no está activado)
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Abrir el notebook
jupyter notebook notebooks/01_preprocessing_exploration.ipynb
```

### Parte 2: Extracción de Descriptores

```bash
jupyter notebook notebooks/02_shape_and_texture_descriptors.ipynb
```

### Parte 3: Pipeline de Clasificación Completo

```bash
jupyter notebook notebooks/03_Pipeline_Clasificacion.ipynb
```

**Nota:** Los notebooks 01 y 02 utilizan módulos de `src/` para reutilizar código. El notebook 03 es autocontenido y puede ejecutarse independientemente.

#### Ejecución del pipeline de clasificación:

1. Asegúrate de tener el dataset en `data/raw/chest_xray/`
2. Ejecuta las celdas en orden
3. Cuando llegues a la sección 9, ejecuta: `example_workflow()`
4. Los resultados se guardarán automáticamente en `results/`

O abrir directamente los archivos `.ipynb` en VS Code con la extensión de Jupyter instalada.

---

## Verificación de la Instalación

Para verificar que todo está correctamente configurado:

```bash
python verificar_proyecto.py
```

Este script verificará:
- Estructura de directorios
- Archivos principales
- Dependencias instaladas
- Disponibilidad del dataset

---

## Estado del Proyecto

- [x] Parte 1: Exploración y Preprocesamiento (Notebook 01)
- [x] Parte 2: Extracción de Descriptores de Forma y Textura (Notebook 02)
- [x] Parte 3: Pipeline Completo de Clasificación (Notebook 03)
  - [x] Descriptores clásicos (HOG, Hu, Contorno, LBP, GLCM, Gabor)
  - [x] Clasificadores tradicionales (SVM, Random Forest, k-NN, LogReg)
  - [x] CNN mejorada con BatchNormalization
  - [x] Métricas completas (Accuracy, Precision, Recall, F1, Matriz de Confusión, ROC/AUC)
  - [x] Comparación entre descriptores (Forma vs Textura vs Combinado)
- [ ] Reporte técnico final
- [ ] Publicación en plataforma de blogging

---

## Autores

- David Londoño
- Andrés Churio
- Sebastián Montoya

**Universidad Nacional de Colombia – Facultad de Minas**  
**Curso:** Visión por Computador (3009228)  
**Semestre:** 2025-02

---

## Referencias

- Kermany, D. S., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." Cell, 172(5), 1122-1131.
- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) - Kaggle
- Documentación de OpenCV: https://docs.opencv.org/
- Documentación de scikit-image: https://scikit-image.org/

---

Última actualización: Diciembre 2025