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

a. HOG (Histogram of Oriented Gradients):
Descriptor de forma basado en la distribución de gradientes locales, útil para capturar bordes, estructuras anatómicas y patrones geométricos presentes en las radiografías.

b. LBP (Local Binary Patterns):
Método de textura que codifica relaciones espaciales locales entre intensidades vecinas. Ha demostrado eficacia en imágenes médicas por su robustez frente a cambios de iluminación.

c. GLCM (Gray Level Co-occurrence Matrix) / Características de Haralick:
Mide la coocurrencia de intensidades y permite caracterizar texturas mediante propiedades como homogeneidad, contraste y energía.

d. Momentos de Hu:
Conjunto de siete momentos invariantes a traslación, rotación y escala, tradicionalmente utilizados para describir la forma.

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

Segmentación (opcional): Se evaluó la pertinencia de aislar regiones anatómicamente relevantes, aunque dada la estructura del dataset se utiliza principalmente el preprocesamiento global.

El conjunto preprocesado se almacena en data/processed/ para su reutilización en las fases posteriores.

3. Extracción de Características Handcrafted

Se implementaron múltiples descriptores clásicos de forma y textura. Cada imagen preprocesada se transforma en un vector numérico mediante funciones desarrolladas en el módulo feature_extraction.py.
Luego, estos vectores conforman la matriz de características usada por los clasificadores tradicionales.

4. Modelos de Clasificación

A partir de las características extraídas, se entrenaron múltiples clasificadores:

SVM con kernels lineales y RBF

Random Forest

k-NN

En paralelo, se entrenaron modelos de Deep Learning:

CNN básicas diseñadas desde cero

Transfer Learning con arquitecturas preentrenadas

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
- Git (opcional, para clonar el repositorio)

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
│   ├── raw/                    # Dataset original
│   └── processed/              # Imágenes preprocesadas
├── src/
│   ├── utils.py               # Funciones auxiliares
│   └── preprocessing.py       # Pipeline de preprocesamiento
├── notebooks/
│   └── 01_preprocessing_exploration.ipynb  # Notebook Parte 1
├── results/
│   ├── figures/               # Gráficos generados
│   └── logs/                  # Logs de ejecución
├── tests/                     # Tests unitarios
├── README.md                  # Documentación principal
├── requirements.txt           # Dependencias
└── verificar_proyecto.py      # Script de verificación
```

---

## Ejecución del Proyecto

### Parte 1: Preprocesamiento y Exploración

```bash
# Activar entorno virtual (si no está activado)
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Abrir el notebook
jupyter notebook notebooks/01_preprocessing_exploration.ipynb
```

O abrir directamente el archivo `.ipynb` en VS Code con la extensión de Jupyter instalada.

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

- [x] Parte 1: Exploración y Preprocesamiento
- [ ] Parte 2: Extracción de Características Clásicas
- [ ] Parte 3: Clasificación con Métodos Clásicos
- [ ] Parte 4: Deep Learning
- [ ] Parte 5: Comparación y Análisis Final

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

Última actualización: Noviembre 2025