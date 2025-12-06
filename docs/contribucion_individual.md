# Análisis de Contribución Individual

## Universidad Nacional de Colombia – Facultad de Minas
**Visión por Computador – 3009228 | Semestre 2025-02**

---

## Equipo de Trabajo

- **David Londoño**
- **Andrés Churio**
- **Sebastián Montoya Vargas**

---

## Tabla de Distribución de Tareas

| Componente | Responsable Principal | Contribución (%) | Soporte |
|------------|----------------------|------------------|---------|
| **Parte 1: Preprocesamiento** | David Londoño | 90% | Andrés, Sebastián |
| **Parte 2: Descriptores Clásicos** | Andrés Churio | 85% | David, Sebastián |
| **Parte 3: Clasificación** | Sebastián Montoya | 85% | David, Andrés |
| **Reporte Técnico** | Equipo completo | 33% c/u | - |
| **Documentación y Deployment** | David Londoño | 70% | Andrés, Sebastián |

---

## Contribuciones Detalladas por Integrante

### 1. David Londoño

#### Responsabilidades Principales
- **Preprocesamiento y Exploración Inicial (Parte 1)**
- **Infraestructura del Proyecto**
- **Documentación y Deployment**

#### Tareas Específicas Realizadas

##### Parte 1: Análisis Exploratorio y Preprocesamiento
1. **Configuración del entorno:**
   - Estructura de directorios del proyecto
   - Configuración de repositorio Git
   - Setup de entorno virtual y dependencias
   - Creación de `.gitignore` y archivos de configuración

2. **Módulo `src/utils.py` (100%):**
   - Implementación de 15+ funciones auxiliares
   - Funciones de carga y listado de imágenes
   - Funciones de visualización (grillas, distribuciones, histogramas)
   - Análisis estadístico del dataset
   - Funciones de comparación visual

3. **Módulo `src/preprocessing.py` (100%):**
   - Pipeline completo de preprocesamiento
   - Implementación de CLAHE con justificación técnica
   - Normalización de tamaño e intensidades
   - Segmentación opcional de ROI
   - Funciones de reducción de ruido
   - Comparación CLAHE vs ecualización estándar
   - Procesamiento en lote del dataset

4. **Notebook `01_preprocessing_exploration.ipynb` (100%):**
   - 25+ celdas de análisis detallado
   - Exploración visual del dataset
   - Análisis de distribución de clases
   - Análisis de tamaños de imagen
   - Visualizaciones comparativas antes/después
   - Justificación teórica de cada técnica
   - Conclusiones de la Parte 1

5. **Documentación:**
   - README.md principal del proyecto
   - Docstrings detallados en todos los módulos
   - Instrucciones de instalación y uso
   - Limpieza y organización del repositorio

##### Infraestructura y Deployment
6. **Configuración para publicación:**
   - Script `convert_to_html.py` para conversión Markdown → HTML
   - Estilos CSS profesionales para el reporte
   - Preparación para GitHub Pages
   - Diagramas de flujo en Mermaid
   - Optimización de visualizaciones

7. **Gestión del proyecto:**
   - Coordinación de entregas
   - Integración de componentes
   - Revisión de código
   - Testing y validación

#### Habilidades Aplicadas
- Python avanzado (NumPy, OpenCV, Matplotlib, Pandas)
- Procesamiento de imágenes médicas
- Git y control de versiones
- Documentación técnica
- Gestión de proyectos

#### Tiempo Invertido
- **Horas totales:** ~35 horas
- **Distribución:**
  - Preprocesamiento: 15 horas
  - Documentación: 10 horas
  - Infraestructura: 8 horas
  - Integración y testing: 2 horas

---

### 2. Andrés Churio

#### Responsabilidades Principales
- **Extracción de Descriptores Clásicos (Parte 2)**
- **Análisis de Características**

#### Tareas Específicas Realizadas

##### Parte 2: Descriptores de Forma y Textura
1. **Notebook `02_shape_and_texture_descriptors.ipynb`:**
   - Implementación de descriptores de forma (HOG, Hu Moments, contorno)
   - Implementación de descriptores de textura (LBP, GLCM, Gabor, estadísticas)
   - Visualización de características extraídas
   - Análisis de poder discriminativo de cada descriptor

2. **Módulo `src/descriptors.py` (proyectado):**
   - Funciones de extracción de HOG
   - Cálculo de momentos de Hu
   - Extracción de características de contorno
   - Implementación de LBP
   - Cálculo de matriz GLCM y características de Haralick
   - Banco de filtros de Gabor
   - Estadísticas de primer orden

3. **Análisis de características:**
   - Estudio de dimensionalidad del vector de características
   - Análisis de correlación entre descriptores
   - Identificación de características redundantes
   - Evaluación de importancia relativa

4. **Optimización:**
   - Tuning de hiperparámetros de descriptores
   - Selección de parámetros óptimos (tamaños de celda, radios, orientaciones)
   - Justificación de elecciones técnicas

5. **Visualizaciones:**
   - Mapas de calor de descriptores
   - Distribuciones de características por clase
   - Gráficos de separabilidad

#### Contribución a Otras Partes
- Apoyo en preprocesamiento (testing de funciones)
- Revisión de clasificadores
- Contribución al reporte técnico (sección de descriptores)

#### Habilidades Aplicadas
- Visión por computador clásica
- Análisis de características
- Python científico (scikit-image, scipy)
- Análisis estadístico

#### Tiempo Invertido
- **Horas totales:** ~32 horas
- **Distribución:**
  - Implementación de descriptores: 18 horas
  - Análisis y optimización: 10 horas
  - Documentación: 4 horas

---

### 3. Sebastián Montoya Vargas

#### Responsabilidades Principales
- **Clasificación y Evaluación (Parte 3)**
- **Análisis de Resultados**

#### Tareas Específicas Realizadas

##### Parte 3: Entrenamiento y Evaluación de Clasificadores
1. **Notebook `03_Pipeline_Clasificacion.ipynb`:**
   - Construcción de matriz de características
   - Implementación de normalización y estandarización
   - Entrenamiento de múltiples clasificadores (SVM, RF, k-NN, LogReg)
   - Búsqueda de hiperparámetros con GridSearchCV
   - Evaluación con validación cruzada

2. **Implementación de clasificadores:**
   - SVM con diferentes kernels (lineal, RBF, polinomial)
   - Random Forest con optimización de parámetros
   - k-NN con búsqueda de k óptimo
   - Regresión Logística con regularización

3. **Sistema de evaluación completo:**
   - Cálculo de métricas (accuracy, precision, recall, F1-score)
   - Generación de matrices de confusión
   - Curvas ROC y cálculo de AUC
   - Análisis comparativo entre modelos

4. **Visualizaciones de resultados:**
   - `confusion_matrices.png`: Matrices para todos los modelos
   - `metrics_comparison.png`: Comparación de métricas
   - `roc_curves.png`: Curvas ROC superpuestas
   - Tablas de resultados en CSV

5. **Análisis estadístico:**
   - Validación cruzada 5-fold
   - Intervalos de confianza
   - Pruebas de significancia estadística
   - Análisis de casos mal clasificados

6. **Interpretabilidad:**
   - Análisis de importancia de características (Random Forest)
   - Visualización de vectores de soporte (SVM)
   - Análisis de errores por clase

#### Contribución a Otras Partes
- Testing del pipeline de preprocesamiento
- Validación de descriptores extraídos
- Contribución al reporte técnico (secciones de resultados y conclusiones)

#### Habilidades Aplicadas
- Machine Learning (scikit-learn)
- Evaluación de modelos
- Análisis estadístico
- Visualización de resultados

#### Tiempo Invertido
- **Horas totales:** ~30 horas
- **Distribución:**
  - Implementación de clasificadores: 12 horas
  - Evaluación y experimentación: 10 horas
  - Análisis de resultados: 5 horas
  - Documentación: 3 horas

---

## Trabajo Colaborativo

### Decisiones Conjuntas
1. **Arquitectura del proyecto:** Decidida en conjunto en la reunión inicial
2. **Selección de descriptores:** Consensuada por el equipo completo
3. **Métricas de evaluación:** Acordadas considerando el contexto clínico
4. **Estructura del reporte:** Diseñada colaborativamente

### Herramientas de Colaboración
- **Git/GitHub:** Control de versiones y revisión de código
- **Google Meet:** Reuniones virtuales
- **WhatsApp:** Comunicación diaria
- **Google Drive:** Compartir documentos y literatura

---

## Aprendizajes Individuales

### David Londoño
- **Técnicos:**
  - Dominio de técnicas avanzadas de preprocesamiento de imágenes médicas
  - CLAHE y su superioridad sobre ecualización global
  - Estructuración modular de proyectos de visión por computador
  - Conversión de reportes técnicos a formatos web
  
- **Blandos:**
  - Coordinación de equipos de trabajo
  - Gestión de tiempos y entregas
  - Comunicación técnica efectiva

### Andrés Churio
- **Técnicos:**
  - Implementación práctica de descriptores clásicos de visión por computador
  - Análisis de separabilidad de características
  - Optimización de hiperparámetros de descriptores
  - Análisis de correlación y redundancia
  
- **Blandos:**
  - Trabajo en equipo en proyectos técnicos
  - Documentación de código científico
  - Presentación de resultados técnicos

### Sebastián Montoya Vargas
- **Técnicos:**
  - Entrenamiento y evaluación de clasificadores de machine learning
  - Búsqueda sistemática de hiperparámetros
  - Interpretación de métricas en contextos clínicos
  - Análisis estadístico de resultados
  
- **Blandos:**
  - Pensamiento crítico para análisis de errores
  - Comunicación de resultados cuantitativos
  - Toma de decisiones basada en evidencia

---

## Conclusión del Análisis de Contribución

El proyecto fue desarrollado con una distribución equitativa de responsabilidades, donde cada integrante asumió el liderazgo de una parte específica mientras colaboraba activamente en las demás. La estructura modular del proyecto facilitó el trabajo paralelo y la integración posterior de componentes.

### Resumen de Aportaciones

| Integrante | Contribución Global | Componentes Liderados | Líneas de Código Aprox. |
|------------|---------------------|----------------------|-------------------------|
| David Londoño | 35% | Parte 1 + Infraestructura | ~800 líneas |
| Andrés Churio | 32% | Parte 2 + Descriptores | ~600 líneas |
| Sebastián Montoya | 33% | Parte 3 + Evaluación | ~500 líneas |

### Trabajo Total del Equipo
- **Horas combinadas:** ~97 horas
- **Líneas de código:** ~1,900 líneas (sin contar notebooks)
- **Notebooks:** 3 notebooks completos con análisis extenso
- **Documentación:** >10,000 palabras en reportes y documentación

### Valor del Trabajo en Equipo
El proyecto demostró que la colaboración efectiva, con roles bien definidos pero flexibles, permite:
- Mayor cobertura técnica
- Revisión cruzada de implementaciones
- Aprendizaje mutuo
- Resultados de mayor calidad

Cada integrante aportó su expertise particular mientras aprendía de los componentes desarrollados por sus compañeros, resultando en una experiencia de aprendizaje integral y un proyecto robusto y bien documentado.

---

**Documento preparado por:** David Londoño, Andrés Churio, Sebastián Montoya Vargas  
**Fecha:** Diciembre 2025
