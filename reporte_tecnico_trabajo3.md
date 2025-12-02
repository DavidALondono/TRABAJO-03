# Clasificación de Imágenes Médicas con Descriptores Clásicos y Deep Learning

---

**Universidad Nacional de Colombia – Facultad de Minas**  
**Visión por Computador – 3009228**  
**Semestre 2025-02**

**Autores:**  
- David Londoño  
- Andrés Churio  
- Sebastián Montoya

**Fecha:** Diciembre 2025

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Marco Teórico](#marco-teórico)
3. [Metodología](#metodología)
   - 3.1 [Parte 1: Análisis Exploratorio y Preprocesamiento](#parte-1-análisis-exploratorio-y-preprocesamiento)
   - 3.2 [Parte 2: Extracción de Descriptores Clásicos](#parte-2-extracción-de-descriptores-clásicos)
   - 3.3 [Parte 3: Clasificación](#parte-3-clasificación)
4. [Resultados y Discusión](#resultados-y-discusión)
5. [Conclusiones](#conclusiones)
6. [Referencias](#referencias)

---

## Introducción

El diagnóstico de neumonía mediante radiografías de tórax es una tarea fundamental en la práctica clínica, especialmente en poblaciones pediátricas donde la enfermedad presenta una alta incidencia y puede derivar en complicaciones severas si no se identifica de manera oportuna. La interpretación de estas imágenes requiere conocimiento especializado y, en escenarios con alta demanda asistencial o escasez de radiólogos, puede generar retrasos, inconsistencias y sobrecarga operativa.

En las últimas décadas, las técnicas de Visión por Computador han demostrado ser herramientas valiosas para apoyar procesos diagnósticos, al ofrecer métodos capaces de analizar imágenes médicas de forma objetiva, reproducible y eficiente. Este trabajo tiene como objetivo comparar dos aproximaciones para la clasificación de radiografías de tórax:

- **Métodos clásicos** basados en descriptores manuales (handcrafted features).
- **Técnicas modernas de Deep Learning**, particularmente arquitecturas convolucionales (CNN).

A partir de este análisis se busca evaluar el potencial, limitaciones y aplicabilidad práctica de cada enfoque en el contexto del análisis automatizado de imágenes médicas.

### Objetivos

#### Objetivo General

Desarrollar y comparar sistemas de clasificación automática de radiografías de tórax para el diagnóstico de neumonía, utilizando tanto descriptores clásicos de forma y textura como arquitecturas de redes neuronales convolucionales.

#### Objetivos Específicos

1. Implementar un pipeline de preprocesamiento adecuado para radiografías de tórax que permita estandarizar las imágenes y mejorar la calidad visual.

2. Extraer y evaluar descriptores clásicos de forma y textura (HOG, LBP, GLCM, momentos de Hu, descriptores de Fourier, filtros de Gabor) para caracterizar las radiografías.

3. Entrenar y comparar clasificadores tradicionales (SVM, Random Forest, k-NN, Regresión Logística) utilizando las características extraídas.

4. Implementar y entrenar una arquitectura CNN para la clasificación de las radiografías.

5. Comparar el desempeño de ambos enfoques en términos de métricas de clasificación (exactitud, precisión, recall, F1-score) y analizar las ventajas y desventajas de cada metodología.

---

## Marco Teórico

### Clasificación de Imágenes Médicas

La clasificación automática de imágenes médicas consiste en asignar una etiqueta diagnóstica a una imagen a partir de patrones visuales identificables. Este proceso puede apoyarse en enfoques tradicionales basados en extracción explícita de características diseñadas por expertos, o en modelos de aprendizaje profundo capaces de aprender representaciones jerárquicas directamente desde los datos de manera automática.

En el contexto de radiografías de tórax, los patrones relevantes incluyen opacidades, infiltrados, consolidaciones y otras anomalías asociadas con neumonía que se manifiestan como cambios en la textura y distribución de intensidades.

### Preprocesamiento de Imágenes Médicas

El preprocesamiento es una etapa crítica que busca estandarizar las imágenes y mejorar su calidad antes de la extracción de características o el entrenamiento de modelos. Las técnicas comunes incluyen:

- **Normalización de tamaño:** Redimensionar todas las imágenes a dimensiones consistentes para facilitar el procesamiento en lotes.
- **Mejora de contraste:** Métodos como CLAHE (Contrast Limited Adaptive Histogram Equalization) que mejoran el contraste local sin amplificar ruido excesivamente.
- **Normalización de intensidades:** Escalar los valores de píxeles a rangos estándar para reducir variabilidad por condiciones de adquisición.
- **Segmentación de ROI:** Aislar la región pulmonar relevante para eliminar información no diagnóstica.

### Descriptores Clásicos

Los métodos clásicos se fundamentan en la extracción manual de características que capturan propiedades geométricas y texturales de las imágenes:

#### Descriptores de Forma

- **HOG (Histogram of Oriented Gradients):** Codifica la distribución de orientaciones de gradientes locales, útil para capturar estructuras y bordes.
- **Momentos de Hu:** Siete momentos invariantes a traslación, rotación y escala que describen propiedades geométricas de la forma.
- **Descriptores de contorno:** Características derivadas de los contornos detectados (área, perímetro, circularidad, excentricidad).
- **Descriptores de Fourier:** Representación en el dominio de la frecuencia para caracterizar formas complejas.

#### Descriptores de Textura

- **LBP (Local Binary Patterns):** Codifica patrones de textura local mediante comparaciones entre píxeles vecinos, robusto a cambios de iluminación.
- **GLCM (Gray Level Co-occurrence Matrix):** Matriz de coocurrencia que permite calcular características de Haralick (contraste, homogeneidad, energía, correlación) para caracterizar texturas.
- **Filtros de Gabor:** Conjunto de filtros orientados en múltiples frecuencias y orientaciones que capturan información de textura direccional.
- **Estadísticas de primer orden:** Media, desviación estándar, asimetría y curtosis de las distribuciones de intensidad.

### Clasificadores Tradicionales

Entre los clasificadores más empleados en problemas de clasificación con descriptores manuales se encuentran:

- **SVM (Support Vector Machines):** Modelos robustos para problemas de alta dimensión que buscan el hiperplano óptimo de separación.
- **Random Forest:** Ensamble de árboles de decisión que ofrece interpretabilidad y buena capacidad de generalización.
- **k-NN (k-Nearest Neighbors):** Clasificador basado en la proximidad entre vectores de características en el espacio de representación.
- **Regresión Logística:** Modelo lineal probabilístico que establece fronteras de decisión lineales.

### Redes Neuronales Convolucionales (CNN)

Las CNNs han revolucionado el campo de la visión por computador al aprender representaciones jerárquicas directamente desde los datos. En lugar de diseñar características manualmente, las CNNs aprenden filtros convolucionales que detectan patrones de bajo nivel (bordes, texturas) en capas iniciales y características más abstractas (patrones específicos de enfermedades) en capas profundas. Arquitecturas populares como ResNet, VGG y EfficientNet han demostrado desempeño superior en tareas de clasificación de imágenes médicas.

---

## Metodología

### Parte 1: Análisis Exploratorio y Preprocesamiento

Esta sección describe el trabajo ya implementado en el repositorio, correspondiente a la exploración inicial del dataset y el diseño del pipeline de preprocesamiento.

#### Dataset Utilizado

Se utilizó el dataset **Chest X-Ray Images (Pneumonia)** disponible en Kaggle, que contiene radiografías de tórax pediátricas organizadas en dos clases:

- **NORMAL:** Radiografías de pacientes sin neumonía.
- **PNEUMONIA:** Radiografías de pacientes diagnosticados con neumonía (bacteriana o viral).

El dataset está dividido en tres conjuntos:

- **Entrenamiento (train):** 5,216 imágenes (1,341 normales + 3,875 con neumonía).
- **Validación (val):** 16 imágenes (8 normales + 8 con neumonía).
- **Prueba (test):** 624 imágenes (234 normales + 390 con neumonía).

**Análisis de distribución:** Se observó un desbalance significativo en el conjunto de entrenamiento, con aproximadamente 74% de imágenes con neumonía y 26% normales. Este desbalance debe considerarse durante el entrenamiento mediante técnicas como pesos de clase balanceados o data augmentation.

#### Exploración Visual

Se cargaron y visualizaron muestras representativas de ambas clases utilizando grillas de imágenes. La exploración visual permitió identificar:

- **Variabilidad en calidad de imagen:** Diferencias en exposición, contraste y nitidez entre radiografías.
- **Variabilidad en posicionamiento:** Ligeras rotaciones y desplazamientos del paciente.
- **Patrones diagnósticos:** Las radiografías con neumonía presentan opacidades difusas, consolidaciones o infiltrados que alteran la textura pulmonar normal.

Estas observaciones justificaron la necesidad de un preprocesamiento robusto para estandarizar las imágenes antes del análisis cuantitativo.

#### Análisis de Tamaños de Imagen

Se analizaron las dimensiones originales de las imágenes en el conjunto de entrenamiento, encontrando una gran variabilidad:

- **Dimensiones:** Las imágenes tienen tamaños diversos, con anchos y altos que varían considerablemente.
- **Relación de aspecto:** La mayoría de las radiografías mantienen relaciones de aspecto similares, pero no idénticas.

Esta heterogeneidad dimensional refuerza la necesidad de normalizar el tamaño de todas las imágenes a una dimensión estándar (224×224 píxeles) para permitir el procesamiento en lotes y la compatibilidad con arquitecturas CNN preentrenadas.

#### Pipeline de Preprocesamiento Implementado

El pipeline completo de preprocesamiento se encuentra implementado en el módulo `src/preprocessing.py` e incluye las siguientes etapas:

##### 1. Normalización de Tamaño

**Técnica:** Redimensionamiento a 224×224 píxeles utilizando interpolación `INTER_AREA`.

**Justificación:**
- Las redes neuronales convolucionales requieren tamaños de entrada fijos.
- 224×224 es un estándar ampliamente utilizado en arquitecturas CNN preentrenadas (VGG, ResNet, EfficientNet).
- La interpolación `INTER_AREA` es óptima para reducir el tamaño de imagen preservando información visual.

**Implementación:** Función `resize_image()` en `preprocessing.py`.

##### 2. Mejora de Contraste con CLAHE

**Técnica:** CLAHE (Contrast Limited Adaptive Histogram Equalization) con parámetros `clip_limit=2.0` y `tile_grid_size=(8,8)`.

**Justificación:**
- Las radiografías presentan variaciones de exposición que pueden ocultar estructuras relevantes.
- CLAHE mejora el contraste local adaptativamente sin exagerar el ruido.
- La limitación del clip evita la amplificación excesiva en regiones homogéneas.
- Es especialmente útil para resaltar infiltrados y opacidades sutiles característicos de neumonía.

**Comparación con ecualización de histograma estándar:** Se implementó una comparación visual entre CLAHE y ecualización global de histograma. Los resultados mostraron que CLAHE preserva mejor las estructuras anatómicas finas sin saturar regiones, mientras que la ecualización global tiende a amplificar ruido y producir artefactos en áreas homogéneas.

**Implementación:** Función `apply_clahe()` en `preprocessing.py`.

##### 3. Normalización de Intensidades

**Técnica:** Escalado de valores de píxeles al rango [0, 1].

**Justificación:**
- Reduce la variabilidad artificial causada por diferencias en los equipos de adquisición.
- Facilita la convergencia durante el entrenamiento de modelos de aprendizaje automático.
- Permite comparaciones consistentes entre imágenes.

**Implementación:** Función `normalize_intensity()` en `preprocessing.py`.

##### 4. Segmentación de Región de Interés (Opcional)

**Técnica:** Segmentación basada en umbralización de Otsu y operaciones morfológicas para aislar la región pulmonar.

**Justificación:**
- Elimina información no diagnóstica (fondo, etiquetas, artefactos externos).
- Enfoca el análisis exclusivamente en el área pulmonar relevante.
- Puede mejorar la calidad de las características extraídas al reducir ruido contextual.

**Nota:** Esta técnica es opcional y se aplicó en algunos experimentos para evaluar su impacto en el desempeño. Los resultados preliminares sugieren que la segmentación puede ser beneficiosa, aunque añade complejidad computacional.

**Implementación:** Función `segment_lung_region()` en `preprocessing.py`.

##### 5. Reducción de Ruido (Opcional)

**Técnica:** Filtro bilateral y/o filtro gaussiano.

**Justificación:**
- Las radiografías digitales pueden contener ruido electrónico o artefactos de compresión.
- El filtro bilateral preserva bordes mientras suaviza regiones homogéneas.

**Implementación:** Función `denoise_image()` en `preprocessing.py`.

#### Visualizaciones Generadas

Durante la exploración y validación del preprocesamiento se generaron múltiples visualizaciones:

- **Figura 1:** Grilla de imágenes originales de ambas clases (NORMAL vs PNEUMONIA).
- **Figura 2:** Distribución de clases en los conjuntos de entrenamiento, validación y prueba.
- **Figura 3:** Distribución de tamaños originales de las imágenes (ancho vs alto).
- **Figura 4:** Comparación visual de imágenes originales vs preprocesadas (resize + CLAHE + normalización).
- **Figura 5:** Comparación entre CLAHE y ecualización de histograma estándar.
- **Figura 6:** Histogramas de intensidades antes y después del preprocesamiento.
- **Figura 7:** Ejemplo de segmentación de ROI pulmonar.

Estas figuras se encuentran disponibles en el notebook `notebooks/01_preprocessing_exploration.ipynb` y demuestran visualmente el impacto positivo del preprocesamiento en la calidad de las radiografías.

#### Conclusiones de la Parte 1

1. **Desbalance de clases:** El conjunto de entrenamiento presenta un desbalance significativo que debe abordarse mediante técnicas apropiadas durante el entrenamiento de modelos.

2. **Variabilidad dimensional:** La heterogeneidad en los tamaños de imagen justifica plenamente la normalización a dimensiones estándar.

3. **Impacto de CLAHE:** La mejora de contraste mediante CLAHE resulta visualmente superior a la ecualización global de histograma, preservando mejor las estructuras anatómicas relevantes.

4. **Pipeline robusto:** El pipeline implementado (resize → CLAHE → normalización) proporciona imágenes estandarizadas de calidad mejorada, adecuadas para la extracción de descriptores y el entrenamiento de modelos.

5. **Base sólida:** La exploración y preprocesamiento realizados establecen una base sólida para las fases subsiguientes del proyecto (extracción de características y clasificación).

---

### Parte 2: Extracción de Descriptores Clásicos

**NOTA:** Esta sección está en desarrollo por el equipo. A continuación se presenta la estructura planificada.

#### Introducción

En esta fase se extraerán descriptores clásicos de forma y textura de las radiografías preprocesadas. Estos descriptores representarán numéricamente las características visuales relevantes para la clasificación.

#### Descriptores de Forma

Se implementarán los siguientes descriptores para capturar información geométrica y estructural:

##### HOG (Histogram of Oriented Gradients)

**TODO:** Describir implementación concreta, parámetros utilizados (tamaño de celda, bins de orientación, normalización) y justificación de elecciones.

**Parámetros preliminares sugeridos:**
- Tamaño de celda: (8, 8)
- Tamaño de bloque: (2, 2)
- Bins de orientación: 9
- Normalización: L2-Hys

##### Momentos de Hu

**TODO:** Describir implementación, preprocesamiento de la imagen (umbralización para obtener silueta), cálculo de los 7 momentos invariantes y análisis de su poder discriminativo.

##### Descriptores de Contorno

**TODO:** Describir extracción de contornos (método de detección), cálculo de características derivadas:
- Área del contorno principal
- Perímetro
- Circularidad
- Excentricidad
- Solidez

##### Descriptores de Fourier

**TODO:** Describir transformada de Fourier del contorno, selección de coeficientes representativos y justificación de su uso para capturar propiedades globales de la forma.

#### Descriptores de Textura

Se implementarán descriptores diseñados para caracterizar patrones de textura en las radiografías:

##### LBP (Local Binary Patterns)

**TODO:** Describir implementación (radio, número de puntos, variante utilizada: uniforme, rotacionalmente invariante), construcción del histograma de LBP y justificación de parámetros.

##### GLCM (Gray Level Co-occurrence Matrix) y Características de Haralick

**TODO:** Describir cálculo de la matriz de coocurrencia (direcciones, distancias), extracción de características de Haralick:
- Contraste
- Homogeneidad
- Energía
- Correlación
- Entropía

##### Filtros de Gabor

**TODO:** Describir banco de filtros de Gabor (frecuencias, orientaciones), aplicación sobre las imágenes preprocesadas, cálculo de estadísticas (media, desviación estándar) de las respuestas filtradas.

##### Estadísticas de Primer Orden

**TODO:** Describir cálculo de estadísticas básicas de la distribución de intensidades:
- Media
- Desviación estándar
- Asimetría (skewness)
- Curtosis

#### Construcción del Vector de Características

**TODO:** Describir concatenación de todos los descriptores en un único vector de características por imagen, dimensionalidad final del vector y técnicas de normalización aplicadas.

---

### Parte 3: Clasificación

**NOTA:** Esta sección está en desarrollo por el equipo. A continuación se presenta la estructura planificada.

#### Introducción

En esta fase se entrenarán y evaluarán múltiples clasificadores utilizando las características extraídas, así como una arquitectura CNN que aprenderá representaciones directamente desde las imágenes preprocesadas.

#### Construcción de la Matriz de Características

**TODO:** Describir proceso de construcción de la matriz de características para el conjunto de datos completo (train, val, test), manejo de valores faltantes o inválidos, y formato final de los datos.

#### Normalización y Reducción de Dimensionalidad

**TODO:** Describir técnicas de normalización aplicadas (StandardScaler, MinMaxScaler), análisis de dimensionalidad del vector de características y aplicación opcional de reducción de dimensionalidad (PCA, selección de características) para mejorar eficiencia y desempeño.

#### Clasificadores Tradicionales

Se entrenarán y compararán los siguientes clasificadores:

##### SVM (Support Vector Machines)

**TODO:** Describir kernels probados (lineal, RBF, polinomial), búsqueda de hiperparámetros (C, gamma), y estrategia de validación.

##### Random Forest

**TODO:** Describir número de árboles, profundidad máxima, criterio de división, y búsqueda de hiperparámetros óptimos.

##### k-NN (k-Nearest Neighbors)

**TODO:** Describir valores de k evaluados, métricas de distancia (euclidiana, Manhattan), y análisis de sensibilidad al valor de k.

##### Regresión Logística

**TODO:** Describir regularización aplicada (L1, L2, ElasticNet), parámetro de regularización, y análisis de los coeficientes aprendidos.

#### Arquitectura CNN

**TODO:** Describir arquitectura convolucional implementada:
- Diseño desde cero vs transfer learning (VGG16, ResNet50, EfficientNet)
- Número y configuración de capas convolucionales
- Funciones de activación
- Pooling y dropout
- Capas fully connected
- Función de pérdida y optimizador
- Técnicas de regularización (data augmentation, early stopping)

#### Esquema de Validación

**TODO:** Describir estrategia de validación utilizada:
- División train/val/test
- Validación cruzada (si aplica)
- Métricas de evaluación: exactitud, precisión, recall, F1-score, AUC-ROC, matriz de confusión
- Manejo del desbalance de clases (pesos balanceados, SMOTE, data augmentation)

#### Análisis Comparativo

**TODO:** Describir comparación cuantitativa entre todos los modelos, análisis de fortalezas y debilidades de cada enfoque, y discusión sobre interpretabilidad vs desempeño.

---

## Resultados y Discusión

### Resultados de la Parte 1: Preprocesamiento

Los resultados de la exploración y preprocesamiento se resumen a continuación:

#### Análisis del Dataset

- **Total de imágenes:** 5,856 radiografías de tórax pediátricas.
- **Distribución en entrenamiento:** Desbalance significativo con 74% de casos con neumonía.
- **Variabilidad dimensional:** Tamaños originales heterogéneos que requieren normalización.

#### Impacto del Preprocesamiento

El pipeline de preprocesamiento demostró mejoras visuales significativas:

1. **CLAHE vs Ecualización Global:** CLAHE preserva mejor las estructuras anatómicas finas (costillas, vasos pulmonares, infiltrados) sin introducir artefactos excesivos. La ecualización global tiende a saturar regiones y amplificar ruido.

2. **Normalización de tamaño:** El redimensionamiento a 224×224 mantuvo la información visual relevante sin distorsiones perceptibles.

3. **Estandarización exitosa:** Las imágenes preprocesadas presentan intensidades normalizadas y contraste mejorado, facilitando análisis subsiguientes.

#### Conclusiones Parciales

El preprocesamiento implementado es fundamental para:
- Reducir variabilidad artificial entre radiografías.
- Mejorar la visibilidad de patrones diagnósticos sutiles.
- Estandarizar las imágenes para compatibilidad con modelos de aprendizaje automático.

La calidad visual mejorada de las imágenes preprocesadas sugiere que la extracción de descriptores y el entrenamiento de modelos se beneficiarán significativamente de esta etapa inicial.

### Resultados de la Parte 2: Descriptores Clásicos

**TODO:** Insertar tablas y análisis cuando se tengan resultados de extracción de características.

**Análisis pendiente:**
- Distribución de valores de cada descriptor.
- Análisis de correlación entre descriptores.
- Importancia relativa de cada tipo de descriptor.
- Visualización de poder discriminativo (t-SNE, PCA).

### Resultados de la Parte 3: Clasificación

**TODO:** Insertar tablas de métricas y análisis comparativo cuando estén listos los resultados de clasificadores.

**Métricas esperadas:**

| Modelo              | Exactitud | Precisión | Recall | F1-Score | AUC-ROC |
|---------------------|-----------|-----------|--------|----------|---------|
| SVM (RBF)           | TODO      | TODO      | TODO   | TODO     | TODO    |
| Random Forest       | TODO      | TODO      | TODO   | TODO     | TODO    |
| k-NN                | TODO      | TODO      | TODO   | TODO     | TODO    |
| Regresión Logística | TODO      | TODO      | TODO   | TODO     | TODO    |
| CNN                 | TODO      | TODO      | TODO   | TODO     | TODO    |

**Análisis pendiente:**
- Matrices de confusión para cada modelo.
- Curvas ROC comparativas.
- Análisis de casos mal clasificados.
- Comparación de tiempos de entrenamiento e inferencia.
- Discusión sobre trade-offs entre interpretabilidad y desempeño.

---

## Conclusiones

### Conclusiones Actuales (Parte 1)

1. **Importancia del preprocesamiento:** El pipeline implementado (normalización de tamaño, CLAHE, normalización de intensidades) es fundamental para estandarizar radiografías heterogéneas y mejorar la calidad visual, facilitando análisis posteriores.

2. **Superioridad de CLAHE:** La mejora de contraste mediante CLAHE resulta superior a la ecualización global de histograma, preservando mejor estructuras anatómicas relevantes sin amplificar ruido excesivamente.

3. **Desafío del desbalance:** El desbalance de clases identificado debe abordarse cuidadosamente durante el entrenamiento de modelos para evitar sesgos hacia la clase mayoritaria.

4. **Fundamento sólido:** La exploración exhaustiva del dataset y la implementación de un pipeline robusto de preprocesamiento establecen una base sólida para las fases subsiguientes del proyecto.

5. **Preparación para extracción de características:** Las imágenes preprocesadas están en condiciones óptimas para la extracción de descriptores de forma y textura, así como para el entrenamiento de arquitecturas CNN.

### Conclusiones Futuras

**TODO:** Extender conclusiones cuando estén listos los resultados completos de descriptores y clasificadores.

**Aspectos a concluir:**
- Comparación cuantitativa entre descriptores clásicos y CNN.
- Análisis de cuáles descriptores aportaron mayor información discriminativa.
- Evaluación de la viabilidad práctica de cada enfoque en escenarios clínicos.
- Recomendaciones sobre cuándo utilizar métodos clásicos vs deep learning.
- Limitaciones identificadas y trabajo futuro.

---

## Referencias

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Disponible en: http://www.deeplearningbook.org

2. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.

3. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer. Disponible en: http://szeliski.org/Book/

4. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, 1, 886-893.

5. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(7), 971-987.

6. Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), 610-621.

7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.

8. Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *International Conference on Learning Representations (ICLR)*.

9. Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.e9.

10. Pizer, S. M., Amburn, E. P., Austin, J. D., et al. (1987). Adaptive histogram equalization and its variations. *Computer Vision, Graphics, and Image Processing*, 39(3), 355-368.

---

**Fin del Reporte Técnico**
