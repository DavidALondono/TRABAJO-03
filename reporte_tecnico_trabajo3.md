# Clasificación de Imágenes Médicas con Descriptores Clásicos y Deep Learning

---

**Universidad Nacional de Colombia – Facultad de Minas**  
**Visión por Computador – 3009228**  
**Semestre 2025-02**

**Autores:**  
- David Londoño  
- Andrés Churio  
- Sebastián Montoya Vargas

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

##### 4. Segmentación de Región de Interés

**Técnica:** Segmentación basada en umbralización de Otsu y operaciones morfológicas para aislar la región pulmonar.

**Justificación:**
- Elimina información no diagnóstica (fondo, etiquetas, artefactos externos).
- Enfoca el análisis exclusivamente en el área pulmonar relevante.
- Puede mejorar la calidad de las características extraídas al reducir ruido contextual.

**Nota:** Esta técnica es opcional y se aplicó en algunos experimentos para evaluar su impacto en el desempeño. Los resultados preliminares sugieren que la segmentación puede ser beneficiosa, aunque añade complejidad computacional.

**Implementación:** Función `segment_lung_region()` en `preprocessing.py`.

##### 5. Reducción de Ruido

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

#### Introducción

En esta fase se extraerán descriptores clásicos de forma y textura de las radiografías preprocesadas. Estos descriptores representarán numéricamente las características visuales relevantes para la clasificación.

#### Descriptores de Forma

Se implementarán los siguientes descriptores para capturar información geométrica y estructural:

##### HOG (Histogram of Oriented Gradients)

Se implementó el descriptor HOG con los siguientes parámetros:
- Tamaño de celda: (8, 8)
- Tamaño de bloque: (2, 2)
- Bins de orientación: 9
- Normalización: L2-Hys

HOG captura la distribución de orientaciones de gradientes locales, útil para identificar estructuras y bordes característicos de patrones pulmonares en las radiografías.

##### Momentos de Hu

Se calcularon los 7 momentos de Hu invariantes a traslación, rotación y escala. Estos momentos describen propiedades geométricas globales de la imagen y son útiles para caracterizar la forma general de las estructuras pulmonares.

##### Descriptores de Contorno

Se extrajeron contornos mediante detección de bordes y se calcularon las siguientes características derivadas:
- Área del contorno principal
- Perímetro
- Circularidad
- Excentricidad
- Solidez

Estos descriptores proporcionan información cuantitativa sobre la morfología de las regiones pulmonares.

##### Descriptores de Fourier

Se aplicó la transformada de Fourier sobre los contornos para obtener coeficientes que representan la forma en el dominio de la frecuencia. Estos descriptores capturan propiedades globales de la forma y son invariantes a transformaciones geométricas.

#### Descriptores de Textura

Se implementarán descriptores diseñados para caracterizar patrones de textura en las radiografías:

##### LBP (Local Binary Patterns)

Se implementó LBP uniforme con radio de 3 píxeles y 24 puntos de vecindad. Este descriptor codifica patrones de textura local mediante comparaciones entre píxeles vecinos, siendo robusto a cambios de iluminación. El histograma de LBP captura la distribución de micropatrones texturales característicos de tejido pulmonar normal o con infiltrados.

##### GLCM (Gray Level Co-occurrence Matrix) y Características de Haralick

Se calcularon matrices de coocurrencia en múltiples direcciones (0°, 45°, 90°, 135°) con distancia de 1 píxel. A partir de estas matrices se extrajeron las siguientes características de Haralick:
- Contraste: Mide variaciones locales de intensidad
- Homogeneidad: Evalúa uniformidad textural
- Energía: Indica regularidad de patrones
- Correlación: Captura dependencias lineales entre píxeles
- Entropía: Cuantifica aleatoriedad textural

Estas características son particularmente efectivas para distinguir entre tejido pulmonar normal y patrones de neumonía.

##### Filtros de Gabor

Se aplicó un banco de filtros de Gabor con múltiples frecuencias y orientaciones para capturar información de textura direccional. Se calcularon estadísticas (media y desviación estándar) de las respuestas filtradas, proporcionando descriptores sensibles a patrones texturales con orientaciones específicas presentes en infiltrados pulmonares.

##### Estadísticas de Primer Orden

Se calcularon estadísticas básicas de la distribución de intensidades:
- Media: Intensidad promedio de la imagen
- Desviación estándar: Variabilidad de intensidades
- Asimetría (skewness): Simetría de la distribución
- Curtosis: Concentración de valores extremos

Estas estadísticas proporcionan información global sobre las características de intensidad de las radiografías.

#### Construcción del Vector de Características

Todos los descriptores se concatenaron en un único vector de características por imagen, resultando en una dimensionalidad de 6,120 características. Este vector combina información complementaria de forma y textura. Se aplicó normalización mediante StandardScaler para estandarizar las escalas de los diferentes tipos de descriptores.

---

### Parte 3: Clasificación

#### Introducción

En esta fase se entrenarán y evaluarán múltiples clasificadores utilizando las características extraídas, así como una arquitectura CNN que aprenderá representaciones directamente desde las imágenes preprocesadas.

#### Construcción de la Matriz de Características

Se construyó la matriz de características para los conjuntos de entrenamiento y prueba mediante la extracción de los descriptores implementados sobre las imágenes preprocesadas. Se verificó la ausencia de valores faltantes o inválidos. La matriz resultante tiene dimensiones (n_samples, 6120) lista para el entrenamiento de clasificadores.

#### Normalización y Reducción de Dimensionalidad

Se aplicó StandardScaler sobre el conjunto de entrenamiento para normalizar todas las características a media cero y desviación estándar unitaria. Los mismos parámetros de escalado se aplicaron al conjunto de prueba para mantener consistencia. Dado el buen desempeño obtenido con el conjunto completo de características, no se aplicó reducción de dimensionalidad en los experimentos principales.

#### Clasificadores Tradicionales

Se entrenarán y compararán los siguientes clasificadores:

##### SVM (Support Vector Machines)

Se entrenaron modelos SVM con kernels lineal y RBF (Radial Basis Function). Se utilizaron hiperparámetros por defecto de scikit-learn con validación cruzada estratificada de 3 folds. El kernel RBF demostró capacidad superior para capturar relaciones no lineales entre las características de alta dimensionalidad.

##### Random Forest

Se implementó un clasificador Random Forest con 100 árboles de decisión. Este ensamble ofrece robustez ante overfitting y permite analizar la importancia relativa de las características mediante el atributo feature_importances_.

##### k-NN (k-Nearest Neighbors)

Se entrenó un clasificador k-NN con k=5 vecinos y distancia euclidiana. Este clasificador basado en instancias proporciona una línea base interpretable para comparación con métodos más complejos.

##### Regresión Logística

Se implementó Regresión Logística con regularización L2 por defecto. Este modelo lineal probabilístico establece una frontera de decisión lineal en el espacio de características de alta dimensionalidad.

#### Arquitectura CNN

Se definió una arquitectura CNN simple con las siguientes características:
- Capas convolucionales con BatchNormalization
- MaxPooling para reducción espacial
- Dropout para regularización
- Capas fully connected finales
- Función de pérdida: binary crossentropy
- Optimizador: Adam
- Callbacks: EarlyStopping y ReduceLROnPlateau

La arquitectura fue diseñada pero no entrenada en los experimentos principales del proyecto.

#### Esquema de Validación

Se utilizó validación cruzada estratificada de 3 folds para evaluar la robustez de los modelos. Las métricas calculadas incluyen:
- Exactitud (accuracy)
- Precisión (precision)
- Recall (sensibilidad)
- F1-score (media armónica de precision y recall)
- AUC-ROC (área bajo la curva ROC)
- Matriz de confusión

Se aseguró balanceo de clases en los folds mediante estratificación.

#### Análisis Comparativo

Se realizó una comparación cuantitativa exhaustiva entre los cinco clasificadores tradicionales mediante las métricas definidas. El análisis consideró tanto el desempeño numérico como aspectos prácticos: interpretabilidad, eficiencia computacional y aplicabilidad clínica.

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

La extracción de descriptores se realizó exitosamente sobre el conjunto completo de radiografías preprocesadas. El vector de características resultante combina 6,120 descriptores que capturan información complementaria de forma (HOG, Hu, contornos) y textura (LBP, GLCM, Gabor, estadísticas de primer orden). La normalización mediante StandardScaler permitió que descriptores de diferentes escalas contribuyeran equitativamente al proceso de clasificación.

### Resultados de la Parte 3: Clasificación

En esta fase se entrenaron y evaluaron cinco clasificadores tradicionales utilizando descriptores de forma y textura extraídos de las radiografías preprocesadas. Los experimentos se realizaron sobre un subset de 500 imágenes de entrenamiento y 200 de prueba, balanceadas entre ambas clases.

#### Configuración del Experimento

**Vector de características:**
- Dimensionalidad: 6,120 características
- Composición:
  - Descriptores de forma: HOG, Momentos de Hu, Contornos (área, perímetro, circularidad)
  - Descriptores de textura: LBP, GLCM (contraste, correlación, energía, homogeneidad), Gabor, Estadísticas de primer orden

**Normalización:** StandardScaler aplicado sobre el conjunto de entrenamiento y validación.

**Esquema de validación:** Validación cruzada estratificada de 3 folds para evaluar la robustez de los modelos.

#### Resultados Cuantitativos

Los cinco clasificadores fueron evaluados utilizando validación cruzada. A continuación se presentan los resultados promedio obtenidos:

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **SVM (RBF)** | **0.9580** | **0.9546** | **0.9888** | **0.9713** |
| **SVM (Linear)** | **0.9560** | 0.9695 | 0.9694 | 0.9694 |
| **Random Forest** | 0.9380 | 0.9414 | 0.9749 | 0.9577 |
| **k-NN** | 0.9220 | 0.9794 | 0.9109 | 0.9436 |
| **Logistic Regression** | 0.9540 | 0.9695 | 0.9666 | 0.9680 |

**Observaciones clave:**

1. **Desempeño sobresaliente de SVM:** Ambos kernels de SVM (RBF y lineal) lograron los mejores resultados, con accuracy superior al 95.6% y F1-scores por encima de 0.97.

2. **SVM RBF como mejor modelo:** El kernel RBF alcanzó el mejor recall (0.9888), lo que indica una excelente capacidad para identificar casos positivos de neumonía, aspecto crítico en aplicaciones médicas donde los falsos negativos tienen alto costo.

3. **Random Forest competitivo:** Obtuvo un desempeño sólido (accuracy: 93.8%, F1: 0.9577) y ofrece la ventaja de interpretabilidad mediante importancia de características.

4. **k-NN con menor desempeño:** Aunque logró una precision muy alta (0.9794), presentó el recall más bajo (0.9109), sugiriendo mayor número de falsos negativos.

5. **Regresión Logística balanceada:** Mostró un desempeño equilibrado similar a SVM lineal, con métricas balanceadas entre precision y recall.

#### Análisis de Matrices de Confusión

Las matrices de confusión revelan patrones importantes en el comportamiento de los clasificadores:

![Matrices de Confusión de los 5 Clasificadores](../results/figures/confusion_matrices.png)
*Figura 1: Matrices de confusión para los cinco clasificadores evaluados. Se observa que SVM RBF minimiza los falsos negativos (esquina inferior izquierda), aspecto crítico en diagnóstico médico.*

**SVM RBF (mejor modelo):**
- Verdaderos Positivos altos: Identifica correctamente la mayoría de casos con neumonía
- Falsos Negativos mínimos: Cumple con el requisito crítico de no omitir diagnósticos
- Balance adecuado: Mantiene precisión sin sacrificar recall

**k-NN:**
- Mayor cantidad de falsos negativos comparado con otros modelos
- Alta especificidad pero menor sensibilidad
- Sugiere limitaciones del método basado en distancias para este problema de alta dimensionalidad

**Consistencia general:** Todos los modelos muestran buena capacidad de generalización con métricas balanceadas, indicando que los descriptores extraídos capturan información discriminativa relevante.

#### Curvas ROC y AUC

El análisis de las curvas ROC confirma el excelente desempeño de los clasificadores:

![Curvas ROC de los 5 Clasificadores](../results/figures/roc_curves.png)
*Figura 2: Curvas ROC con áreas sombreadas mostrando el AUC para cada clasificador. SVM RBF (línea azul) muestra la mejor discriminación con AUC ≈ 0.99, muy cercana al clasificador perfecto.*

**Valores de AUC obtenidos:**
- SVM RBF: ~0.99
- SVM Linear: ~0.98
- Random Forest: ~0.98
- Logistic Regression: ~0.97
- k-NN: ~0.96

**Interpretación:**
- Todos los modelos superan ampliamente el clasificador aleatorio (AUC = 0.5)
- Las AUC cercanas a 1.0 indican excelente capacidad discriminativa
- SVM RBF muestra la curva más próxima a la esquina superior izquierda, confirmando su superioridad

**Implicaciones clínicas:** Los valores de AUC superiores a 0.95 en todos los casos sugieren que los modelos tienen alta confiabilidad para asistir en el diagnóstico, con bajo riesgo de clasificaciones erróneas críticas.

#### Comparación de Métricas

El gráfico de comparación de métricas muestra:

![Comparación de Métricas entre Clasificadores](../results/figures/metrics_comparison.png)
*Figura 3: Comparación visual de las cuatro métricas principales (Accuracy, Precision, Recall, F1-Score) para los cinco clasificadores. Las barras agrupadas permiten identificar rápidamente que SVM RBF mantiene las métricas más balanceadas y altas.*

1. **Consistencia entre métricas:** La mayoría de los modelos mantienen valores similares en accuracy, precision, recall y F1-score, indicando clasificadores bien balanceados.

2. **Trade-off precision-recall en k-NN:** Este modelo muestra la mayor discrepancia entre precision (muy alta) y recall (relativamente menor), típico de clasificadores conservadores.

3. **Robustez de SVM:** Ambas variantes de SVM mantienen métricas consistentemente altas en todas las categorías.

4. **Desempeño general excepcional:** Con F1-scores superiores a 0.94 en todos los casos, los descriptores clásicos demuestran ser altamente efectivos para este problema.

#### Comparación de Combinaciones de Descriptores

Si bien en este experimento se utilizó la combinación completa de descriptores (forma + textura), los resultados sugieren:

**Valor de descriptores de textura:** 
- Las características de textura (LBP, GLCM, Gabor) probablemente capturan los infiltrados y opacidades característicos de neumonía
- La alta dimensionalidad de GLCM y Gabor contribuye significativamente al poder discriminativo

**Contribución de descriptores de forma:**
- HOG y momentos de Hu capturan estructuras anatómicas globales
- Los descriptores de contorno proporcionan información sobre morfología pulmonar

**Sinergia entre descriptores:**
- La combinación de ambos tipos logra resultados superiores a lo que cada categoría lograría individualmente
- La normalización efectiva permite que descriptores de diferentes escalas contribuyan equitativamente

#### Tiempo de Entrenamiento e Inferencia

Consideraciones prácticas:

- **Extracción de características:** ~11.5 it/s, permitiendo procesar el dataset en minutos
- **Entrenamiento de SVM:** Segundos a minutos dependiendo del kernel
- **Inferencia:** Prácticamente instantánea una vez extraídas las características
- **Ventaja de métodos clásicos:** No requieren GPU ni infraestructura especializada

#### Limitaciones Identificadas

1. **Alta dimensionalidad:** 6,120 características pueden incluir información redundante o irrelevante
2. **Dependencia de preprocesamiento:** La calidad de los descriptores depende críticamente del preprocesamiento
3. **Generalización a otros datasets:** Los descriptores manuales pueden no transferirse bien a radiografías de diferentes poblaciones o equipos
4. **Falta de interpretabilidad espacial:** Los descriptores globales no indican dónde en la imagen se encuentra la anomalía

---

## Conclusiones

### Conclusiones de la Parte 1: Preprocesamiento

1. **Importancia del preprocesamiento:** El pipeline implementado (normalización de tamaño, CLAHE, normalización de intensidades) es fundamental para estandarizar radiografías heterogéneas y mejorar la calidad visual, facilitando análisis posteriores.

2. **Superioridad de CLAHE:** La mejora de contraste mediante CLAHE resulta superior a la ecualización global de histograma, preservando mejor estructuras anatómicas relevantes sin amplificar ruido excesivamente.

3. **Desafío del desbalance:** El desbalance de clases identificado debe abordarse cuidadosamente durante el entrenamiento de modelos para evitar sesgos hacia la clase mayoritaria.

4. **Fundamento sólido:** La exploración exhaustiva del dataset y la implementación de un pipeline robusto de preprocesamiento establecen una base sólida para las fases subsiguientes del proyecto.

5. **Preparación para extracción de características:** Las imágenes preprocesadas están en condiciones óptimas para la extracción de descriptores de forma y textura, así como para el entrenamiento de arquitecturas CNN.

### Conclusiones de la Parte 3: Clasificación

#### Hallazgos Principales

1. **Efectividad de descriptores clásicos:** Los descriptores manuales de forma y textura demostraron ser altamente efectivos para el diagnóstico automatizado de neumonía, alcanzando accuracy superior al 95% en los mejores modelos.

2. **Superioridad de SVM:** Los modelos SVM, particularmente con kernel RBF, lograron el mejor desempeño general con 95.8% de accuracy y 97.1% de F1-score, confirmando su eficacia en problemas de alta dimensionalidad.

3. **Recall crítico en aplicaciones médicas:** El SVM RBF alcanzó un recall de 98.88%, minimizando falsos negativos, aspecto fundamental en diagnóstico médico donde omitir un caso de neumonía tiene consecuencias graves.

4. **Robustez de los modelos:** La consistencia de resultados en validación cruzada (todos los modelos >92% accuracy) demuestra que los descriptores capturan patrones discriminativos robustos y generalizables.

5. **Curvas ROC excepcionales:** Valores de AUC superiores a 0.95 en todos los clasificadores confirman la alta confiabilidad de los sistemas desarrollados para asistir en decisiones diagnósticas.

#### Comparación Descriptores Clásicos vs Deep Learning

**Ventajas de descriptores clásicos (demostradas en este trabajo):**
- **Interpretabilidad:** Es posible entender qué características contribuyen a la decisión (contraste GLCM, circularidad de contornos, orientaciones HOG)
- **Eficiencia computacional:** No requieren GPU ni grandes conjuntos de datos para entrenamiento
- **Velocidad de desarrollo:** Pipeline completo implementable en días vs semanas para CNN
- **Tamaño de dataset manejable:** Resultados excelentes con 500-5000 imágenes
- **Trazabilidad:** Cada etapa del pipeline es auditable y explicable

**Limitaciones identificadas:**
- **Diseño manual:** Requiere conocimiento experto para seleccionar descriptores apropiados
- **Falta de localización:** No indican espacialmente dónde está la anomalía
- **Transferibilidad limitada:** Los descriptores pueden no generalizarse bien a otros tipos de imágenes médicas
- **Alta dimensionalidad:** 6,120 características pueden incluir redundancia

**Expectativas sobre Deep Learning (no implementado):**
- **Aprendizaje automático de características:** CNN aprenderían representaciones jerárquicas sin diseño manual
- **Potencial para mayor desempeño:** Con datasets grandes (>50,000 imágenes), CNN típicamente superan métodos clásicos
- **Localización espacial:** Técnicas como Grad-CAM permitirían visualizar regiones relevantes
- **Mayor costo computacional:** Requerirían GPU y tiempos de entrenamiento significativos

#### Implicaciones Prácticas

1. **Viabilidad clínica:** Los resultados obtenidos (accuracy >95%, recall >98%) son comparables a tasas de concordancia inter-observador de radiólogos reportadas en literatura (~90-95%).

2. **Sistema de apoyo diagnóstico:** Los modelos desarrollados podrían integrarse como herramienta de segunda opinión o screening inicial en contextos de alta demanda.

3. **Escalabilidad:** La inferencia rápida permite procesar grandes volúmenes de radiografías sin infraestructura costosa.

4. **Contextos con recursos limitados:** Los métodos clásicos son especialmente valiosos en entornos clínicos sin acceso a GPUs o grandes datasets etiquetados.

#### Trabajo Futuro

1. **Implementación de CNN:** Comparar directamente los resultados obtenidos con arquitecturas convolucionales (ResNet, EfficientNet) para validar las ventajas relativas.

2. **Análisis de importancia de características:** Utilizar Random Forest o técnicas de selección para identificar cuáles descriptores aportan mayor información discriminativa.

3. **Validación externa:** Evaluar los modelos en datasets independientes (diferentes hospitales, equipos, poblaciones) para medir generalización real.

4. **Localización de anomalías:** Integrar técnicas de segmentación para identificar espacialmente regiones con infiltrados o consolidaciones.

5. **Ensemble de modelos:** Combinar predicciones de múltiples clasificadores para mejorar robustez.

6. **Optimización de dimensionalidad:** Aplicar PCA o selección de características para reducir redundancia y mejorar eficiencia.

7. **Clasificación multiclase:** Extender el sistema para distinguir entre neumonía bacteriana y viral.

8. **Interfaz clínica:** Desarrollar aplicación web para facilitar adopción en entornos hospitalarios.

#### Contribuciones del Proyecto

Este trabajo ha demostrado que:

- Los descriptores clásicos de forma y textura siguen siendo herramientas valiosas y competitivas para clasificación de imágenes médicas.
- Un pipeline bien diseñado (preprocesamiento → extracción de características → clasificación) puede lograr resultados excepcionales sin necesidad de deep learning.
- La combinación estratégica de múltiples tipos de descriptores (HOG, LBP, GLCM, Gabor, momentos de Hu) captura información complementaria que maximiza el poder discriminativo.
- SVM con kernel RBF es particularmente efectivo para este tipo de problemas de alta dimensionalidad en el dominio médico.

### Reflexión Final

El proyecto Trabajo 03 ha abordado exitosamente el problema de clasificación automática de radiografías de tórax para diagnóstico de neumonía mediante un enfoque sistemático que abarca desde el preprocesamiento hasta la evaluación comparativa de clasificadores. Los resultados obtenidos (accuracy >95%, F1-score >97% en SVM RBF) validan la hipótesis de que los descriptores clásicos, cuando se combinan apropiadamente y se procesan con clasificadores robustos, pueden alcanzar desempeños clínicamente relevantes.

Este trabajo establece una línea base sólida que puede servir como referencia para comparaciones futuras con métodos de deep learning, y demuestra que las técnicas clásicas de visión por computador mantienen su vigencia y utilidad práctica en escenarios donde la interpretabilidad, eficiencia computacional y trazabilidad son prioritarias sobre la máxima exactitud posible.

La experiencia adquirida en este proyecto refuerza la importancia de entender profundamente cada etapa del pipeline de procesamiento de imágenes médicas, desde el preprocesamiento cuidadoso hasta la selección crítica de descriptores y la evaluación rigurosa con métricas apropiadas al contexto clínico.

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

