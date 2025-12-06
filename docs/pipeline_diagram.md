# Diagramas de Flujo del Pipeline

## Pipeline General de Clasificación de Radiografías

```mermaid
flowchart TD
    A[Dataset Raw: Chest X-Ray Images] --> B[Preprocesamiento]
    B --> C[Extracción de Características]
    C --> D[Clasificación]
    D --> E[Evaluación y Resultados]
    
    B --> B1[Resize 224x224]
    B1 --> B2[CLAHE]
    B2 --> B3[Normalización Intensidad]
    B3 --> B4[Opcional: Segmentación ROI]
    
    C --> C1[Descriptores de Forma]
    C --> C2[Descriptores de Textura]
    
    C1 --> C1A[HOG]
    C1 --> C1B[Momentos de Hu]
    C1 --> C1C[Contorno]
    
    C2 --> C2A[LBP]
    C2 --> C2B[GLCM/Haralick]
    C2 --> C2C[Gabor]
    C2 --> C2D[Estadísticas]
    
    C1A --> F[Concatenar Features]
    C1B --> F
    C1C --> F
    C2A --> F
    C2B --> F
    C2C --> F
    C2D --> F
    
    F --> G[Normalización StandardScaler]
    G --> D
    
    D --> D1[SVM RBF]
    D --> D2[Random Forest]
    D --> D3[k-NN]
    D --> D4[Regresión Logística]
    
    D1 --> E
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> E1[Métricas: Accuracy, Precision, Recall, F1]
    E --> E2[Matriz de Confusión]
    E --> E3[Curvas ROC]
```

## Pipeline Detallado de Preprocesamiento

```mermaid
flowchart LR
    A[Imagen Original<br/>Tamaño Variable] --> B{Verificar<br/>Canales}
    B -->|RGB| C[Convertir a Grayscale]
    B -->|Grayscale| D[Resize a 224x224]
    C --> D
    
    D --> E[Aplicar CLAHE<br/>clip_limit=2.0<br/>tile_grid=(8,8)]
    
    E --> F[Normalizar<br/>Rango [0, 1]]
    
    F --> G{Segmentar<br/>ROI?}
    G -->|Sí| H[Umbralización Otsu]
    G -->|No| K[Imagen Preprocesada]
    
    H --> I[Operaciones Morfológicas<br/>Closing + Opening]
    I --> J[Máscara de Pulmones]
    J --> K
    
    K --> L[Guardar/Procesar]
```

## Pipeline de Extracción de Descriptores HOG

```mermaid
flowchart TD
    A[Imagen Preprocesada<br/>224x224 Grayscale] --> B[Calcular Gradientes<br/>Gx, Gy]
    
    B --> C[Magnitud y Orientación<br/>de Gradientes]
    
    C --> D[Dividir en Celdas<br/>8x8 píxeles]
    
    D --> E[Histograma de<br/>Orientaciones por Celda<br/>9 bins]
    
    E --> F[Agrupar en Bloques<br/>2x2 celdas]
    
    F --> G[Normalización L2-Hys<br/>por Bloque]
    
    G --> H[Concatenar Histogramas]
    
    H --> I[Vector HOG Final<br/>Dimensión: 1,764]
```

## Pipeline de Extracción LBP

```mermaid
flowchart TD
    A[Imagen Preprocesada<br/>224x224] --> B[Seleccionar Parámetros<br/>P=8 vecinos, R=1 radio]
    
    B --> C[Para cada píxel central]
    
    C --> D[Comparar con 8 vecinos<br/>circulares]
    
    D --> E[Generar código binario<br/>8 bits]
    
    E --> F[Convertir a decimal<br/>Rango: 0-255]
    
    F --> G[Construir Histograma<br/>256 bins]
    
    G --> H{LBP Uniforme?}
    H -->|Sí| I[Reducir a 59 bins<br/>patrones uniformes]
    H -->|No| J[Mantener 256 bins]
    
    I --> K[Normalizar Histograma]
    J --> K
    
    K --> L[Vector LBP Final<br/>Dimensión: 59 o 256]
```

## Pipeline de Clasificación con SVM

```mermaid
flowchart TD
    A[Matriz de Características<br/>N_samples x N_features] --> B[Dividir Train/Test<br/>80/20]
    
    B --> C[Conjunto Entrenamiento]
    B --> D[Conjunto Prueba]
    
    C --> E[Normalización<br/>StandardScaler]
    E --> F[Aplicar transformación]
    
    F --> G[Entrenar SVM RBF<br/>GridSearchCV]
    
    G --> H[Búsqueda de<br/>Hiperparámetros<br/>C, gamma]
    
    H --> I[Mejor Modelo<br/>C=10, gamma=0.001]
    
    D --> J[Aplicar misma<br/>transformación]
    
    J --> K[Predicción]
    I --> K
    
    K --> L[Calcular Métricas]
    
    L --> M[Accuracy: 95.51%]
    L --> N[Precision: 94.32%]
    L --> O[Recall: 98.46%]
    L --> P[F1-Score: 96.34%]
    
    L --> Q[Matriz de Confusión]
    L --> R[Curva ROC<br/>AUC: 0.986]
```

## Pipeline de Validación Cruzada

```mermaid
flowchart TD
    A[Dataset Completo] --> B[5-Fold Cross-Validation]
    
    B --> C1[Fold 1]
    B --> C2[Fold 2]
    B --> C3[Fold 3]
    B --> C4[Fold 4]
    B --> C5[Fold 5]
    
    C1 --> D1[Train: 4 folds<br/>Val: 1 fold]
    C2 --> D2[Train: 4 folds<br/>Val: 1 fold]
    C3 --> D3[Train: 4 folds<br/>Val: 1 fold]
    C4 --> D4[Train: 4 folds<br/>Val: 1 fold]
    C5 --> D5[Train: 4 folds<br/>Val: 1 fold]
    
    D1 --> E[Entrenar y Evaluar]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    
    E --> F[Promediar Métricas]
    
    F --> G[Accuracy Mean ± Std]
    F --> H[Precision Mean ± Std]
    F --> I[Recall Mean ± Std]
    F --> J[F1-Score Mean ± Std]
```

## Flujo de Decisión: Selección de Clasificador

```mermaid
flowchart TD
    A[Problema de Clasificación] --> B{Tamaño del<br/>Dataset}
    
    B -->|Pequeño<br/>< 1000| C[k-NN o<br/>Regresión Logística]
    B -->|Mediano<br/>1000-10000| D{Alta<br/>Dimensionalidad?}
    B -->|Grande<br/>> 10000| E[SVM o<br/>Random Forest]
    
    D -->|Sí| F[SVM con kernel RBF]
    D -->|No| G[Random Forest]
    
    C --> H[Evaluar]
    E --> H
    F --> H
    G --> H
    
    H --> I{Accuracy<br/>> 90%?}
    
    I -->|Sí| J[Modelo Aceptado]
    I -->|No| K[Ajustar Hiperparámetros<br/>o Features]
    
    K --> H
    
    J --> L[Validación Final<br/>en Test Set]
```

## Arquitectura de Sistema Completo

```mermaid
graph TB
    subgraph Entrada
        A1[Radiografías Raw]
        A2[Metadata Clínica]
    end
    
    subgraph Preprocesamiento
        B1[Módulo de Limpieza]
        B2[Módulo CLAHE]
        B3[Módulo Normalización]
    end
    
    subgraph Extracción
        C1[Extractor HOG]
        C2[Extractor LBP]
        C3[Extractor GLCM]
        C4[Extractor Gabor]
        C5[Extractor Hu Moments]
    end
    
    subgraph Clasificación
        D1[Ensemble de Modelos]
        D2[SVM RBF]
        D3[Random Forest]
        D4[Voting Classifier]
    end
    
    subgraph Salida
        E1[Predicción: NORMAL/PNEUMONIA]
        E2[Probabilidades]
        E3[Confianza Score]
        E4[Visualización Explicativa]
    end
    
    A1 --> B1
    B1 --> B2
    B2 --> B3
    
    B3 --> C1
    B3 --> C2
    B3 --> C3
    B3 --> C4
    B3 --> C5
    
    C1 --> D1
    C2 --> D1
    C3 --> D1
    C4 --> D1
    C5 --> D1
    
    D1 --> D2
    D1 --> D3
    D2 --> D4
    D3 --> D4
    
    D4 --> E1
    D4 --> E2
    D4 --> E3
    D4 --> E4
    
    A2 --> E4
```

---

## Notas sobre los Diagramas

Estos diagramas ilustran el flujo completo del sistema de clasificación de radiografías implementado en el proyecto. Cada diagrama representa una etapa específica o una vista diferente del pipeline:

1. **Pipeline General:** Vista de alto nivel de todo el proceso
2. **Preprocesamiento:** Detalles técnicos de la estandarización de imágenes
3. **HOG y LBP:** Algoritmos específicos de extracción de características
4. **Clasificación:** Proceso de entrenamiento y evaluación de modelos
5. **Validación Cruzada:** Estrategia de evaluación robusta
6. **Selección de Clasificador:** Lógica de decisión
7. **Arquitectura Completa:** Integración de todos los componentes

Estos diagramas pueden ser renderizados con herramientas que soporten sintaxis Mermaid (GitHub, VS Code, GitLab, etc.).
