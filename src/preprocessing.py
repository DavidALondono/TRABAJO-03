"""
Pipeline de Preprocesamiento para Imágenes Médicas (Rayos X)
=============================================================

Este módulo implementa el pipeline completo de preprocesamiento
para imágenes de rayos X de tórax, incluyendo:

1. Normalización de tamaño
2. Mejora de contraste con CLAHE
3. Segmentación de región de interés (ROI)
4. Normalización de intensidad
5. Reducción de ruido

Autor: David A. Londoño
Fecha: Noviembre 2025
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# FUNCIONES DE NORMALIZACIÓN DE TAMAÑO
# ============================================================

def resize_image(image: np.ndarray, 
                target_size: Tuple[int, int] = (224, 224),
                interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Redimensiona una imagen al tamaño objetivo.
    
    Justificación:
    - Las redes neuronales requieren tamaños de entrada fijos
    - INTER_AREA es óptimo para reducir tamaño (preserva información)
    - 224x224 es un estándar en arquitecturas CNN pre-entrenadas
    
    Args:
        image (np.ndarray): Imagen de entrada
        target_size (Tuple[int, int]): Tamaño objetivo (ancho, alto)
        interpolation (int): Método de interpolación de OpenCV
        
    Returns:
        np.ndarray: Imagen redimensionada
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def resize_keep_aspect_ratio(image: np.ndarray,
                             target_size: int = 224,
                             pad_color: int = 0) -> np.ndarray:
    """
    Redimensiona manteniendo la relación de aspecto y añade padding.
    
    Este método es preferible cuando se desea evitar distorsión
    en las estructuras anatómicas de la radiografía.
    
    Args:
        image (np.ndarray): Imagen de entrada
        target_size (int): Tamaño del lado mayor
        pad_color (int): Color del padding (0 = negro)
        
    Returns:
        np.ndarray: Imagen redimensionada con padding
    """
    h, w = image.shape[:2]
    
    # Calcular escala
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Redimensionar
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Crear canvas con padding
    canvas = np.full((target_size, target_size), pad_color, dtype=np.uint8)
    
    # Centrar la imagen
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


# ============================================================
# MEJORA DE CONTRASTE - CLAHE
# ============================================================

def apply_clahe(image: np.ndarray,
               clip_limit: float = 2.0,
               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Justificación técnica:
    - Las radiografías a menudo tienen bajo contraste debido a variaciones
      en la exposición y configuración del equipo
    - CLAHE mejora el contraste localmente sin amplificar excesivamente el ruido
    - Es adaptativo: ajusta el contraste en regiones pequeñas (tiles)
    - El límite de recorte previene sobre-amplificación en regiones homogéneas
    
    Beneficios en radiografías:
    - Mejora la visibilidad de estructuras pulmonares sutiles
    - Hace más evidentes los patrones de infiltración en neumonía
    - Preserva detalles en regiones oscuras y claras simultáneamente
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        clip_limit (float): Límite de recorte para contraste (1.0-4.0 típico)
        tile_grid_size (Tuple[int, int]): Tamaño de la cuadrícula de tiles
        
    Returns:
        np.ndarray: Imagen con contraste mejorado
    """
    # Crear objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Aplicar CLAHE
    enhanced = clahe.apply(image)
    
    return enhanced


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Aplica ecualización de histograma tradicional (global).
    
    Nota: CLAHE es generalmente preferible para imágenes médicas,
    pero esta función se incluye para comparación.
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        
    Returns:
        np.ndarray: Imagen ecualizada
    """
    return cv2.equalizeHist(image)


# ============================================================
# REDUCCIÓN DE RUIDO
# ============================================================

def denoise_image(image: np.ndarray,
                 method: str = 'bilateral',
                 strength: int = 10) -> np.ndarray:
    """
    Reduce el ruido en la imagen usando diferentes métodos.
    
    Args:
        image (np.ndarray): Imagen de entrada
        method (str): 'bilateral', 'gaussian', 'median', 'nlmeans'
        strength (int): Fuerza del filtro
        
    Returns:
        np.ndarray: Imagen sin ruido
    """
    if method == 'bilateral':
        # Preserva bordes mientras suaviza regiones homogéneas
        return cv2.bilateralFilter(image, d=9, sigmaColor=strength*7.5, sigmaSpace=strength*7.5)
    
    elif method == 'gaussian':
        # Suavizado gaussiano simple
        kernel_size = strength if strength % 2 == 1 else strength + 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif method == 'median':
        # Bueno para ruido tipo sal y pimienta
        kernel_size = strength if strength % 2 == 1 else strength + 1
        return cv2.medianBlur(image, kernel_size)
    
    elif method == 'nlmeans':
        # Non-Local Means - más lento pero efectivo
        return cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)
    
    else:
        return image


# ============================================================
# SEGMENTACIÓN DE REGIÓN DE INTERÉS (ROI)
# ============================================================

def segment_lung_region(image: np.ndarray,
                       threshold_method: str = 'otsu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmenta la región pulmonar en una radiografía de tórax.
    
    Justificación:
    - Enfoca el análisis en la región de interés (pulmones)
    - Elimina áreas irrelevantes (fondo, márgenes, anotaciones)
    - Reduce variabilidad debida a diferencias en encuadre
    - Mejora el rendimiento de los clasificadores al eliminar ruido contextual
    
    Args:
        image (np.ndarray): Radiografía en escala de grises
        threshold_method (str): 'otsu', 'adaptive', 'manual'
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (máscara binaria, imagen segmentada)
    """
    # Aplicar CLAHE primero para mejorar contraste
    enhanced = apply_clahe(image, clip_limit=3.0)
    
    # Binarización
    if threshold_method == 'otsu':
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_method == 'adaptive':
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    else:
        _, binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    
    # Operaciones morfológicas para limpiar la máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Cerrar pequeños agujeros
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Abrir para eliminar ruido pequeño
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Encontrar el contorno más grande (asumiendo que son los pulmones)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Seleccionar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Crear máscara refinada
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, [largest_contour], -1, 255, -1)
        
        # Dilatar ligeramente para incluir bordes
        refined_mask = cv2.dilate(refined_mask, kernel, iterations=3)
    else:
        refined_mask = mask
    
    # Aplicar máscara a la imagen original
    segmented = cv2.bitwise_and(image, image, mask=refined_mask)
    
    return refined_mask, segmented


def crop_to_roi(image: np.ndarray, mask: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    Recorta la imagen a la región de interés definida por la máscara.
    
    Args:
        image (np.ndarray): Imagen original
        mask (np.ndarray): Máscara binaria
        padding (int): Píxeles de padding alrededor del ROI
        
    Returns:
        np.ndarray: Imagen recortada
    """
    # Encontrar coordenadas del bounding box
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Añadir padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    # Recortar
    cropped = image[y:y+h, x:x+w]
    
    return cropped


# ============================================================
# NORMALIZACIÓN DE INTENSIDAD
# ============================================================

def normalize_intensity(image: np.ndarray,
                       method: str = 'minmax',
                       target_range: Tuple[int, int] = (0, 255)) -> np.ndarray:
    """
    Normaliza las intensidades de la imagen.
    
    Args:
        image (np.ndarray): Imagen de entrada
        method (str): 'minmax' o 'zscore'
        target_range (Tuple[int, int]): Rango objetivo para minmax
        
    Returns:
        np.ndarray: Imagen normalizada
    """
    if method == 'minmax':
        # Normalización min-max
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val - min_val > 0:
            normalized = (image - min_val) / (max_val - min_val)
            normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
            return normalized.astype(np.uint8)
        else:
            return image
    
    elif method == 'zscore':
        # Normalización Z-score (estandarización)
        mean = np.mean(image)
        std = np.std(image)
        
        if std > 0:
            normalized = (image - mean) / std
            # Escalar a rango 0-255
            normalized = ((normalized + 3) / 6) * 255  # Asumiendo ~3 std
            return np.clip(normalized, 0, 255).astype(np.uint8)
        else:
            return image
    
    return image


# ============================================================
# PIPELINE COMPLETO
# ============================================================

def preprocess_xray(image: np.ndarray,
                   target_size: Tuple[int, int] = (224, 224),
                   apply_clahe_enhancement: bool = True,
                   apply_denoising: bool = False,
                   segment_roi: bool = False,
                   normalize: bool = True) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento para radiografías.
    
    Pasos del pipeline:
    1. Reducción de ruido 
    2. Normalización de intensidad
    3. Mejora de contraste con CLAHE
    4. Segmentación de ROI 
    5. Redimensionamiento al tamaño objetivo
    
    Args:
        image (np.ndarray): Imagen original en escala de grises
        target_size (Tuple[int, int]): Tamaño final de la imagen
        apply_clahe_enhancement (bool): Aplicar CLAHE
        apply_denoising (bool): Aplicar reducción de ruido
        segment_roi (bool): Segmentar región pulmonar
        normalize (bool): Normalizar intensidades
        
    Returns:
        np.ndarray: Imagen preprocesada
    """
    processed = image.copy()
    
    # 1. Reducción de ruido 
    if apply_denoising:
        processed = denoise_image(processed, method='bilateral', strength=10)
    
    # 2. Normalización de intensidad
    if normalize:
        processed = normalize_intensity(processed, method='minmax')
    
    # 3. Mejora de contraste con CLAHE
    if apply_clahe_enhancement:
        processed = apply_clahe(processed, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # 4. Segmentación de ROI 
    if segment_roi:
        mask, processed = segment_lung_region(processed)
        processed = crop_to_roi(processed, mask, padding=10)
    
    # 5. Redimensionamiento
    processed = resize_image(processed, target_size)
    
    return processed


def preprocess_dataset(images: list,
                      target_size: Tuple[int, int] = (224, 224),
                      apply_clahe_enhancement: bool = True,
                      apply_denoising: bool = False,
                      segment_roi: bool = False,
                      normalize: bool = True,
                      verbose: bool = True) -> list:
    """
    Aplica el pipeline de preprocesamiento a un conjunto de imágenes.
    
    Args:
        images (list): Lista de imágenes
        target_size (Tuple[int, int]): Tamaño objetivo
        apply_clahe_enhancement (bool): Aplicar CLAHE
        apply_denoising (bool): Aplicar reducción de ruido
        segment_roi (bool): Segmentar ROI
        normalize (bool): Normalizar intensidades
        verbose (bool): Mostrar progreso
        
    Returns:
        list: Lista de imágenes preprocesadas
    """
    processed_images = []
    total = len(images)
    
    for i, img in enumerate(images):
        if verbose and (i + 1) % 100 == 0:
            print(f"Procesadas {i+1}/{total} imágenes...")
        
        processed = preprocess_xray(
            img,
            target_size=target_size,
            apply_clahe_enhancement=apply_clahe_enhancement,
            apply_denoising=apply_denoising,
            segment_roi=segment_roi,
            normalize=normalize
        )
        processed_images.append(processed)
    
    if verbose:
        print(f"✓ Preprocesamiento completado: {total} imágenes")
    
    return processed_images


# ============================================================
# FUNCIONES DE ANÁLISIS
# ============================================================

def compare_preprocessing_methods(image: np.ndarray) -> dict:
    """
    Compara diferentes métodos de preprocesamiento sobre una imagen.
    
    Args:
        image (np.ndarray): Imagen original
        
    Returns:
        dict: Diccionario con diferentes versiones procesadas
    """
    results = {
        'original': image,
        'clahe': apply_clahe(image),
        'hist_eq': apply_histogram_equalization(image),
        'bilateral': denoise_image(image, method='bilateral'),
        'complete_pipeline': preprocess_xray(image, target_size=image.shape[::-1])
    }
    
    return results


if __name__ == "__main__":
    print("Módulo de preprocesamiento cargado correctamente.")
    print("\nFunciones disponibles:")
    print("- resize_image()")
    print("- apply_clahe()")
    print("- segment_lung_region()")
    print("- preprocess_xray() [Pipeline completo]")
    print("- preprocess_dataset()")
