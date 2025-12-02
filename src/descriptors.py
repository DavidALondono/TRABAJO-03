"""
Pipeline de extracción de descriptores para Imágenes Médicas (Rayos X)
=============================================================
Este módulo implementa el pipeline completo de descriptores
para imágenes de rayos X de tórax, incluyendo:

  - HOG
  - Momentos de Hu
  - Segmentación por Otsu
  - Contorno
  - Área, perímetro, circularidad, excentricidad

  - Local Binary Patterns (LBP)
  - Gray Level Co-occurrence Matrix (GLCM)
  - Filtros de Gabor

Autor: Andres D. Churio
Fecha: Noviembre 2025
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor

# ----------------------------------------------------------------------
# 1. HOG descriptor
# ----------------------------------------------------------------------
def get_hog(img, orientations=9, pixels_per_cell=(16, 16)):
    """
    Obtiene las características HOG de una imagen.
    """
    resized = cv2.resize(img, (256, 256))

    features, hog_image = hog(
        resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )

    return features, hog_image


# ----------------------------------------------------------------------
# 2. Momentos invariantes de Hu
# ----------------------------------------------------------------------
def extract_hu_moments(img):
    """
    Extrae los 7 momentos invariantes de Hu.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    M = cv2.moments(thresh)
    hu = cv2.HuMoments(M).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu, thresh


# ----------------------------------------------------------------------
# 3. Segmentación + Morfología
# ----------------------------------------------------------------------
def segment_image(img):
    """
    Segmenta con Otsu y limpia ruido morfológicamente.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    return mask_clean


# ----------------------------------------------------------------------
# 4. Contorno y descriptores morfológicos
# ----------------------------------------------------------------------
def contour_descriptors(mask):
    """
    Calcula área, perímetro, circularidad, excentricidad
    a partir de una máscara binaria.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # área y perímetro
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # circularidad
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0

    # excentricidad (ajuste de elipse)
    if len(cnt) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        eccentricity = np.sqrt(1 - (MA / ma) ** 2)
    else:
        eccentricity = None

    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "eccentricity": eccentricity,
        "contour": cnt
    }


# ----------------------------------------------------------------------
# 6. Carga de una imagen
# ----------------------------------------------------------------------
def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar: {path}")
    return img


# ----------------------------------------------------------------------
# 7. Cálculo de local binary patterns
# ----------------------------------------------------------------------
def compute_lbp(img, P=8, R=1, method='uniform'):
    """
    img     : imagen en escala de grises
    P       : número de vecinos
    R       : radio
    method  : 'default', 'ror', 'uniform', 'var'
    """
    lbp = local_binary_pattern(img, P, R, method)

    # Histograma: número de patrones depende de P
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    return lbp, hist


# ----------------------------------------------------------------------
# 8a. Cálculo de Matriz de Coocurrencia de Niveles de Gris
# ----------------------------------------------------------------------

def glcm_manual(image, distances=[1], angles=[0], levels=256):
    """
    Calcula GLCM manualmente usando solo NumPy.
    image: imagen en escala de grises (uint8)
    distances: lista de distancias (p.ej. [1,2])
    angles: lista de ángulos en radianes (p.ej. [0, np.pi/4])
    """
    glcms = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.uint32)

    for di, d in enumerate(distances):
        for ai, angle in enumerate(angles):

            dx = int(np.round(np.cos(angle) * d))
            dy = int(np.round(np.sin(angle) * d))

            for y in range(image.shape[0]):
                for x in range(image.shape[1]):
                    y2 = y + dy
                    x2 = x + dx

                    if 0 <= y2 < image.shape[0] and 0 <= x2 < image.shape[1]:
                        i = image[y, x]
                        j = image[y2, x2]
                        glcms[i, j, di, ai] += 1

    return glcms


# ----------------------------------------------------------------------
# 8b. Cálculo de propieades de Coocurrencia de Niveles de Gris
# ----------------------------------------------------------------------


def glcm_props(glcm):
    """
    Calcula contraste, energía, homogeneidad, correlación.
    glcm: matriz GLCM normalizada (2D)
    """
    glcm_norm = glcm / glcm.sum()

    i = np.arange(glcm.shape[0])[:, None]
    j = np.arange(glcm.shape[1])[None, :]

    # Contraste
    contrast = np.sum((i - j) ** 2 * glcm_norm)

    # Energía
    energy = np.sum(glcm_norm ** 2)

    # Homogeneidad
    homogeneity = np.sum(glcm_norm / (1 + np.abs(i - j)))

    # Correlación
    mu_i = np.sum(i * glcm_norm)
    mu_j = np.sum(j * glcm_norm)
    sigma_i = np.sqrt(np.sum((i - mu_i) ** 2 * glcm_norm))
    sigma_j = np.sqrt(np.sum((j - mu_j) ** 2 * glcm_norm))

    if sigma_i * sigma_j == 0:
        correlation = 1  # edge case
    else:
        correlation = np.sum((i - mu_i)*(j - mu_j)*glcm_norm) / (sigma_i * sigma_j)

    return contrast, energy, homogeneity, correlation


# ----------------------------------------------------------------------
# 9a. Aplicación del filtro de Gabor
# ----------------------------------------------------------------------

def apply_gabor(img, frequency, theta):
    """
    Aplica un filtro de Gabor con una frecuencia y orientación dadas.
    img        : imagen en escala de grises
    frequency  : frecuencia espacial (0.1 a 0.5 es común)
    theta      : orientación en radianes (0, π/4, π/2...)
    """
    real, imag = gabor(img, frequency=frequency, theta=theta)
    magnitude = np.sqrt(real**2 + imag**2)
    return magnitude

# ----------------------------------------------------------------------
# 9b. Definición de un banco de filtro de Gabor
# ----------------------------------------------------------------------

def gabor_bank(img, frequencies, orientations):
    responses = {}

    for f in frequencies:
        for theta in orientations:
            mag = apply_gabor(img, f, theta)

            responses[(f, theta)] = {
                "magnitude": mag,
                "mean": mag.mean(),
                "std": mag.std()
            }

    return responses
