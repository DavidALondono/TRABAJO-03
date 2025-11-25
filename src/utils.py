"""
Utilidades Generales para el Proyecto de Clasificación de Imágenes Médicas
===========================================================================

Este módulo contiene funciones auxiliares para:
- Manejo de rutas y archivos
- Carga de imágenes
- Visualización
- Logging
- Análisis estadístico

Autor: David A. Londoño
Fecha: Noviembre 2025
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd


# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

# Configuración de estilo para gráficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================
# MANEJO DE RUTAS Y ARCHIVOS
# ============================================================

def get_project_root() -> Path:
    """
    Obtiene el directorio raíz del proyecto.
    
    Returns:
        Path: Ruta al directorio raíz del proyecto
    """
    return Path(__file__).parent.parent


def create_directory(path: str) -> None:
    """
    Crea un directorio si no existe.
    
    Args:
        path (str): Ruta del directorio a crear
    """
    os.makedirs(path, exist_ok=True)
    

def list_images(directory: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> List[str]:
    """
    Lista todas las imágenes en un directorio con las extensiones especificadas.
    
    Args:
        directory (str): Ruta del directorio
        extensions (List[str]): Lista de extensiones válidas
        
    Returns:
        List[str]: Lista de rutas completas a las imágenes
    """
    images = []
    for ext in extensions:
        images.extend(Path(directory).rglob(f'*{ext}'))
    return [str(img) for img in images]


# ============================================================
# CARGA DE IMÁGENES
# ============================================================

def load_image(image_path: str, color_mode: str = 'grayscale') -> Optional[np.ndarray]:
    """
    Carga una imagen desde el disco.
    
    Args:
        image_path (str): Ruta a la imagen
        color_mode (str): 'grayscale' o 'color'
        
    Returns:
        np.ndarray: Imagen cargada o None si hay error
    """
    try:
        if color_mode == 'grayscale':
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            print(f"Error: No se pudo cargar la imagen {image_path}")
        return img
    except Exception as e:
        print(f"Error al cargar {image_path}: {str(e)}")
        return None


def load_dataset_images(data_dir: str, max_images: Optional[int] = None) -> Dict[str, List[np.ndarray]]:
    """
    Carga imágenes del dataset organizadas por clase.
    
    Args:
        data_dir (str): Directorio raíz del dataset
        max_images (int, optional): Número máximo de imágenes por clase
        
    Returns:
        Dict: Diccionario con clases como keys y listas de imágenes como values
    """
    dataset = {}
    
    # Iterar sobre cada subdirectorio (clase)
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            images = []
            
            # Cargar imágenes de la clase
            image_paths = list_images(str(class_dir))
            if max_images:
                image_paths = image_paths[:max_images]
                
            for img_path in image_paths:
                img = load_image(img_path)
                if img is not None:
                    images.append(img)
            
            dataset[class_name] = images
            print(f"Cargadas {len(images)} imágenes de la clase '{class_name}'")
    
    return dataset


# ============================================================
# VISUALIZACIÓN
# ============================================================

def plot_image_grid(images: List[np.ndarray], 
                    titles: List[str] = None,
                    rows: int = 2, 
                    cols: int = 4,
                    figsize: Tuple[int, int] = (15, 8),
                    cmap: str = 'gray') -> None:
    """
    Muestra una cuadrícula de imágenes.
    
    Args:
        images (List[np.ndarray]): Lista de imágenes
        titles (List[str], optional): Títulos para cada imagen
        rows (int): Número de filas
        cols (int): Número de columnas
        figsize (Tuple[int, int]): Tamaño de la figura
        cmap (str): Mapa de colores para matplotlib
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap=cmap)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(class_counts: Dict[str, int], 
                           title: str = "Distribución de Clases") -> None:
    """
    Visualiza la distribución de clases en un gráfico de barras.
    
    Args:
        class_counts (Dict[str, int]): Diccionario con conteos por clase
        title (str): Título del gráfico
    """
    plt.figure(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color=['#3498db', '#e74c3c'])
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Número de Imágenes', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_image_comparison(original: np.ndarray, 
                         processed: np.ndarray,
                         title1: str = "Original",
                         title2: str = "Procesada",
                         cmap: str = 'gray') -> None:
    """
    Compara dos imágenes lado a lado.
    
    Args:
        original (np.ndarray): Imagen original
        processed (np.ndarray): Imagen procesada
        title1 (str): Título de la imagen original
        title2 (str): Título de la imagen procesada
        cmap (str): Mapa de colores
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title(title1, fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(processed, cmap=cmap)
    axes[1].set_title(title2, fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_histogram(image: np.ndarray, 
                   title: str = "Histograma de Intensidades") -> None:
    """
    Muestra el histograma de una imagen en escala de grises.
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        title (str): Título del histograma
    """
    plt.figure(figsize=(10, 5))
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    plt.xlabel('Intensidad de Píxel', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multiple_histograms(images: List[np.ndarray], 
                            labels: List[str],
                            title: str = "Comparación de Histogramas") -> None:
    """
    Compara histogramas de múltiples imágenes.
    
    Args:
        images (List[np.ndarray]): Lista de imágenes
        labels (List[str]): Etiquetas para cada histograma
        title (str): Título del gráfico
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (img, label) in enumerate(zip(images, labels)):
        color = colors[i % len(colors)]
        plt.hist(img.ravel(), bins=256, range=[0, 256], 
                alpha=0.5, label=label, color=color)
    
    plt.xlabel('Intensidad de Píxel', fontsize=11)
    plt.ylabel('Frecuencia', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# ANÁLISIS ESTADÍSTICO
# ============================================================

def get_image_statistics(image: np.ndarray) -> Dict[str, float]:
    """
    Calcula estadísticas básicas de una imagen.
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        
    Returns:
        Dict: Diccionario con estadísticas
    """
    return {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'median': np.median(image)
    }


def analyze_dataset_sizes(data_dir: str) -> pd.DataFrame:
    """
    Analiza los tamaños de las imágenes en el dataset.
    
    Args:
        data_dir (str): Directorio del dataset
        
    Returns:
        pd.DataFrame: DataFrame con estadísticas de tamaños
    """
    sizes = []
    
    for class_dir in Path(data_dir).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            image_paths = list_images(str(class_dir))
            
            for img_path in image_paths:
                img = load_image(img_path)
                if img is not None:
                    height, width = img.shape[:2]
                    sizes.append({
                        'clase': class_name,
                        'ancho': width,
                        'alto': height,
                        'aspecto': width / height
                    })
    
    return pd.DataFrame(sizes)


# ============================================================
# GUARDADO DE RESULTADOS
# ============================================================

def save_figure(fig, filename: str, results_dir: str = '../results/figures/') -> None:
    """
    Guarda una figura de matplotlib.
    
    Args:
        fig: Figura de matplotlib
        filename (str): Nombre del archivo
        results_dir (str): Directorio de destino
    """
    create_directory(results_dir)
    filepath = os.path.join(results_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {filepath}")


def save_processed_image(image: np.ndarray, 
                        filename: str, 
                        output_dir: str = '../data/processed/') -> None:
    """
    Guarda una imagen procesada.
    
    Args:
        image (np.ndarray): Imagen a guardar
        filename (str): Nombre del archivo
        output_dir (str): Directorio de destino
    """
    create_directory(output_dir)
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Imagen guardada en: {filepath}")


# ============================================================
# UTILIDADES DE LOGGING
# ============================================================

def print_section_header(title: str, char: str = '=') -> None:
    """
    Imprime un encabezado de sección formateado.
    
    Args:
        title (str): Título de la sección
        char (str): Carácter para el borde
    """
    border = char * 60
    print(f"\n{border}")
    print(f"{title.center(60)}")
    print(f"{border}\n")


if __name__ == "__main__":
    print("Módulo de utilidades cargado correctamente.")
    print_section_header("Sistema de Utilidades - Proyecto Clasificación Médica")
