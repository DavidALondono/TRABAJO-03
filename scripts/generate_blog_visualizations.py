"""
Script para generar todas las visualizaciones necesarias para el reporte técnico tipo blog.
Este script extrae visualizaciones clave de los notebooks y genera figuras adicionales
para documentar completamente el proceso de análisis.

Autor: David Londoño
Fecha: Diciembre 10, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from pathlib import Path
import sys

# Configurar el path para importar módulos del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import resize_image, apply_clahe, normalize_intensity

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Directorios
RESULTS_DIR = Path(__file__).parent.parent / "results" / "figures" / "blog"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

def save_figure(fig, filename, dpi=300):
    """Guarda una figura con alta calidad."""
    filepath = RESULTS_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Guardada: {filename}")
    plt.close(fig)

def generate_class_distribution():
    """Genera visualización de distribución de clases en train/val/test."""
    print("\n📊 Generando distribución de clases...")
    
    # Contar imágenes en cada conjunto
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    counts = {}
    for split in splits:
        counts[split] = {}
        for cls in classes:
            path = DATA_DIR / split / cls
            if path.exists():
                counts[split][cls] = len(list(path.glob('*.jpeg')))
            else:
                counts[split][cls] = 0
    
    # Crear visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Barras agrupadas
    x = np.arange(len(splits))
    width = 0.35
    
    normal_counts = [counts[split]['NORMAL'] for split in splits]
    pneumonia_counts = [counts[split]['PNEUMONIA'] for split in splits]
    
    axes[0].bar(x - width/2, normal_counts, width, label='NORMAL', color='#2ecc71', alpha=0.8)
    axes[0].bar(x + width/2, pneumonia_counts, width, label='PNEUMONIA', color='#e74c3c', alpha=0.8)
    
    axes[0].set_xlabel('Conjunto de Datos', fontweight='bold')
    axes[0].set_ylabel('Número de Imágenes', fontweight='bold')
    axes[0].set_title('Distribución de Clases por Conjunto', fontweight='bold', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Entrenamiento', 'Validación', 'Prueba'])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Agregar valores encima de las barras
    for i, (n, p) in enumerate(zip(normal_counts, pneumonia_counts)):
        axes[0].text(i - width/2, n + 50, str(n), ha='center', va='bottom', fontweight='bold')
        axes[0].text(i + width/2, p + 50, str(p), ha='center', va='bottom', fontweight='bold')
    
    # Gráfico 2: Pie chart del conjunto de entrenamiento
    train_total = counts['train']['NORMAL'] + counts['train']['PNEUMONIA']
    train_percentages = [
        (counts['train']['NORMAL'] / train_total) * 100,
        (counts['train']['PNEUMONIA'] / train_total) * 100
    ]
    
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    axes[1].pie(train_percentages, labels=['NORMAL', 'PNEUMONIA'], autopct='%1.1f%%',
                colors=colors, explode=explode, shadow=True, startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Proporción de Clases en Entrenamiento\n(Desbalance de Clases)', 
                      fontweight='bold', fontsize=13)
    
    plt.suptitle('Análisis Exploratorio: Distribución del Dataset', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    save_figure(fig, '01_class_distribution.png')

def generate_sample_images():
    """Genera grilla de imágenes de ejemplo de ambas clases."""
    print("\n🖼️  Generando ejemplos de imágenes...")
    
    fig, axes = plt.subplots(4, 6, figsize=(16, 11))
    fig.suptitle('Ejemplos de Radiografías: NORMAL vs PNEUMONIA', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Cargar ejemplos de cada clase
    for row_idx, cls in enumerate(['NORMAL', 'PNEUMONIA']):
        image_dir = DATA_DIR / 'train' / cls
        image_files = sorted(list(image_dir.glob('*.jpeg')))[:6]  # 6 ejemplos
        
        for col_idx, img_file in enumerate(image_files):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            
            # Fila superior: NORMAL
            ax = axes[row_idx*2, col_idx]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(cls, fontsize=12, fontweight='bold', rotation=0, 
                            labelpad=40, va='center')
            
            # Fila inferior: mismo con marcador
            ax = axes[row_idx*2 + 1, col_idx]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f'{img.shape[1]}×{img.shape[0]}', fontsize=9, pad=3)
    
    plt.tight_layout()
    save_figure(fig, '02_sample_images.png')

def generate_size_distribution():
    """Genera distribución de tamaños de imágenes."""
    print("\n📐 Generando distribución de tamaños...")
    
    # Recolectar tamaños
    widths, heights = [], []
    
    train_normal = DATA_DIR / 'train' / 'NORMAL'
    train_pneumonia = DATA_DIR / 'train' / 'PNEUMONIA'
    
    # Muestrear 200 imágenes de cada clase
    for img_path in list(train_normal.glob('*.jpeg'))[:200]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            widths.append(w)
            heights.append(h)
    
    for img_path in list(train_pneumonia.glob('*.jpeg'))[:200]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            widths.append(w)
            heights.append(h)
    
    # Crear visualización
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Scatter plot
    axes[0].scatter(widths, heights, alpha=0.5, s=20, c='#3498db')
    axes[0].set_xlabel('Ancho (píxeles)', fontweight='bold')
    axes[0].set_ylabel('Alto (píxeles)', fontweight='bold')
    axes[0].set_title('Distribución de Dimensiones Originales', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(224, color='red', linestyle='--', linewidth=2, label='Target: 224×224', alpha=0.7)
    axes[0].axvline(224, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].legend()
    
    # Histograma de anchos
    axes[1].hist(widths, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    axes[1].axvline(224, color='red', linestyle='--', linewidth=2, label='Target: 224')
    axes[1].set_xlabel('Ancho (píxeles)', fontweight='bold')
    axes[1].set_ylabel('Frecuencia', fontweight='bold')
    axes[1].set_title('Distribución de Anchos', fontweight='bold', fontsize=12)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Histograma de altos
    axes[2].hist(heights, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[2].axvline(224, color='red', linestyle='--', linewidth=2, label='Target: 224')
    axes[2].set_xlabel('Alto (píxeles)', fontweight='bold')
    axes[2].set_ylabel('Frecuencia', fontweight='bold')
    axes[2].set_title('Distribución de Altos', fontweight='bold', fontsize=12)
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Análisis de Variabilidad Dimensional (muestra de 400 imágenes)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_figure(fig, '03_size_distribution.png')

def generate_preprocessing_comparison():
    """Genera comparación antes/después del preprocesamiento."""
    print("\n🔧 Generando comparación de preprocesamiento...")
    
    # Seleccionar una imagen de ejemplo
    img_path = list((DATA_DIR / 'train' / 'PNEUMONIA').glob('*.jpeg'))[10]
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    # Aplicar pipeline de preprocesamiento paso a paso
    step1_resized = resize_image(original, target_size=(224, 224))
    step2_clahe = apply_clahe(step1_resized)
    step3_normalized = normalize_intensity(step2_clahe)
    
    # Crear visualización
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    images = [
        (original, 'Original'),
        (step1_resized, 'Resize 224×224'),
        (step2_clahe, 'CLAHE'),
        (step3_normalized, 'Normalizado')
    ]
    
    # Fila superior: imágenes
    for idx, (img, title) in enumerate(images):
        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].set_title(title, fontweight='bold', fontsize=11)
        axes[0, idx].axis('off')
    
    # Fila inferior: histogramas
    for idx, (img, title) in enumerate(images):
        if img is not None and img.size > 0:
            hist, bins = np.histogram(img.flatten(), bins=50, range=(0, 255))
            axes[1, idx].plot(bins[:-1], hist, color='#3498db', linewidth=2)
            axes[1, idx].fill_between(bins[:-1], hist, alpha=0.3, color='#3498db')
            axes[1, idx].set_xlabel('Intensidad', fontweight='bold', fontsize=9)
            axes[1, idx].set_ylabel('Frecuencia', fontweight='bold', fontsize=9)
            axes[1, idx].set_title(f'Histograma: {title}', fontsize=10)
            axes[1, idx].grid(True, alpha=0.3)
    
    plt.suptitle('Pipeline de Preprocesamiento: Transformación Paso a Paso', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    save_figure(fig, '04_preprocessing_pipeline.png')

def generate_clahe_comparison():
    """Genera comparación entre CLAHE y ecualización estándar."""
    print("\n⚡ Generando comparación CLAHE vs Ecualización...")
    
    # Seleccionar imagen
    img_path = list((DATA_DIR / 'train' / 'NORMAL').glob('*.jpeg'))[15]
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    resized = resize_image(original, target_size=(224, 224))
    
    # Aplicar técnicas
    clahe_applied = apply_clahe(resized)
    
    # Ecualización de histograma estándar
    equalized = cv2.equalizeHist(resized)
    
    # Crear visualización
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    images = [
        (resized, 'Original'),
        (clahe_applied, 'CLAHE\n(clipLimit=2.0, tileGridSize=8×8)'),
        (equalized, 'Ecualización\nEstándar')
    ]
    
    # Fila superior: imágenes
    for idx, (img, title) in enumerate(images):
        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].set_title(title, fontweight='bold', fontsize=11, pad=10)
        axes[0, idx].axis('off')
    
    # Fila inferior: histogramas
    colors = ['#95a5a6', '#27ae60', '#e74c3c']
    for idx, (img, title) in enumerate(images):
        hist, bins = np.histogram(img.flatten(), bins=50, range=(0, 255))
        axes[1, idx].plot(bins[:-1], hist, color=colors[idx], linewidth=2)
        axes[1, idx].fill_between(bins[:-1], hist, alpha=0.3, color=colors[idx])
        axes[1, idx].set_xlabel('Intensidad', fontweight='bold')
        axes[1, idx].set_ylabel('Frecuencia', fontweight='bold')
        axes[1, idx].set_title(f'Histograma', fontsize=10)
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].set_ylim(0, max(hist) * 1.1)
    
    plt.suptitle('Comparación de Técnicas de Mejora de Contraste', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Texto explicativo
    fig.text(0.5, 0.02, 
             'CLAHE preserva mejor las estructuras anatómicas y evita la amplificación excesiva de ruido\n' +
             'comparado con la ecualización de histograma estándar',
             ha='center', fontsize=10, style='italic', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_figure(fig, '05_clahe_vs_equalization.png')

def generate_before_after_grid():
    """Genera grilla comparativa de múltiples imágenes antes/después."""
    print("\n🎨 Generando grilla antes/después...")
    
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle('Comparación Antes/Después del Preprocesamiento Completo', 
                 fontsize=15, fontweight='bold')
    
    # 2 imágenes de cada clase
    normal_images = list((DATA_DIR / 'train' / 'NORMAL').glob('*.jpeg'))[:2]
    pneumonia_images = list((DATA_DIR / 'train' / 'PNEUMONIA').glob('*.jpeg'))[:2]
    
    all_images = normal_images + pneumonia_images
    
    for idx, img_path in enumerate(all_images):
        # Original
        original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        original_resized = resize_image(original, target_size=(224, 224))
        
        # Preprocesada
        preprocessed = apply_clahe(original_resized)
        preprocessed = normalize_intensity(preprocessed)
        
        # Mostrar
        row = idx
        
        axes[row, 0].imshow(original_resized, cmap='gray')
        axes[row, 0].set_title('Original', fontweight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(preprocessed, cmap='gray')
        axes[row, 1].set_title('Preprocesada', fontweight='bold')
        axes[row, 1].axis('off')
        
        # Histogramas
        hist_orig = np.histogram(original_resized.flatten(), bins=30, range=(0, 255))[0]
        hist_prep = np.histogram(preprocessed.flatten(), bins=30, range=(0, 255))[0]
        
        axes[row, 2].bar(range(30), hist_orig, color='#95a5a6', alpha=0.7)
        axes[row, 2].set_title('Hist. Original', fontsize=9)
        axes[row, 2].set_xlim(0, 30)
        axes[row, 2].tick_params(labelsize=7)
        
        axes[row, 3].bar(range(30), hist_prep, color='#27ae60', alpha=0.7)
        axes[row, 3].set_title('Hist. Preprocesado', fontsize=9)
        axes[row, 3].set_xlim(0, 30)
        axes[row, 3].tick_params(labelsize=7)
        
        # Etiqueta de clase
        cls = 'NORMAL' if idx < 2 else 'PNEUMONIA'
        axes[row, 0].text(-0.2, 0.5, f'{cls}\n#{idx%2 + 1}', 
                         transform=axes[row, 0].transAxes,
                         fontsize=10, fontweight='bold', va='center', 
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    save_figure(fig, '06_before_after_grid.png')

def main():
    """Ejecuta la generación de todas las visualizaciones."""
    print("=" * 70)
    print("  GENERADOR DE VISUALIZACIONES PARA REPORTE TÉCNICO TIPO BLOG")
    print("=" * 70)
    print(f"\n📁 Directorio de salida: {RESULTS_DIR}\n")
    
    try:
        # Visualizaciones de exploración de datos
        generate_class_distribution()
        generate_sample_images()
        generate_size_distribution()
        
        # Visualizaciones de preprocesamiento
        generate_preprocessing_comparison()
        generate_clahe_comparison()
        generate_before_after_grid()
        
        print("\n" + "=" * 70)
        print("  ✅ TODAS LAS VISUALIZACIONES GENERADAS EXITOSAMENTE")
        print("=" * 70)
        print(f"\n📊 Total de figuras generadas: 6")
        print(f"📁 Ubicación: {RESULTS_DIR}")
        print("\nPróximo paso: Ejecutar generate_descriptor_visualizations.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
