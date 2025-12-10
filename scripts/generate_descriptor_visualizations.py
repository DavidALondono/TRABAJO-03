"""
Script para generar visualizaciones de descriptores de forma y textura.
Muestra ejemplos visuales de HOG, LBP, GLCM, filtros de Gabor y comparaciones entre clases.

Autor: David Londoño
Fecha: Diciembre 10, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from skimage.feature import hog, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import resize_image, apply_clahe, normalize_intensity

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "figures" / "blog"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "chest_xray"

def save_figure(fig, filename, dpi=300):
    """Guarda una figura."""
    filepath = RESULTS_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Guardada: {filename}")
    plt.close(fig)

def preprocess_image(img_path):
    """Carga y preprocesa una imagen."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img = resize_image(img, target_size=(224, 224))
    img = apply_clahe(img)
    img = normalize_intensity(img)
    return img

def generate_hog_visualization():
    """Genera visualización de HOG en imágenes de ambas clases."""
    print("\n🔍 Generando visualización de HOG...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Descriptor HOG (Histogram of Oriented Gradients)', 
                 fontsize=15, fontweight='bold')
    
    classes = ['NORMAL', 'PNEUMONIA']
    
    for idx, cls in enumerate(classes):
        # Cargar imagen
        img_path = list((DATA_DIR / 'train' / cls).glob('*.jpeg'))[8]
        img = preprocess_image(img_path)
        
        # Calcular HOG
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=True)
        
        # Imagen original
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f'Original\n({cls})', fontweight='bold')
        axes[idx, 0].axis('off')
        
        # HOG visualization
        axes[idx, 1].imshow(hog_image, cmap='hot')
        axes[idx, 1].set_title('HOG Features', fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Histograma de orientaciones
        axes[idx, 2].hist(fd, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        axes[idx, 2].set_title('Distribución de Features', fontweight='bold')
        axes[idx, 2].set_xlabel('Valor')
        axes[idx, 2].set_ylabel('Frecuencia')
        axes[idx, 2].grid(axis='y', alpha=0.3)
        
        # Estadísticas
        stats_text = f'Dimensión: {len(fd)}\n'
        stats_text += f'Media: {np.mean(fd):.4f}\n'
        stats_text += f'Std: {np.std(fd):.4f}\n'
        stats_text += f'Max: {np.max(fd):.4f}'
        
        axes[idx, 3].text(0.1, 0.5, stats_text, fontsize=11,
                         verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[idx, 3].axis('off')
        axes[idx, 3].set_title('Estadísticas', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '07_hog_descriptor.png')

def generate_lbp_visualization():
    """Genera visualización de LBP en imágenes de ambas clases."""
    print("\n🎯 Generando visualización de LBP...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Descriptor LBP (Local Binary Patterns)', 
                 fontsize=15, fontweight='bold')
    
    classes = ['NORMAL', 'PNEUMONIA']
    radius = 3
    n_points = 8 * radius
    
    for idx, cls in enumerate(classes):
        # Cargar imagen
        img_path = list((DATA_DIR / 'train' / cls).glob('*.jpeg'))[12]
        img = preprocess_image(img_path)
        
        # Calcular LBP
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        
        # Imagen original
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f'Original\n({cls})', fontweight='bold')
        axes[idx, 0].axis('off')
        
        # LBP image
        axes[idx, 1].imshow(lbp, cmap='viridis')
        axes[idx, 1].set_title(f'LBP Map (R={radius})', fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Histograma LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        axes[idx, 2].bar(range(len(hist)), hist, color='#9b59b6', alpha=0.7)
        axes[idx, 2].set_title('Histograma LBP', fontweight='bold')
        axes[idx, 2].set_xlabel('Patrón')
        axes[idx, 2].set_ylabel('Frecuencia')
        axes[idx, 2].grid(axis='y', alpha=0.3)
        
        # Zoom de una región
        y, x = 80, 80
        size = 40
        zoom = img[y:y+size, x:x+size]
        axes[idx, 3].imshow(zoom, cmap='gray')
        axes[idx, 3].set_title('Zoom: Textura Local', fontweight='bold')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    save_figure(fig, '08_lbp_descriptor.png')

def generate_glcm_visualization():
    """Genera visualización de GLCM y características de Haralick."""
    print("\n📐 Generando visualización de GLCM...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Matriz de Co-ocurrencia (GLCM) y Características de Haralick', 
                 fontsize=15, fontweight='bold')
    
    classes = ['NORMAL', 'PNEUMONIA']
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for idx, cls in enumerate(classes):
        # Cargar imagen
        img_path = list((DATA_DIR / 'train' / cls).glob('*.jpeg'))[15]
        img = preprocess_image(img_path)
        
        # Reducir niveles de gris para GLCM
        img_reduced = (img / 16).astype(np.uint8)
        
        # Calcular GLCM
        glcm = graycomatrix(img_reduced, distances, angles, 
                           levels=16, symmetric=True, normed=True)
        
        # Imagen original
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f'Original\n({cls})', fontweight='bold')
        axes[idx, 0].axis('off')
        
        # GLCM para un ángulo
        axes[idx, 1].imshow(glcm[:, :, 0, 0], cmap='hot', interpolation='nearest')
        axes[idx, 1].set_title('GLCM (0°)', fontweight='bold')
        axes[idx, 1].set_xlabel('Intensidad j')
        axes[idx, 1].set_ylabel('Intensidad i')
        
        # Características de Haralick
        features = {
            'Contraste': graycoprops(glcm, 'contrast')[0, :],
            'Homogeneidad': graycoprops(glcm, 'homogeneity')[0, :],
            'Energía': graycoprops(glcm, 'energy')[0, :],
            'Correlación': graycoprops(glcm, 'correlation')[0, :]
        }
        
        # Gráfico de características por dirección
        x = ['0°', '45°', '90°', '135°']
        width = 0.2
        x_pos = np.arange(len(x))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, (feat_name, feat_values) in enumerate(features.items()):
            axes[idx, 2].bar(x_pos + i*width, feat_values, width, 
                            label=feat_name, color=colors[i], alpha=0.7)
        
        axes[idx, 2].set_xlabel('Dirección', fontweight='bold')
        axes[idx, 2].set_ylabel('Valor', fontweight='bold')
        axes[idx, 2].set_title('Características de Haralick', fontweight='bold')
        axes[idx, 2].set_xticks(x_pos + width * 1.5)
        axes[idx, 2].set_xticklabels(x)
        axes[idx, 2].legend(fontsize=8)
        axes[idx, 2].grid(axis='y', alpha=0.3)
        
        # Resumen estadístico
        summary_text = 'Promedios:\n'
        for feat_name, feat_values in features.items():
            summary_text += f'{feat_name}: {np.mean(feat_values):.3f}\n'
        
        axes[idx, 3].text(0.1, 0.5, summary_text, fontsize=10,
                         verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[idx, 3].axis('off')
        axes[idx, 3].set_title('Resumen', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, '09_glcm_haralick.png')

def generate_gabor_visualization():
    """Genera visualización de filtros de Gabor."""
    print("\n🌊 Generando visualización de Filtros de Gabor...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)
    fig.suptitle('Banco de Filtros de Gabor y Respuestas', 
                 fontsize=15, fontweight='bold')
    
    # Cargar imagen
    img_path = list((DATA_DIR / 'train' / 'PNEUMONIA').glob('*.jpeg'))[20]
    img = preprocess_image(img_path)
    
    # Imagen original
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title('Original\n(PNEUMONIA)', fontweight='bold')
    ax_orig.axis('off')
    
    # Crear y aplicar filtros de Gabor
    ksize = 31
    sigma = 4.0
    lambd = 10.0
    gamma = 0.5
    
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequencies = [0.1, 0.2, 0.3, 0.4]
    
    responses = []
    
    # Mostrar algunos filtros y sus respuestas
    for idx, theta in enumerate(thetas):
        # Kernel del filtro
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        
        # Filtro
        ax_filter = fig.add_subplot(gs[0, idx+1])
        ax_filter.imshow(kernel, cmap='RdBu_r')
        ax_filter.set_title(f'Filtro {int(np.degrees(theta))}°', fontweight='bold', fontsize=10)
        ax_filter.axis('off')
        
        # Respuesta filtrada
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        responses.append(filtered)
        
        ax_resp = fig.add_subplot(gs[1, idx+1])
        ax_resp.imshow(filtered, cmap='viridis')
        ax_resp.set_title(f'Respuesta {int(np.degrees(theta))}°', fontweight='bold', fontsize=10)
        ax_resp.axis('off')
    
    # Estadísticas de respuestas
    ax_stats = fig.add_subplot(gs[1, 0])
    angles_deg = [int(np.degrees(t)) for t in thetas]
    means = [np.abs(r).mean() for r in responses]
    stds = [np.abs(r).std() for r in responses]
    
    x_pos = np.arange(len(angles_deg))
    ax_stats.bar(x_pos, means, yerr=stds, alpha=0.7, color='#3498db', 
                capsize=5, edgecolor='black', linewidth=1.5)
    ax_stats.set_xticks(x_pos)
    ax_stats.set_xticklabels([f'{a}°' for a in angles_deg])
    ax_stats.set_xlabel('Orientación', fontweight='bold')
    ax_stats.set_ylabel('Energía Media', fontweight='bold')
    ax_stats.set_title('Energía por Dirección', fontweight='bold')
    ax_stats.grid(axis='y', alpha=0.3)
    
    # Respuestas combinadas
    ax_comb = fig.add_subplot(gs[2, :3])
    combined = np.zeros_like(img, dtype=np.float32)
    for resp in responses:
        combined += np.abs(resp)
    ax_comb.imshow(combined, cmap='hot')
    ax_comb.set_title('Respuesta Combinada (todas las orientaciones)', fontweight='bold')
    ax_comb.axis('off')
    
    # Texto explicativo
    ax_text = fig.add_subplot(gs[2, 3:])
    explanation = (
        'Filtros de Gabor:\n\n'
        '• Capturan patrones de textura\n'
        '  direccionales\n\n'
        '• Múltiples orientaciones\n'
        '  (0°, 45°, 90°, 135°)\n\n'
        '• Múltiples frecuencias\n\n'
        '• Sensibles a estructuras\n'
        '  lineales y bordes\n\n'
        '• Útiles para detectar\n'
        '  infiltrados pulmonares'
    )
    ax_text.text(0.1, 0.5, explanation, fontsize=10,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_text.axis('off')
    
    save_figure(fig, '10_gabor_filters.png')

def generate_descriptor_comparison():
    """Genera comparación de todos los descriptores en una misma imagen."""
    print("\n🎨 Generando comparación de descriptores...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Comparación de Descriptores en la Misma Imagen', 
                 fontsize=15, fontweight='bold')
    
    # Cargar dos imágenes (una de cada clase)
    normal_path = list((DATA_DIR / 'train' / 'NORMAL').glob('*.jpeg'))[10]
    pneumonia_path = list((DATA_DIR / 'train' / 'PNEUMONIA').glob('*.jpeg'))[10]
    
    for row_idx, (img_path, cls) in enumerate([(normal_path, 'NORMAL'), 
                                                 (pneumonia_path, 'PNEUMONIA')]):
        if row_idx >= 2:  # Solo dos filas
            break
            
        img = preprocess_image(img_path)
        
        # Original
        axes[row_idx, 0].imshow(img, cmap='gray')
        axes[row_idx, 0].set_title(f'Original\n({cls})', fontweight='bold')
        axes[row_idx, 0].axis('off')
        
        # HOG
        _, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
        axes[row_idx, 1].imshow(hog_img, cmap='hot')
        axes[row_idx, 1].set_title('HOG', fontweight='bold')
        axes[row_idx, 1].axis('off')
        
        # LBP
        lbp = local_binary_pattern(img, 24, 3, method='uniform')
        axes[row_idx, 2].imshow(lbp, cmap='viridis')
        axes[row_idx, 2].set_title('LBP', fontweight='bold')
        axes[row_idx, 2].axis('off')
        
        # Gabor (orientación 0)
        kernel = cv2.getGaborKernel((31, 31), 4.0, 0, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_resp = cv2.filter2D(img, cv2.CV_32F, kernel)
        axes[row_idx, 3].imshow(np.abs(gabor_resp), cmap='viridis')
        axes[row_idx, 3].set_title('Gabor (0°)', fontweight='bold')
        axes[row_idx, 3].axis('off')
    
    # Tercera fila: comparación de histogramas
    for col_idx, descriptor in enumerate(['Original', 'HOG', 'LBP', 'Gabor']):
        axes[2, col_idx].text(0.5, 0.5, f'{descriptor}\nDescriptor',
                             ha='center', va='center', fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        axes[2, col_idx].axis('off')
    
    plt.tight_layout()
    save_figure(fig, '11_descriptor_comparison.png')

def main():
    """Ejecuta la generación de visualizaciones de descriptores."""
    print("=" * 70)
    print("  GENERADOR DE VISUALIZACIONES DE DESCRIPTORES")
    print("=" * 70)
    print(f"\n📁 Directorio de salida: {RESULTS_DIR}\n")
    
    try:
        generate_hog_visualization()
        generate_lbp_visualization()
        generate_glcm_visualization()
        generate_gabor_visualization()
        generate_descriptor_comparison()
        
        print("\n" + "=" * 70)
        print("  ✅ VISUALIZACIONES DE DESCRIPTORES GENERADAS")
        print("=" * 70)
        print(f"\n📊 Total de figuras generadas: 5")
        print(f"📁 Ubicación: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
