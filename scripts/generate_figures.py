"""
Script para generar figuras complementarias del reporte técnico.
Este script crea visualizaciones adicionales para el blog post.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Crear directorio para figuras si no existe
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

def create_pipeline_overview():
    """Crear diagrama visual del pipeline general."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Definir etapas
    stages = [
        ('Dataset Raw', 1, '#E8F4F8'),
        ('Preprocesamiento', 2.5, '#B8E6F0'),
        ('Extracción Features', 4, '#88D8E8'),
        ('Clasificación', 5.5, '#58CAE0'),
        ('Evaluación', 7, '#28BCD8'),
        ('Resultados', 8.5, '#0097A7')
    ]
    
    # Dibujar cajas y flechas
    for i, (stage, x, color) in enumerate(stages):
        # Caja
        rect = patches.FancyBboxPatch((x-0.4, 1.5), 0.8, 1, 
                                      boxstyle="round,pad=0.1",
                                      edgecolor='black', facecolor=color, 
                                      linewidth=2)
        ax.add_patch(rect)
        
        # Texto
        ax.text(x, 2, stage, ha='center', va='center', 
                fontsize=11, fontweight='bold', wrap=True)
        
        # Flecha
        if i < len(stages) - 1:
            ax.annotate('', xy=(stages[i+1][1]-0.45, 2), xytext=(x+0.45, 2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_title('Pipeline General de Clasificación de Radiografías de Tórax',
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pipeline_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Generada: pipeline_overview.png")
    plt.close()


def create_preprocessing_steps():
    """Visualización de las etapas de preprocesamiento."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    steps = [
        'Original\n(Variable size)',
        'Resize\n224×224',
        'CLAHE\nContraste mejorado',
        'Normalización\n[0, 1]'
    ]
    
    colors = ['#FFE5E5', '#FFD6A5', '#CAFFBF', '#9BF6FF']
    
    for i, (ax, step, color) in enumerate(zip(axes, steps, colors)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Simular imagen
        rect = patches.Rectangle((0.15, 0.2), 0.7, 0.6, 
                                 facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Texto
        ax.text(0.5, 0.9, step, ha='center', va='center',
               fontsize=12, fontweight='bold')
        
        # Flecha
        if i < len(steps) - 1:
            ax.annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5),
                       xycoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=3, color='#0097A7'))
    
    fig.suptitle('Etapas del Pipeline de Preprocesamiento',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'preprocessing_steps.png', dpi=300, bbox_inches='tight')
    print("✓ Generada: preprocessing_steps.png")
    plt.close()


def create_feature_extraction_diagram():
    """Diagrama de extracción de características."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Imagen preprocesada (centro superior)
    rect = patches.FancyBboxPatch((3.5, 10), 3, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#E8F4F8', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 10.75, 'Imagen Preprocesada\n224×224', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    # Descriptores de forma (izquierda)
    shape_features = [
        ('HOG', 1, 7.5, '#FFE5E5'),
        ('Momentos Hu', 1, 6, '#FFD6A5'),
        ('Contorno', 1, 4.5, '#CAFFBF')
    ]
    
    ax.text(1.5, 9, 'Descriptores\nde Forma', ha='center', va='center',
           fontsize=13, fontweight='bold', bbox=dict(boxstyle='round', 
           facecolor='#F0F0F0', edgecolor='black', linewidth=2))
    
    for name, x, y, color in shape_features:
        rect = patches.FancyBboxPatch((x-0.4, y-0.3), 1.8, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.5, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Flecha desde imagen
        ax.annotate('', xy=(x+0.9, y+0.2), xytext=(3.5, 10.3),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # Descriptores de textura (derecha)
    texture_features = [
        ('LBP', 8, 8, '#9BF6FF'),
        ('GLCM', 8, 6.8, '#A0C4FF'),
        ('Gabor', 8, 5.6, '#BDB2FF'),
        ('Estadísticas', 8, 4.4, '#FFC6FF')
    ]
    
    ax.text(8.5, 9, 'Descriptores\nde Textura', ha='center', va='center',
           fontsize=13, fontweight='bold', bbox=dict(boxstyle='round',
           facecolor='#F0F0F0', edgecolor='black', linewidth=2))
    
    for name, x, y, color in texture_features:
        rect = patches.FancyBboxPatch((x-0.4, y-0.3), 1.8, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.5, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Flecha desde imagen
        ax.annotate('', xy=(x+0.1, y+0.2), xytext=(6.5, 10.3),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # Vector de características concatenado (abajo)
    rect = patches.FancyBboxPatch((2, 2), 6, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFFACD', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 2.6, 'Vector de Características Concatenado', ha='center', va='center',
           fontsize=12, fontweight='bold')
    ax.text(5, 2.2, 'Dimensión: 1,764 (HOG) + 59 (LBP) + 13 (GLCM) + ...', 
           ha='center', va='center', fontsize=9)
    
    # Flechas de concatenación
    for x in [1.5, 8.5]:
        ax.annotate('', xy=(5, 3.2), xytext=(x, 4.2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#0097A7'))
    
    # Normalización
    ax.annotate('', xy=(5, 0.8), xytext=(5, 2),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    rect = patches.FancyBboxPatch((3, 0), 4, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#90EE90', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 0.4, 'Normalización (StandardScaler)', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    ax.set_title('Extracción y Concatenación de Descriptores', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_extraction_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Generada: feature_extraction_diagram.png")
    plt.close()


def create_classification_workflow():
    """Workflow de clasificación y evaluación."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Features normalizadas
    rect = patches.FancyBboxPatch((3.5, 9.5), 3, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#90EE90', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 10, 'Features Normalizadas', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    # Split train/test
    ax.annotate('', xy=(3, 8.5), xytext=(4.5, 9.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7, 8.5), xytext=(5.5, 9.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Train
    rect = patches.FancyBboxPatch((1.5, 7.5), 2.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFE5E5', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(2.75, 8, 'Train Set\n80%', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Test
    rect = patches.FancyBboxPatch((6, 7.5), 2.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#CAFFBF', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.25, 8, 'Test Set\n20%', ha='center', va='center',
           fontsize=11, fontweight='bold')
    
    # Clasificadores
    classifiers = [
        ('SVM RBF', 0.8, 5.5, '#A0C4FF'),
        ('Random Forest', 2.3, 5.5, '#BDB2FF'),
        ('k-NN', 3.8, 5.5, '#FFC6FF'),
        ('Log. Reg.', 5.3, 5.5, '#FFFFB3')
    ]
    
    ax.text(3, 6.5, 'Entrenamiento', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    for name, x, y, color in classifiers:
        rect = patches.FancyBboxPatch((x, y), 1.3, 0.7,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.65, y+0.35, name, ha='center', va='center',
               fontsize=9, fontweight='bold')
        
        # Flecha desde train
        ax.annotate('', xy=(x+0.65, y+0.7), xytext=(2.75, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # Predicciones
    ax.text(7.25, 6.5, 'Predicción', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    for i, (name, x, y, color) in enumerate(classifiers):
        pred_x = 6.8 + i * 0.35
        rect = patches.FancyBboxPatch((pred_x, y), 0.3, 0.7,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Flecha de predicción
        ax.annotate('', xy=(pred_x+0.15, y+0.7), xytext=(x+0.65, y),
                   arrowprops=dict(arrowstyle='->', lw=1.2, color='gray'))
        
        # Flecha desde test
        ax.annotate('', xy=(pred_x+0.15, y+0.7), xytext=(7.25, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=1.2, color='green', alpha=0.5))
    
    # Evaluación
    rect = patches.FancyBboxPatch((3, 3.5), 4, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFFACD', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 4.3, 'Evaluación de Modelos', ha='center', va='center',
           fontsize=12, fontweight='bold')
    ax.text(5, 3.9, 'Accuracy, Precision, Recall, F1-Score, AUC', ha='center', va='center',
           fontsize=9)
    
    # Flechas a evaluación
    for i in range(4):
        ax.annotate('', xy=(4 + i*0.5, 4.7), xytext=(6.95 + i*0.35, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#0097A7'))
    
    # Resultados
    results = [
        ('Matrices de\nConfusión', 1.5, 1.5, '#FFE5E5'),
        ('Curvas ROC', 3.8, 1.5, '#CAFFBF'),
        ('Comparación\nMétricas', 6.1, 1.5, '#9BF6FF')
    ]
    
    for name, x, y, color in results:
        rect = patches.FancyBboxPatch((x, y), 1.8, 1,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.9, y+0.5, name, ha='center', va='center',
               fontsize=10, fontweight='bold')
        
        # Flecha desde evaluación
        ax.annotate('', xy=(x+0.9, y+1), xytext=(5, 3.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # Mejor modelo
    rect = patches.FancyBboxPatch((3.5, 0), 3, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFD700', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 0.4, 'Mejor Modelo: SVM RBF\nAccuracy: 95.51%', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.set_title('Workflow de Clasificación y Evaluación',
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_workflow.png', dpi=300, bbox_inches='tight')
    print("✓ Generada: classification_workflow.png")
    plt.close()


def create_methodology_summary():
    """Resumen visual de la metodología completa."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Título principal
    fig.suptitle('Metodología Completa del Proyecto',
                fontsize=18, fontweight='bold', y=0.98)
    
    # Parte 1: Preprocesamiento
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 2)
    ax1.axis('off')
    ax1.text(5, 1.7, 'PARTE 1: PREPROCESAMIENTO', ha='center', va='center',
            fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='#E8F4F8', edgecolor='black', linewidth=2))
    
    steps1 = ['Dataset', 'Resize', 'CLAHE', 'Normalización']
    for i, step in enumerate(steps1):
        x = 1.5 + i * 2
        rect = patches.FancyBboxPatch((x-0.3, 0.3), 0.6, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#B8E6F0', edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(x, 0.7, step, ha='center', va='center', fontsize=9, fontweight='bold')
        if i < len(steps1) - 1:
            ax1.annotate('', xy=(x+0.5, 0.7), xytext=(x+0.3, 0.7),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Parte 2: Descriptores
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 2)
    ax2.axis('off')
    ax2.text(5, 1.7, 'PARTE 2: EXTRACCIÓN DE DESCRIPTORES', ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE5E5', edgecolor='black', linewidth=2))
    
    ax2.text(2.5, 1.1, 'Forma:', ha='center', va='center', fontsize=11, fontweight='bold')
    shape_desc = ['HOG', 'Hu', 'Contorno']
    for i, desc in enumerate(shape_desc):
        x = 1 + i * 0.8
        rect = patches.Rectangle((x, 0.3), 0.6, 0.5, facecolor='#FFD6A5', 
                                edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
        ax2.text(x+0.3, 0.55, desc, ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax2.text(7.5, 1.1, 'Textura:', ha='center', va='center', fontsize=11, fontweight='bold')
    texture_desc = ['LBP', 'GLCM', 'Gabor', 'Stats']
    for i, desc in enumerate(texture_desc):
        x = 5.5 + i * 0.8
        rect = patches.Rectangle((x, 0.3), 0.6, 0.5, facecolor='#CAFFBF',
                                edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
        ax2.text(x+0.3, 0.55, desc, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Parte 3: Clasificación
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 2)
    ax3.axis('off')
    ax3.text(5, 1.7, 'PARTE 3: CLASIFICACIÓN Y EVALUACIÓN', ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#9BF6FF', edgecolor='black', linewidth=2))
    
    classifiers = ['SVM', 'RF', 'k-NN', 'LogReg']
    colors = ['#A0C4FF', '#BDB2FF', '#FFC6FF', '#FFFFB3']
    for i, (clf, color) in enumerate(zip(classifiers, colors)):
        x = 1.5 + i * 1.5
        rect = patches.FancyBboxPatch((x-0.3, 0.7), 0.6, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
        ax3.add_patch(rect)
        ax3.text(x, 0.95, clf, ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax3.text(8, 0.95, '→ Evaluación', ha='center', va='center',
            fontsize=11, fontweight='bold')
    rect = patches.FancyBboxPatch((7.2, 0.2), 1.6, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#FFD700', edgecolor='black', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(8, 0.4, 'Mejor: SVM 95.51%', ha='center', va='center',
            fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'methodology_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Generada: methodology_summary.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" Generando Figuras Complementarias para el Reporte Técnico")
    print("="*60 + "\n")
    
    create_pipeline_overview()
    create_preprocessing_steps()
    create_feature_extraction_diagram()
    create_classification_workflow()
    create_methodology_summary()
    
    print("\n" + "="*60)
    print(" ✓ Todas las figuras fueron generadas exitosamente")
    print(f" ✓ Ubicación: {output_dir.absolute()}")
    print("="*60 + "\n")
