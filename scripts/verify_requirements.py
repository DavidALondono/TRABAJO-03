#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de verificación para asegurar que el repositorio cumple
con todos los requisitos del reporte técnico (blog post).
"""

from pathlib import Path
import re

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_file_exists(filepath, description):
    """Verifica si un archivo existe."""
    path = Path(filepath)
    if path.exists():
        print(f"{Colors.GREEN}✓{Colors.END} {description}: {filepath}")
        return True
    else:
        print(f"{Colors.RED}✗{Colors.END} {description}: {filepath} {Colors.RED}(NO ENCONTRADO){Colors.END}")
        return False

def check_markdown_sections(filepath, required_sections):
    """Verifica que el Markdown contenga las secciones requeridas."""
    path = Path(filepath)
    if not path.exists():
        return False
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    missing = []
    for section in required_sections:
        # Buscar variantes del título de sección
        patterns = [
            f"## {section}",
            f"### {section}",
            section.lower(),
        ]
        if not any(pattern.lower() in content.lower() for pattern in patterns):
            missing.append(section)
    
    return missing

def check_references_count(filepath, min_count=5):
    """Verifica que haya al menos min_count referencias."""
    path = Path(filepath)
    if not path.exists():
        return 0
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar sección de referencias
    ref_section = re.search(r'## Referencias.*?(?=##|\Z)', content, re.DOTALL | re.IGNORECASE)
    if not ref_section:
        return 0
    
    # Contar líneas numeradas
    references = re.findall(r'^\d+\.', ref_section.group(), re.MULTILINE)
    return len(references)

def check_figures_directory():
    """Verifica las figuras generadas."""
    figures_dir = Path('results/figures')
    if not figures_dir.exists():
        return []
    
    expected_figures = [
        'confusion_matrices.png',
        'metrics_comparison.png',
        'roc_curves.png',
        'pipeline_overview.png',
        'preprocessing_steps.png',
        'feature_extraction_diagram.png',
        'classification_workflow.png',
        'methodology_summary.png'
    ]
    
    found = []
    missing = []
    for fig in expected_figures:
        if (figures_dir / fig).exists():
            found.append(fig)
        else:
            missing.append(fig)
    
    return found, missing

def main():
    print("\n" + "="*70)
    print(f"{Colors.BOLD}{Colors.BLUE}VERIFICACIÓN DE REQUISITOS DEL REPORTE TÉCNICO (BLOG POST){Colors.END}")
    print("="*70 + "\n")
    
    all_checks = []
    
    # 1. VERIFICAR ARCHIVOS PRINCIPALES
    print(f"{Colors.BOLD}1. ARCHIVOS PRINCIPALES{Colors.END}")
    print("-" * 70)
    
    files_to_check = [
        ('reporte_tecnico_trabajo3.md', 'Reporte en Markdown'),
        ('reporte_tecnico_trabajo3.html', 'Reporte en HTML (Blog Post)'),
        ('README.md', 'README principal'),
        ('requirements.txt', 'Dependencias'),
        ('.gitignore', 'Configuración Git'),
    ]
    
    for filepath, desc in files_to_check:
        all_checks.append(check_file_exists(filepath, desc))
    
    # 2. VERIFICAR ESTRUCTURA DE DIRECTORIOS
    print(f"\n{Colors.BOLD}2. ESTRUCTURA DE DIRECTORIOS{Colors.END}")
    print("-" * 70)
    
    directories = [
        ('src', 'Código fuente'),
        ('notebooks', 'Notebooks de análisis'),
        ('results/figures', 'Figuras y visualizaciones'),
        ('docs', 'Documentación adicional'),
        ('scripts', 'Scripts auxiliares'),
    ]
    
    for dirpath, desc in directories:
        all_checks.append(check_file_exists(dirpath, desc))
    
    # 3. VERIFICAR SECCIONES DEL REPORTE
    print(f"\n{Colors.BOLD}3. SECCIONES DEL REPORTE TÉCNICO{Colors.END}")
    print("-" * 70)
    
    required_sections = [
        'Introducción',
        'Marco Teórico',
        'Metodología',
        'Resultados',
        'Conclusiones',
        'Referencias',
    ]
    
    missing_sections = check_markdown_sections('reporte_tecnico_trabajo3.md', required_sections)
    
    if not missing_sections:
        print(f"{Colors.GREEN}✓{Colors.END} Todas las secciones requeridas presentes")
        all_checks.append(True)
    else:
        print(f"{Colors.RED}✗{Colors.END} Secciones faltantes: {', '.join(missing_sections)}")
        all_checks.append(False)
    
    # 4. VERIFICAR REFERENCIAS
    print(f"\n{Colors.BOLD}4. REFERENCIAS ACADÉMICAS{Colors.END}")
    print("-" * 70)
    
    ref_count = check_references_count('reporte_tecnico_trabajo3.md', min_count=5)
    
    if ref_count >= 5:
        print(f"{Colors.GREEN}✓{Colors.END} Referencias académicas: {ref_count} (mínimo: 5)")
        all_checks.append(True)
    else:
        print(f"{Colors.RED}✗{Colors.END} Referencias académicas: {ref_count} (mínimo: 5)")
        all_checks.append(False)
    
    # 5. VERIFICAR FIGURAS
    print(f"\n{Colors.BOLD}5. VISUALIZACIONES Y FIGURAS{Colors.END}")
    print("-" * 70)
    
    found_figures, missing_figures = check_figures_directory()
    
    if found_figures:
        print(f"{Colors.GREEN}✓{Colors.END} Figuras encontradas ({len(found_figures)}):")
        for fig in found_figures:
            print(f"  • {fig}")
    
    if missing_figures:
        print(f"\n{Colors.YELLOW}⚠{Colors.END} Figuras faltantes ({len(missing_figures)}):")
        for fig in missing_figures:
            print(f"  • {fig}")
        print(f"\n{Colors.YELLOW}💡 Para generar las figuras faltantes, ejecutar:{Colors.END}")
        print(f"   python scripts/generate_figures.py")
        all_checks.append(False)
    else:
        all_checks.append(True)
    
    # 6. VERIFICAR NOTEBOOKS
    print(f"\n{Colors.BOLD}6. NOTEBOOKS DE ANÁLISIS{Colors.END}")
    print("-" * 70)
    
    notebooks = [
        'notebooks/01_preprocessing_exploration.ipynb',
        'notebooks/02_shape_and_texture_descriptors.ipynb',
        'notebooks/03_Pipeline_Clasificacion.ipynb',
    ]
    
    for nb in notebooks:
        all_checks.append(check_file_exists(nb, f"Notebook"))
    
    # 7. VERIFICAR DOCUMENTACIÓN ADICIONAL
    print(f"\n{Colors.BOLD}7. DOCUMENTACIÓN COMPLEMENTARIA{Colors.END}")
    print("-" * 70)
    
    docs = [
        ('docs/pipeline_diagram.md', 'Diagramas de flujo'),
        ('docs/contribucion_individual.md', 'Análisis de contribución'),
        ('docs/README_GITHUB_PAGES.md', 'README para GitHub Pages'),
    ]
    
    for filepath, desc in docs:
        all_checks.append(check_file_exists(filepath, desc))
    
    # 8. VERIFICAR MÓDULOS DE CÓDIGO
    print(f"\n{Colors.BOLD}8. MÓDULOS DE CÓDIGO FUENTE{Colors.END}")
    print("-" * 70)
    
    modules = [
        ('src/__init__.py', 'Inicializador del paquete'),
        ('src/utils.py', 'Funciones auxiliares'),
        ('src/preprocessing.py', 'Pipeline de preprocesamiento'),
    ]
    
    for filepath, desc in modules:
        all_checks.append(check_file_exists(filepath, desc))
    
    # RESUMEN FINAL
    print("\n" + "="*70)
    print(f"{Colors.BOLD}RESUMEN DE VERIFICACIÓN{Colors.END}")
    print("="*70)
    
    total_checks = len(all_checks)
    passed_checks = sum(all_checks)
    
    percentage = (passed_checks / total_checks) * 100
    
    print(f"\nChecks pasados: {passed_checks}/{total_checks} ({percentage:.1f}%)")
    
    if percentage == 100:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ¡REPOSITORIO LISTO PARA PUBLICACIÓN!{Colors.END}")
        print(f"\n{Colors.BLUE}Próximos pasos:{Colors.END}")
        print("  1. Revisar el reporte HTML: reporte_tecnico_trabajo3.html")
        print("  2. Generar figuras faltantes: python scripts/generate_figures.py")
        print("  3. Hacer commit y push a GitHub")
        print("  4. Activar GitHub Pages en la configuración del repositorio")
        print("  5. Compartir el enlace del blog post")
    elif percentage >= 80:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ REPOSITORIO CASI LISTO{Colors.END}")
        print(f"\n{Colors.YELLOW}Completar elementos faltantes antes de publicar.{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ REPOSITORIO INCOMPLETO{Colors.END}")
        print(f"\n{Colors.RED}Se requieren más elementos para cumplir requisitos.{Colors.END}")
    
    print("\n" + "="*70 + "\n")
    
    return percentage == 100

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
