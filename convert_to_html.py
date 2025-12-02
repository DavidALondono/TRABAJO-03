#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para convertir el reporte técnico de Markdown a HTML
con estilos académicos profesionales.
"""

import re
from pathlib import Path

def markdown_to_html(md_file, html_file):
    """Convierte Markdown a HTML con estilos profesionales."""
    
    # Leer el archivo Markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Reemplazar imágenes con rutas relativas correctas
    content = content.replace('../results/figures/', 'results/figures/')
    
    # Convertir elementos Markdown básicos
    # Encabezados
    content = re.sub(r'^##### (.+)$', r'<h5>\1</h5>', content, flags=re.MULTILINE)
    content = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
    
    # Texto en negrita
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    
    # Texto en cursiva (solo asteriscos simples, evitando dobles)
    content = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', content)
    
    # Enlaces
    content = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', content)
    
    # Imágenes con descripción
    content = re.sub(r'!\[(.+?)\]\((.+?)\)\n\*(.+?)\*', 
                     r'<figure><img src="\2" alt="\1" style="max-width:100%; height:auto;"><figcaption><em>\3</em></figcaption></figure>', 
                     content)
    
    # Listas con viñetas
    lines = content.split('\n')
    in_list = False
    result = []
    
    for line in lines:
        if line.strip().startswith('- '):
            if not in_list:
                result.append('<ul>')
                in_list = True
            item = line.strip()[2:]
            result.append(f'<li>{item}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(line)
    
    if in_list:
        result.append('</ul>')
    
    content = '\n'.join(result)
    
    # Tablas Markdown
    def convert_table(match):
        table_text = match.group(0)
        lines = [l.strip() for l in table_text.split('\n') if l.strip()]
        
        html = '<table>\n<thead>\n<tr>\n'
        # Encabezados
        headers = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        for header in headers:
            html += f'<th>{header}</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Filas de datos (saltar línea de separación)
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            html += '<tr>\n'
            for cell in cells:
                html += f'<td>{cell}</td>\n'
            html += '</tr>\n'
        
        html += '</tbody>\n</table>'
        return html
    
    # Buscar tablas (líneas que comienzan con |)
    content = re.sub(r'(\|.+\|\n)+', convert_table, content, flags=re.MULTILINE)
    
    # Párrafos
    paragraphs = content.split('\n\n')
    formatted_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if para and not any(para.startswith(tag) for tag in ['<h', '<ul', '<ol', '<table', '<figure', '<hr']):
            # Si no es un elemento HTML, envolverlo en <p>
            if not para.startswith('<'):
                para = f'<p>{para}</p>'
        formatted_paragraphs.append(para)
    
    content = '\n\n'.join(formatted_paragraphs)
    
    # Líneas horizontales
    content = re.sub(r'^---$', '<hr>', content, flags=re.MULTILINE)
    
    # Plantilla HTML completa
    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clasificación de Imágenes Médicas - Reporte Técnico</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: white;
            padding: 40px 60px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #1a1a1a;
            font-size: 2.2em;
            margin-bottom: 20px;
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 15px;
        }}
        
        h2 {{
            color: #2c3e50;
            font-size: 1.8em;
            margin-top: 40px;
            margin-bottom: 15px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        
        h3 {{
            color: #34495e;
            font-size: 1.5em;
            margin-top: 30px;
            margin-bottom: 12px;
        }}
        
        h4 {{
            color: #555;
            font-size: 1.3em;
            margin-top: 25px;
            margin-bottom: 10px;
        }}
        
        h5 {{
            color: #666;
            font-size: 1.1em;
            margin-top: 20px;
            margin-bottom: 8px;
            font-weight: bold;
        }}
        
        p {{
            text-align: justify;
            margin-bottom: 15px;
            color: #444;
        }}
        
        ul, ol {{
            margin-left: 30px;
            margin-bottom: 15px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        strong {{
            color: #2c3e50;
            font-weight: bold;
        }}
        
        em {{
            font-style: italic;
            color: #555;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        thead tr {{
            background-color: #2c3e50;
            color: white;
            text-align: left;
        }}
        
        th, td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
        }}
        
        tbody tr {{
            border-bottom: 1px solid #ddd;
        }}
        
        tbody tr:nth-of-type(even) {{
            background-color: #f8f9fa;
        }}
        
        tbody tr:hover {{
            background-color: #e8f4f8;
        }}
        
        figure {{
            margin: 30px 0;
            text-align: center;
        }}
        
        figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        figcaption {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
            font-style: italic;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .author-info {{
            text-align: center;
            margin: 20px 0;
            color: #666;
        }}
        
        .header-section {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 4px solid #3498db;
            margin: 30px 0;
        }}
        
        .toc h2 {{
            margin-top: 0;
            font-size: 1.5em;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        @media print {{
            body {{
                background-color: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
{content}
    </div>
</body>
</html>"""
    
    # Escribir el archivo HTML
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✓ Archivo HTML generado exitosamente: {html_file}")

if __name__ == "__main__":
    md_file = Path("reporte_tecnico_trabajo3.md")
    html_file = Path("reporte_tecnico_trabajo3.html")
    
    if md_file.exists():
        markdown_to_html(md_file, html_file)
    else:
        print(f"Error: No se encontró el archivo {md_file}")
