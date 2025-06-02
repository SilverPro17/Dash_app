import nbformat as nbf
import os

# Define paths
project_dir = "C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
notebook_dir = os.path.join(project_dir, "notebooks")
notebook_path = os.path.join(notebook_dir, "02_plankton_eda.ipynb")

# --- Cell 1: Markdown Introduction ---
markdown_intro = """\
# Análise Exploratória de Dados (EDA) - Dataset Principal de Plâncton (CPR)

Este notebook realiza a análise exploratória inicial do dataset Continuous Plankton Recorder (CPR) descarregado do GBIF.

**Formato:** Darwin Core Archive (DwC-A)
**Ficheiro Principal:** `occurrence.txt` (~3.2GB)

**Nota:** Devido ao tamanho do ficheiro `occurrence.txt`, a leitura e análise serão feitas usando `chunking` com Pandas para gerir o uso de memória.
"""

# --- Cell 2: Code Imports and Setup ---
code_imports = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Definir diretórios
project_dir = \"C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
data_dir = os.path.join(project_dir, \"data\")
plankton_data_dir = os.path.join(data_dir, \"plankton_dwca\")
report_dir = os.path.join(project_dir, \"report\")

# Criar diretório de report se não existir
os.makedirs(report_dir, exist_ok=True)

# Caminho para o ficheiro principal
occurrence_file = os.path.join(plankton_data_dir, \"occurrence.txt\")

print(f\"Ficheiro de ocorrências: {occurrence_file}\")
print(\"Bibliotecas importadas com sucesso!\")
"""

try:
    # Read the existing notebook
    with open(notebook_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    # Add cells
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_intro))
    nb['cells'].append(nbf.v4.new_code_cell(code_imports))

    # Write the updated notebook back to the file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Células iniciais adicionadas com sucesso a: {notebook_path}")

except Exception as e:
    print(f"Erro ao adicionar células ao notebook: {e}")

