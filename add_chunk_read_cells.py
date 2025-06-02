import nbformat as nbf
import os

# Define paths
project_dir = "C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
notebook_dir = os.path.join(project_dir, "notebooks")
notebook_path = os.path.join(notebook_dir, "02_plankton_eda.ipynb")

# --- Cell 3: Markdown Section 1 ---
markdown_sec1 = """\
## 1. Leitura Inicial e Inspeção (Primeiro Chunk)

Vamos ler apenas o primeiro chunk do ficheiro `occurrence.txt` para inspecionar as colunas, tipos de dados e ter uma ideia inicial do conteúdo sem carregar o ficheiro inteiro.
"""

# --- Cell 4: Code Read First Chunk ---
code_read_chunk = """\
# Definir o tamanho do chunk (número de linhas)
chunk_size = 100000 # Ajustar conforme necessário
first_chunk = None # Inicializar a variável

try:
    print(f\"A ler o primeiro chunk ({chunk_size} linhas) de {occurrence_file}...\")
    # Ler o primeiro chunk, assumindo separador TAB (comum em DwC-A)
    # Adicionar error_bad_lines=False ou on_bad_lines=\"skip\" se houver linhas mal formatadas
    # Adicionar low_memory=False se houver problemas com tipos mistos
    chunk_iter = pd.read_csv(occurrence_file, sep=\"\\t\", chunksize=chunk_size, low_memory=False, on_bad_lines=\"warn\")
    first_chunk = next(chunk_iter)
    print(\"Primeiro chunk lido com sucesso.\")

    # Inspeção inicial
    print(\"\\nInformações do Primeiro Chunk:\")
    first_chunk.info()

    print(\"\\nPrimeiras 5 linhas:\")
    # Usar display para melhor formatação no Jupyter
    from IPython.display import display
    display(first_chunk.head())

    print(\"\\nColunas disponíveis:\")
    print(list(first_chunk.columns))

except FileNotFoundError:
    print(f\"Erro: Ficheiro não encontrado em {occurrence_file}\")
except StopIteration:
    print(\"Erro: O ficheiro parece estar vazio ou o chunksize é maior que o ficheiro.\")
except Exception as e:
    print(f\"Erro ao ler o primeiro chunk: {e}\")
"""

try:
    # Read the existing notebook
    with open(notebook_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    # Add cells
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_sec1))
    nb['cells'].append(nbf.v4.new_code_cell(code_read_chunk))

    # Write the updated notebook back to the file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Células de leitura do chunk adicionadas com sucesso a: {notebook_path}")

except Exception as e:
    print(f"Erro ao adicionar células ao notebook: {e}")

