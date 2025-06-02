import nbformat as nbf
import os

# Define the notebook path
project_dir = "C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
notebook_dir = os.path.join(project_dir, "notebooks")
notebook_path = os.path.join(notebook_dir, "02_plankton_eda.ipynb")

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Write the notebook to a file
try:
    # Ensure the directory exists
    os.makedirs(notebook_dir, exist_ok=True)
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook criado com sucesso em: {notebook_path}")
except Exception as e:
    print(f"Erro ao criar o notebook: {e}")

