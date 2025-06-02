import nbformat as nbf
import os

# Define paths
project_dir = "C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
notebook_dir = os.path.join(project_dir, "notebooks")
notebook_path = os.path.join(notebook_dir, "02_plankton_eda.ipynb")

# --- Cell 5: Markdown Section 2 ---
markdown_sec2 = """\
## 2. Processamento Completo e Agregação (Todos os Chunks)

Agora, vamos iterar sobre todos os chunks do ficheiro `occurrence.txt` para:
- Selecionar colunas relevantes.
- Converter tipos de dados (datas, coordenadas).
- Realizar limpeza básica.
- Agregar estatísticas gerais (contagem total, intervalo temporal, espécies, etc.).
"""

# --- Cell 6: Code Process All Chunks ---
code_process_chunks = """\
# Colunas de interesse (ajustar conforme necessário)
use_cols = [
    'gbifID', 'eventDate', 'year', 'month', 'day',
    'decimalLatitude', 'decimalLongitude',
    'scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus',
    'basisOfRecord', 'occurrenceStatus', 'individualCount'
]

# Tipos de dados esperados (para otimização e consistência)
dtypes = {
    'gbifID': 'int64',
    'year': 'Int64', # Usar tipo Int64 que suporta NA
    'month': 'Int64',
    'day': 'Int64',
    'decimalLatitude': 'float64',
    'decimalLongitude': 'float64',
    'scientificName': 'str',
    'kingdom': 'str',
    'phylum': 'str',
    'class': 'str',
    'order': 'str',
    'family': 'str',
    'genus': 'str',
    'basisOfRecord': 'str',
    'occurrenceStatus': 'str',
    'individualCount': 'float64' # Pode ser float devido a NA ou contagens não inteiras
}

# Variáveis para agregação
total_records = 0
all_species = set()
min_date, max_date = pd.Timestamp.max, pd.Timestamp.min
min_lat, max_lat = 90.0, -90.0
min_lon, max_lon = 180.0, -180.0
species_counts = pd.Series(dtype='int64')
records_per_year = pd.Series(dtype='int64')

# Lista para guardar amostras de dados para visualização (se necessário)
data_samples = []
sample_fraction = 0.01 # Guardar 1% dos dados para visualização

print(f\"Iniciando processamento completo de {occurrence_file} em chunks de {chunk_size} linhas...\")

try:
    chunk_iter = pd.read_csv(
        occurrence_file,
        sep='\\t',
        chunksize=chunk_size,
        usecols=lambda c: c in use_cols, # Ler apenas colunas de interesse
        dtype={k: dtypes.get(k, 'object') for k in use_cols}, # Aplicar dtypes, object para os não especificados
        low_memory=False,
        on_bad_lines='warn' # Avisar sobre linhas problemáticas
    )

    for i, chunk in enumerate(chunk_iter):
        print(f\"Processando chunk {i+1}...\")
        total_records += len(chunk)

        # Limpeza e Conversão de Tipos
        # Converter coordenadas para numérico, coercing erros para NaN
        chunk['decimalLatitude'] = pd.to_numeric(chunk['decimalLatitude'], errors='coerce')
        chunk['decimalLongitude'] = pd.to_numeric(chunk['decimalLongitude'], errors='coerce')

        # Tentar construir a data a partir de year, month, day
        # Usar Int64 para permitir NA em colunas de inteiros
        chunk[['year', 'month', 'day']] = chunk[['year', 'month', 'day']].astype('Int64')
        # Construir data apenas se year, month, day forem válidos
        valid_date_idx = chunk[['year', 'month', 'day']].notna().all(axis=1)
        # Usar um formato explícito e errors='coerce' para datas inválidas
        chunk.loc[valid_date_idx, 'parsedDate'] = pd.to_datetime(
            chunk.loc[valid_date_idx, ['year', 'month', 'day']],
            format='%Y%m%d', errors='coerce'
        )
        # Tentar preencher com eventDate se a construção falhar ou não for possível
        chunk['parsedDate'] = chunk['parsedDate'].fillna(pd.to_datetime(chunk['eventDate'], errors='coerce'))

        # Remover linhas sem data ou coordenadas válidas
        chunk.dropna(subset=['parsedDate', 'decimalLatitude', 'decimalLongitude'], inplace=True)

        if not chunk.empty:
            # Atualizar agregados
            min_date = min(min_date, chunk['parsedDate'].min())
            max_date = max(max_date, chunk['parsedDate'].max())
            min_lat = min(min_lat, chunk['decimalLatitude'].min())
            max_lat = max(max_lat, chunk['decimalLatitude'].max())
            min_lon = min(min_lon, chunk['decimalLongitude'].min())
            max_lon = max(max_lon, chunk['decimalLongitude'].max())

            all_species.update(chunk['scientificName'].dropna().unique())
            species_counts = species_counts.add(chunk['scientificName'].value_counts(), fill_value=0)
            records_per_year = records_per_year.add(chunk['parsedDate'].dt.year.value_counts(), fill_value=0)

            # Guardar uma amostra
            data_samples.append(chunk.sample(frac=sample_fraction))

    print(\"Processamento de chunks concluído.\")

    # Concatenar amostras
    if data_samples:
        sample_df = pd.concat(data_samples, ignore_index=True)
        print(f\"DataFrame de amostra criado com {len(sample_df)} registos.\")
    else:
        sample_df = pd.DataFrame(columns=use_cols + ['parsedDate']) # Criar df vazio se não houver amostras
        print(\"Nenhuma amostra de dados foi criada.\")

except Exception as e:
    print(f\"Erro durante o processamento dos chunks: {e}\")
    # Pode ser útil imprimir o estado das variáveis de agregação aqui

"""

# --- Cell 7: Code Display Aggregated Stats ---
code_display_stats = """\
print(\"\\n--- Estatísticas Agregadas ---\")
print(f\"Total de registos processados (após limpeza inicial): {total_records}\")
if min_date != pd.Timestamp.max:
    print(f\"Intervalo Temporal: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}\")
else:
    print(\"Intervalo Temporal: Não foi possível determinar.\")
print(f\"Cobertura Geográfica (Lat): {min_lat:.4f} a {max_lat:.4f}\")
print(f\"Cobertura Geográfica (Lon): {min_lon:.4f} a {max_lon:.4f}\")
print(f\"Número de espécies únicas identificadas: {len(all_species)}\")

print(\"\\nTop 10 Espécies Mais Frequentes:\")
print(species_counts.astype(int).nlargest(10))

print(\"\\nNúmero de Registos por Ano (Top 10 anos):\")
print(records_per_year.astype(int).nlargest(10))
"""

try:
    # Read the existing notebook
    with open(notebook_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    # Add cells
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_sec2))
    nb['cells'].append(nbf.v4.new_code_cell(code_process_chunks))
    nb['cells'].append(nbf.v4.new_code_cell(code_display_stats))

    # Write the updated notebook back to the file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Células de processamento completo adicionadas com sucesso a: {notebook_path}")

except Exception as e:
    print(f"Erro ao adicionar células ao notebook: {e}")

