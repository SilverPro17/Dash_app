import nbformat as nbf
import os

# Define paths
project_dir = "C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
notebook_dir = os.path.join(project_dir, "notebooks")
notebook_path = os.path.join(notebook_dir, "02_plankton_eda.ipynb")
report_dir = os.path.join(project_dir, "report")

# --- Cell 8: Markdown Section 3 ---
markdown_sec3 = """\
## 3. Visualizações Preliminares

Vamos criar algumas visualizações com base nos dados agregados e na amostra recolhida.
"""

# --- Cell 9: Code Plot Bar Charts ---
code_plot_bars = """\
if 'species_counts' in locals() and not species_counts.empty:
    # Plot Top 10 Espécies
    plt.figure(figsize=(10, 6))
    top_species = species_counts.astype(int).nlargest(10)
    sns.barplot(x=top_species.values, y=top_species.index, palette='viridis')
    plt.title('Top 10 Espécies Mais Frequentes')
    plt.xlabel('Número de Ocorrências')
    plt.ylabel('Nome Científico')
    plt.tight_layout()
    save_path_species = os.path.join(report_dir, 'top_10_species.png')
    plt.savefig(save_path_species)
    print(f\"Gráfico Top 10 Espécies salvo em: {save_path_species}\")
    plt.close()
else:
    print(\"Dados de contagem de espécies não disponíveis para plotagem.\")

if 'records_per_year' in locals() and not records_per_year.empty:
    # Plot Registos por Ano
    plt.figure(figsize=(12, 6))
    # Ordenar por ano para o gráfico de linha
    records_per_year_sorted = records_per_year.astype(int).sort_index()
    records_per_year_sorted.plot(kind='line', marker='o') # Usar gráfico de linha
    plt.title('Número Total de Registos por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Número de Ocorrências')
    plt.grid(True)
    plt.tight_layout()
    save_path_years = os.path.join(report_dir, 'records_per_year.png')
    plt.savefig(save_path_years)
    print(f\"Gráfico Registos por Ano salvo em: {save_path_years}\")
    plt.close()
else:
    print(\"Dados de registos por ano não disponíveis para plotagem.\")
"""

# --- Cell 10: Code Plot Interactive Map ---
code_plot_map = """\
if 'sample_df' in locals() and not sample_df.empty:
    print(f\"Gerando mapa interativo com {len(sample_df)} pontos da amostra...\")
    try:
        fig = px.scatter_geo(
            sample_df,
            lat='decimalLatitude',
            lon='decimalLongitude',
            color='scientificName', # Colorir por espécie (pode ficar lento com muitas espécies)
            hover_name='scientificName',
            hover_data=['parsedDate'],
            projection='natural earth',
            title='Distribuição Geográfica das Ocorrências (Amostra)',
            # Limitar o número de categorias de cores se houver muitas espécies
            # color_discrete_sequence=px.colors.qualitative.Plotly[:10] # Exemplo
        )
        # Salvar como HTML interativo
        map_path = os.path.join(report_dir, 'plankton_distribution_map.html')
        fig.write_html(map_path)
        print(f\"Mapa interativo salvo em: {map_path}\")
    except Exception as e:
        print(f\"Erro ao gerar o mapa interativo: {e}\")
else:
    print(\"DataFrame de amostra não disponível para gerar mapa.\")
"""

# --- Cell 11: Markdown Next Steps ---
markdown_next_steps = """\
## 4. Próximos Passos

- Análise mais aprofundada das distribuições geográficas e temporais.
- Limpeza de dados mais rigorosa (ex: validação de coordenadas, tratamento de outliers em `individualCount`).
- Análise do ficheiro `verbatim.txt` se contiver informações adicionais úteis.
- Download e EDA do dataset de Clorofila-a.
- Integração dos dados de plâncton com os dados ambientais (SST, Clorofila-a).
- Desenvolvimento do dashboard interativo.
"""

try:
    # Read the existing notebook
    with open(notebook_path, 'r') as f:
        nb = nbf.read(f, as_version=4)

    # Add cells
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_sec3))
    nb['cells'].append(nbf.v4.new_code_cell(code_plot_bars))
    nb['cells'].append(nbf.v4.new_code_cell(code_plot_map))
    nb['cells'].append(nbf.v4.new_markdown_cell(markdown_next_steps))

    # Write the updated notebook back to the file
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Células de visualização adicionadas com sucesso a: {notebook_path}")

except Exception as e:
    print(f"Erro ao adicionar células de visualização ao notebook: {e}")

