import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
import os
import base64
import numpy as np
import markdown

# --- Inicialização da App Dash ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Diretórios ---
project_dir = "C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
report_dir = os.path.join(project_dir, "report")
data_dir = os.path.join(project_dir, "data")

# --- Carregar Imagens Estáticas ---
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Erro: Imagem não encontrada em {image_path}")
        return None

top_species_img_path = os.path.join(report_dir, 'top_10_species.png')
records_per_year_img_path = os.path.join(report_dir, 'records_per_year.png')
sst_map_img_path = os.path.join(report_dir, 'sst_mean_subset_map.png')
sst_timeseries_img_path = os.path.join(report_dir, 'sst_timeseries_point.png')

encoded_top_species = encode_image(top_species_img_path)
encoded_records_year = encode_image(records_per_year_img_path)
encoded_sst_map = encode_image(sst_map_img_path)
encoded_sst_timeseries = encode_image(sst_timeseries_img_path)

# --- Carregar Relatório Completo ---
relatorio_path = os.path.join(report_dir, 'relatorio_completo.md')
try:
    with open(relatorio_path, 'r', encoding='utf-8') as file:
        relatorio_md = file.read()
    relatorio_html = markdown.markdown(relatorio_md)
except FileNotFoundError:
    print(f"Erro: Relatório não encontrado em {relatorio_path}")
    relatorio_html = "<p>Relatório não encontrado.</p>"
except Exception as e:
    print(f"Erro ao carregar ou processar o relatório: {e}")
    relatorio_html = f"<p>Erro ao processar o relatório: {str(e)}</p>"

# --- Carregar Mapa HTML ---
plankton_map_path = os.path.join(report_dir, 'plankton_distribution_map.html')
try:
    with open(plankton_map_path, 'r', encoding='utf-8') as f:
        plankton_map_html = f.read()
except FileNotFoundError:
    print(f"Erro: Mapa não encontrado em {plankton_map_path}")
    plankton_map_html = "<p>Mapa não encontrado.</p>"
except Exception as e:
    print(f"Erro ao carregar o mapa: {e}")
    plankton_map_html = f"<p>Erro ao carregar o mapa: {str(e)}</p>"

# --- Carregar Dados Reais do Plâncton (Amostra) ---
plankton_sample_path = os.path.join(data_dir, 'plankton_sample.csv')
try:
    plankton_df = pd.read_csv(plankton_sample_path)
    # Converter colunas relevantes para tipos adequados, se necessário
    plankton_df['eventDate'] = pd.to_datetime(plankton_df['eventDate'], errors='coerce')
    plankton_df['year'] = pd.to_numeric(plankton_df['year'], errors='coerce')
    plankton_df['individualCount'] = pd.to_numeric(plankton_df['individualCount'], errors='coerce')
    plankton_df['temperature'] = pd.to_numeric(plankton_df['temperature'], errors='coerce')
    plankton_df.dropna(subset=['year', 'scientificName', 'individualCount', 'temperature'], inplace=True)
    plankton_df['year'] = plankton_df['year'].astype(int)

    # Obter listas para filtros
    available_species = sorted(plankton_df['scientificName'].unique())
    min_year = int(plankton_df['year'].min())
    max_year = int(plankton_df['year'].max())
    years = list(range(min_year, max_year + 1))

except FileNotFoundError:
    print(f"Erro: Ficheiro de amostra de plâncton não encontrado em {plankton_sample_path}")
    # Criar DataFrame vazio ou com dados de exemplo para evitar erros no layout
    plankton_df = pd.DataFrame({
        'year': [2000], 'scientificName': ['N/A'], 'individualCount': [0], 'temperature': [0]
    })
    available_species = ['N/A']
    min_year, max_year = 2000, 2000
    years = [2000]
except Exception as e:
    print(f"Erro ao carregar ou processar dados de plâncton: {e}")
    plankton_df = pd.DataFrame({
        'year': [2000], 'scientificName': ['N/A'], 'individualCount': [0], 'temperature': [0]
    })
    available_species = ['N/A']
    min_year, max_year = 2000, 2000
    years = [2000]

# --- Layout da Aplicação ---
app.layout = html.Div(children=[
    # Cabeçalho
    html.Div([
        html.H1('Exploração Interativa da Distribuição de Plâncton', 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.H3('Análise de dados do Continuous Plankton Recorder (CPR) e variáveis ambientais',
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 'marginBottom': '20px'}),
    
    # Tabs para organizar o conteúdo
    dcc.Tabs([
        # Tab 1: Visão Geral do Plâncton
        dcc.Tab(label='Visão Geral do Plâncton', children=[
            html.Div([
                html.H2('Análise Exploratória do Dataset de Plâncton', 
                        style={'color': '#2980b9', 'marginTop': '20px'}),
                
                # Estatísticas Gerais
                html.Div([
                    html.H3('Estatísticas Gerais', style={'color': '#3498db'}),
                    html.Ul([
                        html.Li('Dataset: Continuous Plankton Recorder (CPR)'),
                        html.Li('Formato: Darwin Core Archive (DwC-A)'),
                        html.Li('Tamanho: ~3.2GB (ficheiro occurrence.txt)'),
                        html.Li('Processamento: Realizado em chunks para otimização de memória'),
                        html.Li(f'Amostra para interatividade: {len(plankton_df)} registos')
                    ])
                ], style={'marginBottom': '20px'}),
                
                # Visualizações lado a lado
                html.Div([
                    # Gráfico Top 10 Espécies
                    html.Div([
                        html.H3('Top 10 Espécies Mais Frequentes', style={'textAlign': 'center'}),
                        html.Img(src=encoded_top_species if encoded_top_species else '', style={'width': '100%'}) if encoded_top_species else html.P("Imagem não encontrada")
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 
                              'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'}),
                    
                    # Gráfico Registos por Ano
                    html.Div([
                        html.H3('Número de Registos por Ano', style={'textAlign': 'center'}),
                        html.Img(src=encoded_records_year if encoded_records_year else '', style={'width': '100%'}) if encoded_records_year else html.P("Imagem não encontrada")
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 
                              'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'marginLeft': '4%'})
                ], style={'marginBottom': '20px'}),
                
                # Mapa Interativo
                html.Div([
                    html.H3('Distribuição Geográfica das Ocorrências (Amostra)', style={'textAlign': 'center'}),
                    html.Iframe(
                        srcDoc=plankton_map_html,
                        width='100%',
                        height='600px',
                        style={'border': 'none', 'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'}
                    )
                ], style={'marginBottom': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'})
            ], style={'padding': '20px'})
        ]),
        
        # Tab 2: Dados Ambientais (SST)
        dcc.Tab(label='Dados Ambientais (SST)', children=[
            html.Div([
                html.H2('Temperatura da Superfície do Mar (SST)', 
                        style={'color': '#2980b9', 'marginTop': '20px'}),
                
                # Informações sobre o dataset SST
                html.Div([
                    html.H3('Sobre o Dataset', style={'color': '#3498db'}),
                    html.Ul([
                        html.Li('Fonte: NOAA OI SST V2 High Resolution'),
                        html.Li('Formato: NetCDF4'),
                        html.Li('Resolução: 0.25° (alta resolução)'),
                        html.Li('Cobertura temporal: 1981-presente')
                    ])
                ], style={'marginBottom': '20px'}),
                
                # Visualizações lado a lado
                html.Div([
                    # Mapa SST
                    html.Div([
                        html.H3('Mapa da SST Média (Subconjunto)', style={'textAlign': 'center'}),
                        html.Img(src=encoded_sst_map if encoded_sst_map else '', style={'width': '100%'}) if encoded_sst_map else html.P("Imagem não encontrada")
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 
                              'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'}),
                    
                    # Série Temporal SST
                    html.Div([
                        html.H3('Série Temporal da SST', style={'textAlign': 'center'}),
                        html.Img(src=encoded_sst_timeseries if encoded_sst_timeseries else '', style={'width': '100%'}) if encoded_sst_timeseries else html.P("Imagem não encontrada")
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 
                              'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'marginLeft': '4%'})
                ], style={'marginBottom': '20px'})
            ], style={'padding': '20px'})
        ]),
        
        # Tab 3: Análise Interativa (Dados Reais)
        dcc.Tab(label='Análise Interativa', children=[
            html.Div([
                html.H2('Análise Interativa com Dados Reais', 
                        style={'color': '#2980b9', 'marginTop': '20px'}),
                
                # Controles interativos
                html.Div([
                    html.H3('Selecione as Espécies', style={'color': '#3498db'}),
                    dcc.Dropdown(
                        id='species-dropdown',
                        options=[{'label': sp, 'value': sp} for sp in available_species],
                        value=available_species[:min(3, len(available_species))],  # Valores padrão (até 3 espécies)
                        multi=True
                    ),
                    
                    html.H3('Intervalo de Anos', style={'color': '#3498db', 'marginTop': '20px'}),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=min_year,
                        max=max_year,
                        step=1,
                        marks={year: str(year) if year % 10 == 0 or year == min_year or year == max_year else '' for year in years},
                        value=[max(min_year, max_year - 20), max_year] # Últimos 20 anos por defeito
                    )
                ], style={'marginBottom': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 
                          'borderRadius': '5px', 'border': '1px solid #ddd'}),
                
                # Gráficos interativos
                html.Div([
                    html.H3('Contagem de Indivíduos por Ano e Espécie', style={'textAlign': 'center'}),
                    dcc.Graph(id='abundance-graph')
                ], style={'marginBottom': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'}),
                
                html.Div([
                    html.H3('Correlação entre Temperatura e Contagem de Indivíduos', style={'textAlign': 'center'}),
                    dcc.Graph(id='correlation-graph')
                ], style={'marginBottom': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'})
            ], style={'padding': '20px'})
        ]),
        
        # Tab 4: Relatório Completo (NOVA)
        dcc.Tab(label='Relatório Completo', children=[
            html.Div([
                html.H2('Relatório Completo do Projeto', 
                        style={'color': '#2980b9', 'marginTop': '20px', 'marginBottom': '20px'}),
                
                # Índice de Conteúdos
                html.Div([
                    html.H3('Índice', style={'color': '#3498db'}),
                    html.Ul([
                        html.Li(html.A('1. Introdução', href='#introducao')),
                        html.Li(html.A('2. Revisão de Literatura', href='#revisao')),
                        html.Li(html.A('3. Descrição dos Datasets', href='#datasets')),
                        html.Li(html.A('4. Metodologia', href='#metodologia')),
                        html.Li(html.A('5. Resultados e Discussão', href='#resultados')),
                        html.Li(html.A('6. Conclusões e Trabalho Futuro', href='#conclusoes')),
                        html.Li(html.A('7. Referências', href='#referencias'))
                    ])
                ], style={'marginBottom': '30px', 'backgroundColor': '#f8f9fa', 'padding': '15px', 
                          'borderRadius': '5px', 'border': '1px solid #ddd'}),
                
                # Conteúdo do Relatório
                html.Div([
                    # Renderizar o HTML do relatório usando Iframe
                    html.Iframe(
                        srcDoc=relatorio_html,
                        style={'width': '100%', 'height': '800px', 'border': 'none'}
                    )
                ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': 'white', 
                          'borderRadius': '5px', 'border': '1px solid #ddd'})
            ], style={'padding': '20px'})
        ]),
        
        # Tab 5: Sobre o Projeto
        dcc.Tab(label='Sobre o Projeto', children=[
            html.Div([
                html.H2('Sobre o Projeto', 
                        style={'color': '#2980b9', 'marginTop': '20px'}),
                
                html.Div([
                    html.H3('Objetivos', style={'color': '#3498db'}),
                    html.P('Este projeto visa desenvolver uma aplicação/análise abrangente que permita a exploração visual interativa de dados de distribuição de plâncton e aplique técnicas de machine learning para identificar padrões e potencialmente prever a abundância de certas espécies ou grupos.'),
                    
                    html.H3('Metodologia', style={'color': '#3498db'}),
                    html.P('A metodologia inclui a aquisição e processamento de dados de plâncton e ambientais, análise exploratória, visualização interativa, modelação estatística e machine learning, e desenvolvimento de uma aplicação web para exploração dos resultados.'),
                    
                    html.H3('Datasets', style={'color': '#3498db'}),
                    html.Ul([
                        html.Li('Continuous Plankton Recorder (CPR) - GBIF'),
                        html.Li('NOAA OI SST V2 High Resolution (Temperatura da Superfície do Mar)'),
                        html.Li('NASA OceanColor Aqua-MODIS (Clorofila-a) - Pendente')
                    ]),
                    
                    html.H3('Ferramentas', style={'color': '#3498db'}),
                    html.Ul([
                        html.Li('Python (Pandas, NumPy, Matplotlib, Seaborn, Plotly)'),
                        html.Li('Dash (Framework para aplicações web interativas)'),
                        html.Li('Scikit-learn (Machine Learning)'),
                        html.Li('Xarray (Dados multidimensionais)')
                    ])
                ], style={'marginBottom': '20px'})
            ], style={'padding': '20px'})
        ])
    ], style={'marginBottom': '20px'})
], style={'maxWidth': '1200px', 'margin': '0 auto', 'fontFamily': 'Arial, sans-serif'})

# --- Callbacks para Interatividade ---
@app.callback(
    Output('abundance-graph', 'figure'),
    [Input('species-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_abundance_graph(selected_species, year_range):
    if not selected_species or plankton_df.empty:
        return px.line(title='Selecione espécies e intervalo de anos para visualizar a contagem.')
    
    # Filtrar DataFrame com base nas seleções
    filtered_df = plankton_df[
        (plankton_df['scientificName'].isin(selected_species)) & 
        (plankton_df['year'] >= year_range[0]) & 
        (plankton_df['year'] <= year_range[1])
    ]
    
    if filtered_df.empty:
        return px.line(title=f'Nenhum dado encontrado para as espécies selecionadas no período {year_range[0]}-{year_range[1]}.')

    # Agrupar por ano e espécie, somando a contagem de indivíduos
    grouped_df = filtered_df.groupby(['year', 'scientificName'])['individualCount'].sum().reset_index()
    
    fig = px.line(
        grouped_df, 
        x='year', 
        y='individualCount', 
        color='scientificName',
        title=f'Contagem Total de Indivíduos por Ano ({year_range[0]}-{year_range[1]})',
        labels={'individualCount': 'Contagem Total de Indivíduos', 'year': 'Ano', 'scientificName': 'Espécie'},
        template='plotly_white'
    )
    
    fig.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

@app.callback(
    Output('correlation-graph', 'figure'),
    [Input('species-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_correlation_graph(selected_species, year_range):
    if not selected_species or plankton_df.empty:
        return px.scatter(title='Selecione espécies e intervalo de anos para visualizar a correlação.')

    # Filtrar DataFrame com base nas seleções
    filtered_df = plankton_df[
        (plankton_df['scientificName'].isin(selected_species)) & 
        (plankton_df['year'] >= year_range[0]) & 
        (plankton_df['year'] <= year_range[1])
    ].copy() # Usar .copy() para evitar SettingWithCopyWarning

    if filtered_df.empty or filtered_df['temperature'].isnull().all():
        return px.scatter(title=f'Nenhum dado de temperatura ou ocorrência encontrado para as espécies selecionadas no período {year_range[0]}-{year_range[1]}.')

    # Remover linhas onde a temperatura é nula para o gráfico de dispersão
    filtered_df.dropna(subset=['temperature', 'individualCount'], inplace=True)

    if filtered_df.empty:
        return px.scatter(title=f'Nenhum dado válido de temperatura e contagem encontrado para as espécies selecionadas no período {year_range[0]}-{year_range[1]}.')

    fig = px.scatter(
        filtered_df, 
        x='temperature', 
        y='individualCount', 
        color='scientificName',
        # trendline='ols', # Linha de tendência pode ser computacionalmente intensiva e visualmente confusa com muitos pontos
        title=f'Contagem de Indivíduos vs. Temperatura ({year_range[0]}-{year_range[1]})',
        labels={'temperature': 'Temperatura (°C)', 'individualCount': 'Contagem de Indivíduos', 'scientificName': 'Espécie'},
        template='plotly_white'
    )
    
    fig.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

# --- Execução da App ---
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)