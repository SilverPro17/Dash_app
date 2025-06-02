# Projeto: Exploração Interativa da Distribuição de Plâncton

## Descrição
Este projeto desenvolve uma aplicação web interativa para exploração e análise de dados de distribuição de plâncton, complementados por dados ambientais como a Temperatura da Superfície do Mar (SST). A aplicação permite visualizar padrões de distribuição espacial e temporal do plâncton, analisar tendências e explorar possíveis correlações com variáveis ambientais.

## Estrutura do Projeto
```
plankton_project/
├── app/                  # Aplicação web Dash
│   └── app.py            # Código principal da aplicação
├── data/                 # Dados processados e amostras
│   ├── plankton_dwca/    # Dados do Continuous Plankton Recorder
│   ├── plankton_sample.csv # Amostra processada para visualização
│   └── sst.mon.mean.nc   # Dados de Temperatura da Superfície do Mar
├── notebooks/            # Jupyter Notebooks para análise
│   ├── 01_sst_eda_optimized_executed.ipynb  # Análise da SST
│   └── 02_plankton_eda_executed.ipynb       # Análise do plâncton
├── report/               # Relatórios e visualizações
│   ├── plankton_distribution_map.html       # Mapa interativo
│   ├── records_per_year.png                 # Gráfico de registos por ano
│   ├── relatorio_completo.md                # Relatório detalhado do projeto
│   ├── sst_mean_subset_map.png              # Mapa da SST média
│   ├── sst_timeseries_point.png             # Série temporal da SST
│   └── top_10_species.png                   # Gráfico das espécies mais frequentes
└── src/                  # Scripts e código fonte
    ├── add_cells_to_notebook.py             # Utilitário para notebooks
    ├── add_chunk_read_cells.py              # Utilitário para leitura em chunks
    ├── add_full_eda_cells.py                # Utilitário para análise exploratória
    ├── add_viz_cells.py                     # Utilitário para visualizações
    ├── create_ml_notebook_fixed.py          # Utilitário para notebook de ML
    ├── create_plankton_notebook.py          # Utilitário para notebook de plâncton
    └── funcionalidades_a_adicionar.md       # Documentação de funcionalidades
```

## Datasets
1. **Continuous Plankton Recorder (CPR)**
   - Fonte: Global Biodiversity Information Facility (GBIF)
   - Formato: Darwin Core Archive (DwC-A)
   - Conteúdo: Registos de ocorrência de espécies de plâncton com coordenadas geográficas, datas e contagens

2. **Temperatura da Superfície do Mar (SST)**
   - Fonte: NOAA Optimum Interpolation Sea Surface Temperature V2 High Resolution
   - Formato: NetCDF4
   - Resolução: 0.25° (alta resolução)
   - Cobertura temporal: 1981-presente

## Funcionalidades da Aplicação Web
- Visualização da distribuição espacial do plâncton em mapa interativo
- Gráficos de séries temporais para análise de tendências
- Filtros interativos por espécie e período temporal
- Análise de correlação entre abundância de plâncton e temperatura
- Relatório completo integrado na aplicação

## Requisitos
- Python 3.8+
- Bibliotecas: pandas, numpy, matplotlib, seaborn, plotly, dash, xarray, netCDF4
- Consulte `requirements.txt` para a lista completa de dependências

## Execução
1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute a aplicação: `python app/app.py`
4. Acesse a aplicação no navegador: `http://localhost:8050`

## Autores
Desenvolvido como projeto para exploração interativa de dados oceanográficos.

## Licença
Este projeto é disponibilizado para fins educacionais e de pesquisa.
