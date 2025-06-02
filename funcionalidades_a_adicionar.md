# Funcionalidades a Adicionar ao Dashboard de Plâncton

Com base na análise do estado atual do projeto e do dashboard existente, identifiquei as seguintes funcionalidades a serem implementadas:

## 1. Integração de Dados Reais do Plâncton
- Substituir os dados simulados por dados reais do dataset CPR
- Processar os dados em chunks para gestão eficiente da memória
- Criar um arquivo CSV pré-processado com amostra representativa para uso no dashboard

## 2. Filtros Interativos Avançados
- Filtros por região geográfica (retângulo ou polígono selecionável no mapa)
- Filtros por período temporal mais granulares (mês/estação do ano)
- Filtros por taxonomia (família, género, espécie)
- Filtros por profundidade de amostragem (se disponível nos dados)

## 3. Visualizações Interativas Adicionais
- Gráficos de séries temporais com médias móveis e tendências
- Mapas de calor de abundância por região
- Gráficos de distribuição sazonal
- Pirâmides taxonómicas interativas

## 4. Análise Estatística
- Testes de correlação entre abundância de plâncton e variáveis ambientais
- Análise de variância (ANOVA) para comparar abundâncias entre regiões/períodos
- Testes de sazonalidade e tendências temporais
- Visualização de resultados estatísticos com intervalos de confiança

## 5. Integração de Dados de Clorofila-a
- Processamento e visualização dos dados de Clorofila-a
- Mapas comparativos entre distribuição de plâncton e concentração de clorofila
- Análise de correlação entre abundância de plâncton e níveis de clorofila

## 6. Machine Learning
- Implementação de clustering para identificar comunidades de plâncton
- Modelos de regressão para prever abundância baseada em variáveis ambientais
- Visualização interativa dos resultados dos modelos
- Interface para ajuste de hiperparâmetros e comparação de modelos

## 7. Melhorias na Interface do Utilizador
- Layout responsivo para diferentes tamanhos de ecrã
- Tooltips informativos para ajudar na interpretação dos gráficos
- Sistema de ajuda contextual
- Opção para exportar visualizações e resultados

## 8. Documentação Integrada
- Página "Como Usar" com instruções detalhadas
- Informações metodológicas sobre análises e modelos
- Referências e fontes de dados
