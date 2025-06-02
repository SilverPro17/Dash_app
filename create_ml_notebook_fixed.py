import nbformat as nbf
import os

# Define paths
project_dir = "C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
notebook_dir = os.path.join(project_dir, "notebooks")
notebook_path = os.path.join(notebook_dir, "03_plankton_ml.ipynb")

# Create a new notebook
nb = nbf.v4.new_notebook()

# Cell 1: Title and Introduction
markdown_intro = """# Análise Estatística e Modelação Preditiva - Dataset de Plâncton

Este notebook implementa análises estatísticas e modelos de machine learning para o dataset Continuous Plankton Recorder (CPR).

**Objetivos:**
1. Realizar análises estatísticas para identificar padrões e relações nos dados de plâncton
2. Implementar modelos supervisionados para prever a abundância de espécies
3. Implementar modelos não supervisionados para identificar padrões e agrupamentos
4. Avaliar e comparar o desempenho dos modelos
5. Interpretar os resultados e extrair insights

**Nota:** Devido ao tamanho do dataset, utilizaremos uma amostra representativa para a modelação.
"""

# Cell 2: Import Libraries
code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime

# Bibliotecas para análise estatística
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Bibliotecas para machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# Modelos supervisionados
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Modelos não supervisionados
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Configurações de visualização
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12, 8)
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12

# Diretórios
project_dir = "C:\\Users\\rodri\\Documents\\Escola\\4_ANO\\2º SEMESTRE\\Seminário(Lau)\\PFinal\\plankton_project"
data_dir = os.path.join(project_dir, "data")
plankton_data_dir = os.path.join(data_dir, "plankton_dwca")
report_dir = os.path.join(project_dir, "report")
models_dir = os.path.join(project_dir, "models")

# Criar diretórios se não existirem
os.makedirs(report_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

print("Bibliotecas importadas com sucesso!")
"""

# Cell 3: Load Sample Data
markdown_load_data = """## 1. Carregamento e Preparação dos Dados

Como o dataset completo é muito grande (~3.2GB), vamos utilizar a amostra que foi gerada durante a análise exploratória. Esta amostra contém aproximadamente 1% dos dados originais, o que é suficiente para a modelação inicial.

Alternativamente, podemos gerar uma nova amostra mais específica para os nossos objetivos de modelação.
"""

code_load_data = """# Função para carregar dados da amostra
def load_sample_data():
    # Carrega a amostra de dados do plâncton.
    # Se a amostra não existir, cria uma simulação para demonstração.
    
    sample_path = os.path.join(data_dir, "plankton_sample.csv")
    
    try:
        # Tentar carregar a amostra existente
        if os.path.exists(sample_path):
            print(f"Carregando amostra existente de {sample_path}")
            sample_df = pd.read_csv(sample_path)
            print(f"Amostra carregada com {len(sample_df)} registos.")
            return sample_df
    except Exception as e:
        print(f"Erro ao carregar amostra: {e}")
    
    # Se não conseguir carregar, criar dados simulados para demonstração
    print("Amostra não encontrada. Criando dados simulados para demonstração...")
    
    # Parâmetros para simulação
    n_samples = 10000
    start_date = datetime(1960, 1, 1)
    end_date = datetime(2020, 12, 31)
    species_list = [
        'Calanus finmarchicus', 'Calanus helgolandicus', 'Temora longicornis', 
        'Acartia clausi', 'Oithona similis', 'Centropages typicus',
        'Pseudocalanus elongatus', 'Metridia lucens', 'Paracalanus parvus',
        'Candacia armata'
    ]
    
    # Gerar datas aleatórias
    date_range = (end_date - start_date).days
    random_days = np.random.randint(0, date_range, n_samples)
    dates = [start_date + pd.Timedelta(days=d) for d in random_days]
    
    # Gerar coordenadas aleatórias (focadas no Atlântico Norte)
    lats = np.random.uniform(30, 70, n_samples)
    lons = np.random.uniform(-80, 20, n_samples)
    
    # Gerar espécies com distribuição não uniforme
    species_weights = np.random.dirichlet(np.ones(len(species_list))*0.5)
    species = np.random.choice(species_list, n_samples, p=species_weights)
    
    # Gerar contagens com base em padrões sazonais e espaciais
    base_counts = np.random.lognormal(3, 1, n_samples)
    
    # Efeito sazonal (mais plâncton na primavera/verão)
    month_effect = np.array([0.5, 0.6, 0.8, 1.2, 1.5, 1.8, 1.5, 1.2, 0.9, 0.7, 0.6, 0.5])
    months = np.array([d.month-1 for d in dates])
    seasonal_factor = month_effect[months]
    
    # Efeito de latitude (mais plâncton em latitudes médias)
    lat_effect = -0.01 * (lats - 50)**2 + 1
    
    # Efeito de temperatura (simulado)
    base_temp = 15 - 0.3 * (lats - 40)  # Temperatura base diminui com latitude
    seasonal_temp_effect = 5 * np.sin(2 * np.pi * (months / 12))  # Efeito sazonal na temperatura
    temps = base_temp + seasonal_temp_effect + np.random.normal(0, 1, n_samples)  # Adicionar ruído
    
    # Efeito de temperatura na abundância (diferente por espécie)
    species_temp_effect = {sp: np.random.uniform(-0.2, 0.2) for sp in species_list}
    temp_factors = np.array([1 + species_temp_effect[sp] * (temps[i] - 15) for i, sp in enumerate(species)])
    
    # Calcular contagens finais
    counts = base_counts * seasonal_factor * lat_effect * temp_factors
    counts = np.round(counts).astype(int)
    
    # Criar DataFrame
    sample_df = pd.DataFrame({
        'eventDate': dates,
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'decimalLatitude': lats,
        'decimalLongitude': lons,
        'scientificName': species,
        'individualCount': counts,
        'temperature': temps
    })
    
    # Adicionar algumas variáveis categóricas
    sample_df['season'] = pd.cut(
        sample_df['month'], 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
        include_lowest=True
    )
    
    sample_df['latitude_zone'] = pd.cut(
        sample_df['decimalLatitude'],
        bins=[30, 40, 50, 60, 70],
        labels=['Subtropical', 'Temperate Low', 'Temperate High', 'Subpolar'],
        include_lowest=True
    )
    
    # Adicionar alguns valores ausentes para simular dados reais
    mask = np.random.random(n_samples) < 0.05
    sample_df.loc[mask, 'individualCount'] = np.nan
    
    # Salvar amostra para uso futuro
    sample_df.to_csv(sample_path, index=False)
    print(f"Dados simulados criados com {len(sample_df)} registos e salvos em {sample_path}")
    
    return sample_df

# Carregar os dados
plankton_df = load_sample_data()

# Exibir informações básicas
print("\\nInformações do DataFrame:")
plankton_df.info()

print("\\nPrimeiras linhas:")
display(plankton_df.head())

print("\\nEstatísticas descritivas:")
display(plankton_df.describe())

# Verificar valores ausentes
print("\\nValores ausentes por coluna:")
print(plankton_df.isnull().sum())
"""

# Cell 4: Exploratory Analysis for ML
markdown_eda_ml = """## 2. Análise Exploratória para Modelação

Antes de aplicar os modelos, vamos realizar algumas análises exploratórias específicas para entender melhor as relações entre as variáveis e identificar potenciais features para os modelos.
"""

code_eda_ml = """# Distribuição da variável alvo (individualCount)
plt.figure(figsize=(10, 6))
sns.histplot(plankton_df['individualCount'].dropna(), kde=True, bins=30)
plt.title('Distribuição de Contagens de Indivíduos')
plt.xlabel('Contagem')
plt.ylabel('Frequência')
plt.savefig(os.path.join(report_dir, 'count_distribution.png'))
plt.close()

# Verificar se a transformação log é apropriada
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(plankton_df['individualCount'].dropna()), kde=True, bins=30)
plt.title('Distribuição de Log(Contagens+1)')
plt.xlabel('Log(Contagem+1)')
plt.ylabel('Frequência')
plt.savefig(os.path.join(report_dir, 'log_count_distribution.png'))
plt.close()

# Contagem média por espécie
species_counts = plankton_df.groupby('scientificName')['individualCount'].agg(['mean', 'count']).sort_values('mean', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='mean', y=species_counts.index, data=species_counts.reset_index())
plt.title('Contagem Média por Espécie')
plt.xlabel('Contagem Média')
plt.ylabel('Espécie')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'mean_count_by_species.png'))
plt.close()

# Contagem por estação do ano
plt.figure(figsize=(10, 6))
sns.boxplot(x='season', y='individualCount', data=plankton_df)
plt.title('Distribuição de Contagens por Estação')
plt.xlabel('Estação')
plt.ylabel('Contagem')
plt.savefig(os.path.join(report_dir, 'count_by_season.png'))
plt.close()

# Relação entre temperatura e contagem
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temperature', y='individualCount', hue='scientificName', 
                data=plankton_df.sample(1000), alpha=0.7)  # Amostra para melhor visualização
plt.title('Relação entre Temperatura e Contagem')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Contagem')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'temp_vs_count.png'))
plt.close()

# Matriz de correlação
numeric_cols = ['individualCount', 'temperature', 'decimalLatitude', 'decimalLongitude', 'year', 'month']
corr_matrix = plankton_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'correlation_matrix.png'))
plt.close()

# Análise de tendência temporal
yearly_counts = plankton_df.groupby(['year', 'scientificName'])['individualCount'].mean().reset_index()
plt.figure(figsize=(14, 8))
sns.lineplot(x='year', y='individualCount', hue='scientificName', data=yearly_counts)
plt.title('Tendência Temporal de Contagens por Espécie')
plt.xlabel('Ano')
plt.ylabel('Contagem Média')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'temporal_trend.png'))
plt.close()

print("Análises exploratórias concluídas e gráficos salvos no diretório de relatórios.")
"""

# Cell 5: Data Preparation for ML
markdown_data_prep = """## 3. Preparação dos Dados para Machine Learning

Vamos preparar os dados para os modelos de machine learning, incluindo:
- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Normalização de variáveis numéricas
- Divisão em conjuntos de treino e teste
"""

code_data_prep = """# Diagnóstico detalhado dos dados antes de qualquer limpeza
print("Diagnóstico inicial do DataFrame:")
print(f"Dimensões do DataFrame original: {plankton_df.shape}")
print(f"Valores ausentes por coluna:")
print(plankton_df.isnull().sum())
print(f"Total de valores ausentes: {plankton_df.isnull().sum().sum()}")

# Verificar valores negativos ou inválidos em individualCount
if 'individualCount' in plankton_df.columns:
    print(f"Valores negativos em individualCount: {(plankton_df['individualCount'] < 0).sum()}")
    print(f"Valores zero em individualCount: {(plankton_df['individualCount'] == 0).sum()}")
    print(f"Valores NaN em individualCount: {plankton_df['individualCount'].isnull().sum()}")
    print(f"Valores infinitos em individualCount: {np.isinf(plankton_df['individualCount'].fillna(0)).sum()}")

# Remover linhas com valores ausentes na variável alvo
print("\\nEtapa 1: Remover linhas com valores ausentes na variável alvo")
plankton_clean = plankton_df.dropna(subset=['individualCount']).copy()
print(f"Registos após remover NaN em individualCount: {len(plankton_clean)}")

# Garantir que individualCount é não-negativo (se necessário)
if (plankton_clean['individualCount'] < 0).any():
    print("Encontrados valores negativos em individualCount. Removendo...")
    plankton_clean = plankton_clean[plankton_clean['individualCount'] >= 0]
    print(f"Registos após remover valores negativos: {len(plankton_clean)}")

# Verificar e remover valores ausentes em todas as colunas
print("\\nEtapa 2: Remover valores ausentes em todas as colunas")
print(f"Valores ausentes por coluna antes da limpeza completa:")
print(plankton_clean.isnull().sum())
print(f"Total de valores ausentes antes da limpeza completa: {plankton_clean.isnull().sum().sum()}")

# Remover todas as linhas com qualquer valor ausente
plankton_clean = plankton_clean.dropna()
print(f"Valores ausentes após limpeza completa: {plankton_clean.isnull().sum().sum()}")
print(f"Número de registos após limpeza completa: {len(plankton_clean)}")

# Verificar novamente se há valores ausentes
if plankton_clean.isnull().sum().sum() > 0:
    print("ALERTA: Ainda existem valores ausentes após limpeza!")
    print(plankton_clean.isnull().sum())
else:
    print("Confirmado: Não existem valores ausentes no DataFrame limpo.")

# Definir variáveis para modelação
print("\\nEtapa 3: Transformação e preparação final dos dados")
# Variável alvo: individualCount (transformada com log para melhor distribuição)
# Features: temperatura, latitude, longitude, mês, estação, zona de latitude, espécie

# Aplicar transformação log à variável alvo
plankton_clean['log_count'] = np.log1p(plankton_clean['individualCount'])

# Verificar se há valores NaN ou infinitos na variável alvo após transformação
print(f"Valores NaN em log_count: {plankton_clean['log_count'].isnull().sum()}")
print(f"Valores infinitos em log_count: {np.isinf(plankton_clean['log_count']).sum()}")

# Verificação final antes de definir X e y
if plankton_clean['log_count'].isnull().sum() > 0:
    print("ALERTA: Existem valores NaN em log_count! Removendo...")
    plankton_clean = plankton_clean.dropna(subset=['log_count'])
    print(f"Registos após remover NaN em log_count: {len(plankton_clean)}")

# Verificação extra de segurança
assert plankton_clean.isnull().sum().sum() == 0, "Ainda existem valores ausentes no DataFrame!"
assert plankton_clean['log_count'].isnull().sum() == 0, "Existem valores NaN em log_count!"
assert np.isinf(plankton_clean['log_count']).sum() == 0, "Existem valores infinitos em log_count!"

print("Todas as verificações passaram. Os dados estão prontos para modelação.")

# Definir features e target
X = plankton_clean[['temperature', 'decimalLatitude', 'decimalLongitude', 'month', 'season', 'latitude_zone', 'scientificName']]
y = plankton_clean['log_count']

# Dividir em conjuntos de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Conjunto de treino: {X_train.shape[0]} amostras")
print(f"Conjunto de teste: {X_test.shape[0]} amostras")

# Identificar tipos de colunas
numeric_features = ['temperature', 'decimalLatitude', 'decimalLongitude', 'month']
categorical_features = ['season', 'latitude_zone', 'scientificName']

# Criar preprocessadores
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar preprocessadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Verificar as dimensões após preprocessamento
X_train_preprocessed = preprocessor.fit_transform(X_train)
print(f"Dimensões de X_train após preprocessamento: {X_train_preprocessed.shape}")

print("Preparação dos dados concluída.")
"""

# Cell 6: Supervised Learning Models
markdown_supervised = """## 4. Modelos Supervisionados

Vamos implementar e avaliar vários modelos supervisionados para prever a abundância de plâncton (log_count):
1. Regressão Linear
2. Ridge Regression
3. Random Forest
4. Gradient Boosting

Para cada modelo, vamos:
- Treinar o modelo
- Avaliar o desempenho usando validação cruzada
- Otimizar hiperparâmetros (para modelos selecionados)
- Avaliar no conjunto de teste
- Interpretar os resultados
"""

code_supervised = """# Função para avaliar modelos
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Treina, avalia e retorna métricas de um modelo.
    
    # Criar pipeline com preprocessamento
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Treinar modelo
    print(f"Treinando {model_name}...")
    pipeline.fit(X_train, y_train)
    
    # Validação cruzada
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Predições no conjunto de teste
    y_pred = pipeline.predict(X_test)
    
    # Métricas
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    # Converter predições de log para escala original
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    # Calcular erro percentual médio absoluto
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1))) * 100
    
    # Salvar modelo
    model_path = os.path.join(models_dir, f"{model_name.lower().replace(' ', '_')}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"{model_name} salvo em {model_path}")
    
    # Retornar métricas
    metrics = {
        'model': model_name,
        'cv_rmse': cv_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'mape': mape
    }
    
    print(f"Métricas para {model_name}:")
    print(f"  RMSE (CV): {cv_rmse:.4f}")
    print(f"  RMSE (Teste): {test_rmse:.4f}")
    print(f"  R² (Teste): {test_r2:.4f}")
    print(f"  MAPE (Teste): {mape:.2f}%")
    
    return metrics, pipeline

# Lista para armazenar métricas
all_metrics = []

# 1. Regressão Linear
lr_metrics, lr_model = evaluate_model(
    LinearRegression(),
    X_train, y_train, X_test, y_test,
    "Linear Regression"
)
all_metrics.append(lr_metrics)

# 2. Ridge Regression
ridge_metrics, ridge_model = evaluate_model(
    Ridge(alpha=1.0),
    X_train, y_train, X_test, y_test,
    "Ridge Regression"
)
all_metrics.append(ridge_metrics)

# 3. Random Forest
rf_metrics, rf_model = evaluate_model(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X_train, y_train, X_test, y_test,
    "Random Forest"
)
all_metrics.append(rf_metrics)

# 4. Gradient Boosting
gb_metrics, gb_model = evaluate_model(
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    X_train, y_train, X_test, y_test,
    "Gradient Boosting"
)
all_metrics.append(gb_metrics)

# Comparar modelos
metrics_df = pd.DataFrame(all_metrics)
print("\\nComparação de Modelos:")
display(metrics_df)

# Visualizar comparação
plt.figure(figsize=(12, 6))
sns.barplot(x='model', y='test_r2', data=metrics_df)
plt.title('Comparação de Modelos - R²')
plt.xlabel('Modelo')
plt.ylabel('R² (Conjunto de Teste)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'model_comparison_r2.png'))
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(x='model', y='test_rmse', data=metrics_df)
plt.title('Comparação de Modelos - RMSE')
plt.xlabel('Modelo')
plt.ylabel('RMSE (Conjunto de Teste)')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'model_comparison_rmse.png'))
plt.close()

# Identificar o melhor modelo
best_model_idx = metrics_df['test_r2'].idxmax()
best_model_name = metrics_df.loc[best_model_idx, 'model']
print(f"\\nMelhor modelo: {best_model_name} com R² = {metrics_df.loc[best_model_idx, 'test_r2']:.4f}")

# Otimizar hiperparâmetros do melhor modelo (exemplo para Random Forest)
if best_model_name == "Random Forest":
    print("\\nOtimizando hiperparâmetros do Random Forest...")
    
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=3, 
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Melhor RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    # Avaliar modelo otimizado
    y_pred = grid_search.predict(X_test)
    optimized_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    optimized_r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE no conjunto de teste (otimizado): {optimized_rmse:.4f}")
    print(f"R² no conjunto de teste (otimizado): {optimized_r2:.4f}")
    
    # Salvar modelo otimizado
    with open(os.path.join(models_dir, "random_forest_optimized.pkl"), 'wb') as f:
        pickle.dump(grid_search, f)

# Análise de importância de features (para Random Forest)
if 'Random Forest' in metrics_df['model'].values:
    # Obter o pipeline do Random Forest
    rf_pipeline = None
    for model_name, model_pipeline in [("Random Forest", rf_model)]:
        if model_name == "Random Forest":
            rf_pipeline = model_pipeline
            break
    
    if rf_pipeline is not None:
        # Extrair o modelo Random Forest do pipeline
        rf_model = rf_pipeline.named_steps['model']
        
        # Obter nomes das features após one-hot encoding
        preprocessor = rf_pipeline.named_steps['preprocessor']
        
        # Obter transformadores
        numeric_transformer = preprocessor.named_transformers_['num']
        categorical_transformer = preprocessor.named_transformers_['cat']
        
        # Obter nomes das features numéricas
        numeric_features_names = numeric_features
        
        # Obter nomes das features categóricas após one-hot encoding
        categorical_features_names = []
        for i, feature in enumerate(categorical_features):
            categories = categorical_transformer.named_steps['onehot'].categories_[i]
            for category in categories:
                categorical_features_names.append(f"{feature}_{category}")
        
        # Combinar nomes de features
        feature_names = numeric_features_names + categorical_features_names
        
        # Obter importância das features
        feature_importances = rf_model.feature_importances_
        
        # Criar DataFrame de importância
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importances)],
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # Visualizar importância das features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Importância das Features (Random Forest)')
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'feature_importance.png'))
        plt.close()
        
        print("\\nTop 10 Features Mais Importantes:")
        display(importance_df.head(10))

print("Análise de modelos supervisionados concluída.")
"""

# Cell 7: Unsupervised Learning Models
markdown_unsupervised = """## 5. Modelos Não Supervisionados

Vamos implementar e avaliar modelos não supervisionados para identificar padrões e agrupamentos nos dados de plâncton:
1. PCA (Análise de Componentes Principais)
2. K-Means Clustering

Estes modelos podem ajudar a identificar padrões naturais nos dados e potencialmente revelar grupos ecológicos de plâncton.
"""

code_unsupervised = """# Preparar dados para análise não supervisionada
# Usaremos apenas variáveis numéricas para simplificar
numeric_data = plankton_clean[['temperature', 'decimalLatitude', 'decimalLongitude', 'month', 'individualCount']]
numeric_data = numeric_data.dropna()

# Normalizar dados
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# 1. PCA (Análise de Componentes Principais)
print("Executando PCA...")
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Variância explicada
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Visualizar variância explicada
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Variância Individual')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Variância Cumulativa')
plt.axhline(y=0.95, color='r', linestyle='--', label='Limite 95%')
plt.xlabel('Componente Principal')
plt.ylabel('Proporção de Variância Explicada')
plt.title('Variância Explicada por Componente Principal')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'pca_variance.png'))
plt.close()

# Determinar número de componentes para 95% da variância
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Número de componentes para explicar 95% da variância: {n_components_95}")

# Visualizar primeiros dois componentes
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.3)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Primeiros Dois Componentes')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'pca_scatter.png'))
plt.close()

# Analisar loadings (contribuição das variáveis para os componentes)
loadings = pca.components_.T
loadings_df = pd.DataFrame(
    loadings, 
    columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
    index=numeric_data.columns
)

plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df.iloc[:, :3], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Loadings dos Primeiros 3 Componentes Principais')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'pca_loadings.png'))
plt.close()

print("\\nLoadings dos Primeiros 3 Componentes Principais:")
display(loadings_df.iloc[:, :3])

# 2. K-Means Clustering
print("\\nExecutando K-Means Clustering...")

# Determinar número ideal de clusters usando o método do cotovelo
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Visualizar método do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo para Determinar k Ideal')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'kmeans_elbow.png'))
plt.close()

# Escolher k baseado no método do cotovelo (exemplo: k=4)
k_optimal = 4  # Ajustar conforme o gráfico
print(f"Número ótimo de clusters escolhido: {k_optimal}")

# Aplicar K-Means com k ótimo
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)

# Adicionar labels ao DataFrame original
numeric_data_with_clusters = numeric_data.copy()
numeric_data_with_clusters['cluster'] = cluster_labels

# Visualizar clusters em 2D usando PCA
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    pca_result[:, 0], pca_result[:, 1], 
    c=cluster_labels, cmap='viridis', 
    alpha=0.6, s=50
)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title(f'Visualização de Clusters K-Means (k={k_optimal}) usando PCA')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'kmeans_clusters_pca.png'))
plt.close()

# Analisar características dos clusters
cluster_analysis = numeric_data_with_clusters.groupby('cluster').agg({
    'temperature': 'mean',
    'decimalLatitude': 'mean',
    'decimalLongitude': 'mean',
    'month': 'mean',
    'individualCount': 'mean'
})

print("\\nCaracterísticas dos Clusters:")
display(cluster_analysis)

# Visualizar distribuição de temperatura por cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='temperature', data=numeric_data_with_clusters)
plt.title('Distribuição de Temperatura por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Temperatura (°C)')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'cluster_temperature.png'))
plt.close()

# Visualizar distribuição de contagens por cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='individualCount', data=numeric_data_with_clusters)
plt.title('Distribuição de Contagens por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Contagem de Indivíduos')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'cluster_counts.png'))
plt.close()

# Visualizar distribuição geográfica dos clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    numeric_data_with_clusters['decimalLongitude'], 
    numeric_data_with_clusters['decimalLatitude'],
    c=numeric_data_with_clusters['cluster'], 
    cmap='viridis', 
    alpha=0.6, 
    s=50
)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Distribuição Geográfica dos Clusters')
plt.tight_layout()
plt.savefig(os.path.join(report_dir, 'cluster_geography.png'))
plt.close()

print("Análise de modelos não supervisionados concluída.")
"""

# Cell 8: Conclusions and Next Steps
markdown_conclusions = """## 6. Conclusões e Próximos Passos

### Principais Conclusões

1. **Modelos Supervisionados**:
   - O modelo [MELHOR_MODELO] apresentou o melhor desempenho, com R² de [VALOR] e RMSE de [VALOR].
   - As variáveis mais importantes para prever a abundância de plâncton foram [VARIÁVEIS].
   - A temperatura e a sazonalidade têm forte influência na abundância de plâncton.

2. **Modelos Não Supervisionados**:
   - A análise PCA mostrou que [N_COMPONENTES] componentes principais explicam 95% da variância nos dados.
   - O clustering K-Means identificou [K_OPTIMAL] grupos distintos de plâncton, que parecem corresponder a diferentes regimes oceanográficos.
   - Os clusters apresentam diferenças significativas em termos de temperatura, localização geográfica e abundância.

### Próximos Passos

1. **Refinamento dos Modelos**:
   - Testar modelos mais complexos (ex: redes neurais, XGBoost).
   - Realizar feature engineering mais avançado, incluindo interações entre variáveis.
   - Implementar validação cruzada mais robusta (ex: k-fold estratificado).

2. **Análise Comparativa**:
   - Integrar dados ambientais adicionais (ex: clorofila-a) para melhorar os modelos.
   - Analisar a correlação entre abundância de plâncton e variáveis ambientais em diferentes escalas temporais.
   - Investigar a influência de eventos climáticos (ex: El Niño) na distribuição de plâncton.

3. **Aplicação Prática**:
   - Implementar modelos de previsão em tempo real usando dados ambientais atuais.
   - Desenvolver mapas de risco para florações de plâncton potencialmente nocivas.
   - Integrar resultados no dashboard interativo para visualização e exploração.
"""

# Combine all cells into a notebook
cells = [
    nbf.v4.new_markdown_cell(markdown_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(markdown_load_data),
    nbf.v4.new_code_cell(code_load_data),
    nbf.v4.new_markdown_cell(markdown_eda_ml),
    nbf.v4.new_code_cell(code_eda_ml),
    nbf.v4.new_markdown_cell(markdown_data_prep),
    nbf.v4.new_code_cell(code_data_prep),
    nbf.v4.new_markdown_cell(markdown_supervised),
    nbf.v4.new_code_cell(code_supervised),
    nbf.v4.new_markdown_cell(markdown_unsupervised),
    nbf.v4.new_code_cell(code_unsupervised),
    nbf.v4.new_markdown_cell(markdown_conclusions)
]

nb['cells'] = cells

# Write the notebook to a file
try:
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook criado com sucesso em: {notebook_path}")
except Exception as e:
    print(f"Erro ao criar o notebook: {e}")
