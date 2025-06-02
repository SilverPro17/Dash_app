# Documentação dos Datasets Selecionados

Este documento detalha as fontes, características e métodos de acesso para os datasets selecionados para o projeto "Exploração Interativa e Modelação Preditiva da Distribuição de Plâncton".

## 1. Dataset Principal: Plâncton

*   **Nome:** The CPR Survey (Continuous Plankton Recorder)
*   **Fonte:** Global Biodiversity Information Facility (GBIF) / Marine Biological Association
*   **Link Principal:** [https://www.gbif.org/dataset/6d56415d-b007-4273-9c74-bcd6b2467434](https://www.gbif.org/dataset/6d56415d-b007-4273-9c74-bcd6b2467434)
*   **Formato Recomendado:** Darwin Core Archive (DwC-A)
*   **Método de Acesso:** Download direto via GBIF (requer login).
*   **Estado Atual:** Aguardando upload do ficheiro DwC-A pelo utilizador.
*   **Características:** Contém dados de ocorrência de plâncton (presença/abundância), com coordenadas geográficas, data/hora e informação taxonómica. Cobertura global, mas com maior densidade no Atlântico Norte. Longa série temporal.
*   **Licença:** CC BY 4.0

## 2. Dataset Adicional 1: Temperatura da Superfície do Mar (SST)

*   **Nome:** NOAA OI SST V2 High Resolution Dataset
*   **Fonte:** NOAA Physical Sciences Laboratory (PSL)
*   **Link Principal:** [https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html)
*   **Formato:** NetCDF4
*   **Método de Acesso:** THREDDS Data Server Catalog
*   **Link THREDDS:** [https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.html](https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.html)
*   **Ficheiro(s) Relevantes:** `sst.mon.mean.nc` (para médias mensais), ou ficheiros diários/semanais conforme necessidade.
*   **Características:** Dados globais de SST em grelha regular, alta resolução espacial (0.25° x 0.25°). Cobertura temporal desde Setembro de 1981 até ao presente, com atualizações diárias. Inclui também dados de concentração de gelo marinho.
*   **Licença:** Dados públicos do governo dos EUA.

## 3. Dataset Adicional 2: Clorofila-a

*   **Nome:** Aqua MODIS Level 3 Mapped Chlorophyll-a Concentration
*   **Fonte:** NASA Ocean Biology Processing Group (OB.DAAC)
*   **Link Principal:** [https://oceancolor.gsfc.nasa.gov/](https://oceancolor.gsfc.nasa.gov/)
*   **Formato:** NetCDF ou HDF (a confirmar)
*   **Método de Acesso:** NASA OceanData Level 3 & 4 Browser
*   **Link Browser/Download:** [https://oceandata.sci.gsfc.nasa.gov/l3/](https://oceandata.sci.gsfc.nasa.gov/l3/)
*   **Ficheiro(s) Relevantes:** Selecionar produto "Chlorophyll concentration", instrumento "Aqua-MODIS", período "Monthly" (ou "8-day"), resolução "4km".
*   **Características:** Dados globais de concentração de Clorofila-a derivados de satélite (sensor MODIS no satélite Aqua). Resolução espacial de 4km. Cobertura temporal desde meados de 2002 até ao presente. Disponível em várias resoluções temporais (diária, 8 dias, mensal, etc.).
*   **Licença:** Dados públicos da NASA.

