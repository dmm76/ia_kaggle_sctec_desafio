# Desafio Extra C2 — Introdução à Inteligência Artificial

## Análise Exploratória e Modelo Preditivo para Reservas de Hotel

🔗 **Repositório GitHub:** [https://github.com/dmm76/ia_kaggle_sctec_desafio.git](https://github.com/dmm76/ia_kaggle_sctec_desafio.git)

---

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Análise%20de%20Dados-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

---

## 1. Identificação do projeto

**Autor:** Douglas Marcelo Monquero  
**Curso:** SCTEC — Trilha de Inteligência Artificial  
**Atividade:** Desafio Extra do curso Introdução à Inteligência Artificial  
**Dataset:** Hotel Booking Demand  
**Fonte dos dados:** Kaggle — Hotel Booking Demand  
**Linguagem utilizada:** Python  
**Tipo de problema:** Classificação supervisionada  
**Variável-alvo:** `is_canceled`

---

## 2. Resumo do projeto

Este projeto foi desenvolvido como parte do **Desafio Extra C2 — Introdução à Inteligência Artificial**, com o objetivo de aplicar conhecimentos introdutórios de análise de dados e aprendizado de máquina em uma base pública real.

A base utilizada foi a **Hotel Booking Demand**, que contém informações sobre reservas de dois tipos de hotel:

- **City Hotel**
- **Resort Hotel**

O projeto realiza uma **Análise Exploratória de Dados (AED)** para compreender o comportamento das reservas, identificar padrões de cancelamento e analisar relações entre variáveis como tipo de hotel, antecedência da reserva, mês de chegada, país de origem, tarifa diária média e tipo de cliente.

Além da análise exploratória, foi desenvolvido um **modelo preditivo de Inteligência Artificial** para prever se uma reserva será ou não cancelada, utilizando a variável `is_canceled` como alvo.

---

## 3. Objetivos

Os principais objetivos deste projeto são:

1. Importar e compreender o conjunto de dados.
2. Verificar a estrutura do dataset, tipos de dados e valores ausentes.
3. Tratar valores nulos, duplicados e inconsistências.
4. Criar variáveis derivadas para enriquecer a análise.
5. Realizar análise exploratória com filtros, agrupamentos e visualizações.
6. Investigar padrões de cancelamento por hotel, mês, país, antecedência e perfil do cliente.
7. Preparar os dados para modelagem preditiva.
8. Separar os dados em conjuntos de treino e teste.
9. Treinar um modelo de classificação.
10. Avaliar o modelo com métricas adequadas, como acurácia, matriz de confusão, precisão, recall e F1-score.

---

## 4. Estrutura final do projeto

A estrutura final do projeto ficou organizada da seguinte forma:

```text
desafio-hotel-booking-ia/
│
├── graficos/
│   ├── adr_por_cancelamento.png
│   ├── cancelamento_por_tipo_cliente.png
│   ├── cancelamentos_por_tipo_hotel.png
│   ├── lead_time_cancelamento.png
│   ├── matriz_confusao.png
│   ├── reservas_por_mes.png
│   ├── taxa_cancelamento_por_hotel.png
│   ├── taxa_cancelamento_por_mes.png
│   └── top_paises_reservas.png
│
├── hotel_booking_analysis.py
├── hotel_bookings.csv
├── README.md
├── relatorio_modelo.txt
└── requirements.txt
```

A pasta `.venv` foi utilizada apenas no ambiente local de desenvolvimento e **não deve ser incluída no arquivo compactado final**, pois não é necessária para a avaliação e pode deixar o projeto muito grande.

---

## 5. Tecnologias e bibliotecas utilizadas

O projeto foi desenvolvido em **Python**, com apoio das seguintes bibliotecas:

| Biblioteca | Finalidade |
|---|---|
| `pandas` | Carregamento, limpeza, transformação e análise dos dados |
| `numpy` | Operações numéricas e criação de variáveis auxiliares |
| `matplotlib` | Geração de gráficos |
| `seaborn` | Visualizações estatísticas |
| `scikit-learn` | Pré-processamento, treinamento e avaliação do modelo |
| `pathlib` | Manipulação de caminhos de arquivos e pastas |

---

## 6. Como executar o projeto

### 6.1 Criar e ativar o ambiente virtual

No PowerShell, dentro da pasta do projeto:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Após a ativação, o terminal deve exibir o prefixo:

```text
(.venv)
```

### 6.2 Instalar as dependências

```powershell
pip install -r requirements.txt
```

Caso o arquivo `requirements.txt` precise ser gerado novamente:

```powershell
pip freeze > requirements.txt
```

### 6.3 Executar o script principal

```powershell
python hotel_booking_analysis.py
```

Ao final da execução, o projeto gera:

- gráficos na pasta `graficos/`;
- matriz de confusão em `graficos/matriz_confusao.png`;
- relatório textual em `relatorio_modelo.txt`;
- resultados impressos no terminal.

---

## 7. Dataset

O dataset utilizado foi o **Hotel Booking Demand**, disponível publicamente no Kaggle.

O arquivo utilizado no projeto é:

```text
hotel_bookings.csv
```

Esse arquivo contém informações sobre reservas hoteleiras, incluindo dados como:

- tipo de hotel;
- status de cancelamento;
- antecedência da reserva;
- ano, mês e dia de chegada;
- quantidade de hóspedes;
- país de origem;
- segmento de mercado;
- canal de distribuição;
- tipo de quarto;
- tipo de depósito;
- tarifa diária média;
- solicitações especiais;
- status final da reserva.

A variável-alvo utilizada na modelagem preditiva foi:

```python
is_canceled
```

Representando:

| Valor | Significado |
|---|---|
| `0` | Reserva não cancelada |
| `1` | Reserva cancelada |

---

## 8. Tratamento e preparação dos dados

Após o carregamento inicial, o dataset apresentou:

```text
119.390 registros
32 colunas
```

Durante a etapa de preparação dos dados, foram realizadas as seguintes ações:

### 8.1 Remoção de duplicados

Foram identificados e removidos:

```text
31.994 registros duplicados
```

### 8.2 Tratamento de valores nulos

As colunas com valores ausentes foram tratadas da seguinte forma:

| Coluna | Tratamento aplicado |
|---|---|
| `company` | Valores nulos substituídos por `0` |
| `agent` | Valores nulos substituídos por `0` |
| `country` | Valores nulos substituídos por `Unknown` |
| `children` | Valores nulos substituídos por `0` |

A decisão de preencher `company` e `agent` com `0` foi tomada porque essas colunas representam identificadores. Dessa forma, o valor `0` indica ausência de empresa ou agente associado à reserva.

### 8.3 Conversão de tipos

Foram convertidas colunas numéricas que estavam em formato decimal, mas representavam valores inteiros:

- `children`
- `agent`
- `company`

Também foi convertida a coluna:

```python
reservation_status_date
```

para o formato de data.

### 8.4 Criação de variáveis derivadas

Foram criadas variáveis auxiliares para enriquecer a análise:

| Variável | Descrição |
|---|---|
| `total_guests` | Soma de adultos, crianças e bebês |
| `total_nights` | Soma de noites de fim de semana e noites durante a semana |
| `lead_time_group` | Faixas de antecedência da reserva |
| `arrival_month_number` | Mês de chegada convertido para número |
| `has_agent` | Indica se a reserva possui agente |
| `has_company` | Indica se a reserva possui empresa |
| `is_family` | Indica se há crianças ou bebês na reserva |
| `has_previous_cancellations` | Indica histórico anterior de cancelamento |
| `country_grouped` | Agrupamento dos principais países e categoria `Other` |

### 8.5 Tratamento de inconsistências

Foram removidos:

```text
166 registros sem hóspedes
1 registro com ADR negativo
1 registro com ADR acima de 1000
```

Após a limpeza, o dataset tratado ficou com:

```text
87.228 registros
34 colunas
```

---

## 9. Análise Exploratória de Dados

A análise exploratória buscou compreender os principais padrões presentes no dataset.

Foram realizadas análises sobre:

- taxa geral de cancelamento;
- cancelamentos por tipo de hotel;
- distribuição de reservas por mês;
- taxa de cancelamento por mês;
- relação entre antecedência da reserva e cancelamento;
- tarifa diária média por status da reserva;
- países com maior número de reservas;
- cancelamento por tipo de cliente.

---

## 10. Principais insights obtidos

### 10.1 Taxa geral de cancelamento

A taxa geral de cancelamento observada foi de aproximadamente:

```text
27,52%
```

Isso indica que cerca de 1 em cada 4 reservas foi cancelada.

### 10.2 Cancelamento por tipo de hotel

| Tipo de hotel | Total de reservas | Cancelamentos | Taxa de cancelamento |
|---|---:|---:|---:|
| City Hotel | 53.273 | 16.034 | 30,10% |
| Resort Hotel | 33.955 | 7.974 | 23,48% |

O **City Hotel** apresentou maior taxa de cancelamento em comparação ao **Resort Hotel**.

### 10.3 Reservas por mês

Os meses com maior volume de reservas foram:

| Mês | Quantidade de reservas |
|---|---:|
| August | 11.242 |
| July | 10.043 |
| May | 8.344 |
| April | 7.900 |
| June | 7.756 |

Os meses de julho e agosto se destacaram pelo alto volume de reservas, indicando possível influência de sazonalidade.

### 10.4 Taxa de cancelamento por mês

Os meses com maiores taxas de cancelamento foram:

| Mês | Taxa de cancelamento |
|---|---:|
| August | 32,22% |
| July | 31,82% |
| April | 30,46% |
| June | 30,34% |
| May | 29,27% |

Os meses com maior movimento também apresentaram taxas elevadas de cancelamento.

### 10.5 Lead time e cancelamento

A análise da antecedência da reserva mostrou uma tendência clara: quanto maior o `lead_time`, maior a taxa de cancelamento.

| Faixa de antecedência | Taxa de cancelamento |
|---|---:|
| 0–7 dias | 8,42% |
| 8–30 dias | 25,39% |
| 31–90 dias | 32,04% |
| 91–180 dias | 35,01% |
| 181–365 dias | 39,69% |
| Acima de 365 dias | 40,78% |

Esse é um dos principais insights do projeto, pois mostra que reservas feitas com muita antecedência tendem a apresentar maior risco de cancelamento.

### 10.6 Tarifa diária média e cancelamento

| Status da reserva | ADR médio |
|---|---:|
| Não cancelada | 102,22 |
| Cancelada | 117,63 |

As reservas canceladas apresentaram tarifa diária média superior às reservas não canceladas, sugerindo que valores mais altos podem estar associados a maior chance de cancelamento.

### 10.7 Países com mais reservas

Os países com maior número de reservas foram:

| País | Quantidade |
|---|---:|
| PRT | 27.354 |
| GBR | 10.423 |
| FRA | 8.823 |
| ESP | 7.244 |
| DEU | 5.385 |
| ITA | 3.061 |
| IRL | 3.015 |
| BEL | 2.081 |
| BRA | 1.993 |
| NLD | 1.910 |

Portugal concentrou o maior volume de reservas do dataset.

### 10.8 Cancelamento por tipo de cliente

| Tipo de cliente | Taxa de cancelamento |
|---|---:|
| Transient | 30,14% |
| Contract | 16,33% |
| Transient-Party | 15,25% |
| Group | 9,80% |

Clientes do tipo **Transient** apresentaram a maior taxa de cancelamento.

---

## 11. Visualizações geradas

As visualizações foram salvas na pasta `graficos/`.

| Arquivo | Descrição |
|---|---|
| `cancelamentos_por_tipo_hotel.png` | Comparação entre reservas canceladas e não canceladas por tipo de hotel |
| `taxa_cancelamento_por_hotel.png` | Taxa de cancelamento por tipo de hotel |
| `reservas_por_mes.png` | Quantidade de reservas por mês |
| `taxa_cancelamento_por_mes.png` | Taxa de cancelamento por mês |
| `lead_time_cancelamento.png` | Relação entre antecedência da reserva e cancelamento |
| `adr_por_cancelamento.png` | Tarifa diária média por status da reserva |
| `top_paises_reservas.png` | Países com maior número de reservas |
| `cancelamento_por_tipo_cliente.png` | Taxa de cancelamento por tipo de cliente |
| `matriz_confusao.png` | Matriz de confusão do modelo preditivo |

---

## 12. Modelagem preditiva

A etapa de modelagem preditiva teve como objetivo criar um modelo de classificação capaz de prever se uma reserva seria cancelada ou não.

### 12.1 Variável-alvo

```python
is_canceled
```

### 12.2 Variáveis utilizadas

Foram utilizadas variáveis numéricas e categóricas relacionadas à reserva, como:

- `lead_time`
- `arrival_date_year`
- `arrival_month_number`
- `arrival_date_week_number`
- `stays_in_weekend_nights`
- `stays_in_week_nights`
- `adults`
- `children`
- `babies`
- `previous_cancellations`
- `booking_changes`
- `days_in_waiting_list`
- `adr`
- `total_of_special_requests`
- `total_guests`
- `total_nights`
- `hotel`
- `meal`
- `market_segment`
- `distribution_channel`
- `reserved_room_type`
- `deposit_type`
- `customer_type`
- `country_grouped`

As colunas `reservation_status` e `reservation_status_date` não foram utilizadas no treinamento, pois representam informações finais da reserva e poderiam causar vazamento de informação, tornando o modelo artificialmente bom.

### 12.3 Separação entre treino e teste

O dataset tratado foi separado em:

```text
69.782 registros para treino
17.446 registros para teste
```

Foi utilizada divisão estratificada para preservar a proporção entre reservas canceladas e não canceladas.

### 12.4 Modelo utilizado

O modelo escolhido foi:

```python
RandomForestClassifier
```

A escolha do Random Forest se justifica por ser um algoritmo robusto, adequado para problemas de classificação e capaz de lidar bem com diferentes tipos de variáveis após o pré-processamento.

Foi utilizado um pipeline com:

- `ColumnTransformer`
- `OneHotEncoder`
- `RandomForestClassifier`

Esse pipeline permitiu tratar variáveis categóricas e numéricas de forma organizada antes do treinamento do modelo.

---

## 13. Avaliação do modelo

O modelo obteve:

```text
Acurácia: 0,7752
```

Ou seja, aproximadamente:

```text
77,52%
```

### 13.1 Matriz de confusão

```text
[[9363 3281]
 [ 641 4161]]
```

Interpretação:

| Resultado | Quantidade |
|---|---:|
| Reservas não canceladas classificadas corretamente | 9.363 |
| Reservas não canceladas classificadas como canceladas | 3.281 |
| Reservas canceladas classificadas como não canceladas | 641 |
| Reservas canceladas classificadas corretamente | 4.161 |

### 13.2 Relatório de classificação

| Classe | Precisão | Recall | F1-score | Suporte |
|---|---:|---:|---:|---:|
| Não cancelada | 0,94 | 0,74 | 0,83 | 12.644 |
| Cancelada | 0,56 | 0,87 | 0,68 | 4.802 |
| Acurácia |  |  | 0,78 | 17.446 |
| Média macro | 0,75 | 0,80 | 0,75 | 17.446 |
| Média ponderada | 0,83 | 0,78 | 0,79 | 17.446 |

### 13.3 Interpretação dos resultados

O modelo apresentou desempenho adequado para um projeto introdutório de Inteligência Artificial.

O principal ponto positivo foi o **recall da classe “Cancelada”**, que atingiu:

```text
0,87
```

Isso indica que o modelo conseguiu identificar boa parte das reservas que realmente foram canceladas.

Por outro lado, a precisão da classe “Cancelada” foi de:

```text
0,56
```

Isso mostra que o modelo também classificou algumas reservas não canceladas como canceladas, gerando falsos positivos.

Mesmo assim, considerando o objetivo educacional do projeto, o resultado é satisfatório, pois demonstra um fluxo completo de aprendizado de máquina:

1. carregamento dos dados;
2. limpeza e preparação;
3. análise exploratória;
4. geração de gráficos;
5. engenharia de atributos;
6. divisão entre treino e teste;
7. treinamento de modelo;
8. avaliação com métricas.

---

## 14. Arquivos gerados

Ao executar o projeto, são gerados os seguintes arquivos:

```text
graficos/adr_por_cancelamento.png
graficos/cancelamento_por_tipo_cliente.png
graficos/cancelamentos_por_tipo_hotel.png
graficos/lead_time_cancelamento.png
graficos/matriz_confusao.png
graficos/reservas_por_mes.png
graficos/taxa_cancelamento_por_hotel.png
graficos/taxa_cancelamento_por_mes.png
graficos/top_paises_reservas.png
relatorio_modelo.txt
```

---

## 15. Possíveis melhorias futuras

Como melhorias futuras, poderiam ser realizadas:

- comparação entre diferentes modelos, como Árvore de Decisão, Regressão Logística, Gradient Boosting e XGBoost;
- ajuste de hiperparâmetros com Grid Search ou Randomized Search;
- análise de importância das variáveis do modelo;
- avaliação com curva ROC e AUC;
- balanceamento mais refinado da classe minoritária;
- criação de um notebook com explicações visuais adicionais;
- exportação do modelo treinado para uso em uma aplicação.

---

## 16. Como compactar para entrega

A entrega deve ser feita em um único arquivo `.zip` ou `.rar`.

No PowerShell, é possível gerar o `.zip` com o comando:

```powershell
Compress-Archive -Path .\hotel_booking_analysis.py, .\hotel_bookings.csv, .\README.md, .\requirements.txt, .\relatorio_modelo.txt, .\graficos -DestinationPath .\desafio-hotel-booking-ia.zip -Force
```

Para conferir o tamanho do arquivo:

```powershell
(Get-Item .\desafio-hotel-booking-ia.zip).Length / 1MB
```

O arquivo final deve ter até **20 MB**.

---

## 17. Checklist de entrega

- [x] Código-fonte em Python.
- [x] Dataset `hotel_bookings.csv`.
- [x] Pasta `graficos/` com visualizações geradas.
- [x] Arquivo `requirements.txt`.
- [x] Arquivo `relatorio_modelo.txt`.
- [x] Documentação `README.md`.
- [ ] Arquivo `.zip` final gerado.
- [ ] Submissão confirmada na plataforma.

---

## 18. Conclusão

O projeto demonstrou a aplicação prática de conceitos introdutórios de Inteligência Artificial e análise de dados em um problema real de reservas hoteleiras.

A análise exploratória permitiu identificar padrões importantes, como maior taxa de cancelamento no City Hotel, influência da antecedência da reserva no cancelamento e maior risco associado a clientes do tipo Transient.

A etapa de modelagem preditiva mostrou que é possível construir um classificador capaz de prever cancelamentos com desempenho razoável. O modelo Random Forest alcançou acurácia de aproximadamente 77,52% e apresentou bom recall para a classe “Cancelada”, indicando capacidade relevante de identificar reservas com maior risco de cancelamento.

Dessa forma, o projeto atende aos requisitos propostos no desafio, contemplando importação e tratamento dos dados, análise exploratória, geração de gráficos, criação de variáveis derivadas, treinamento de modelo preditivo e avaliação dos resultados.

---

## 19. Autor

**Douglas Marcelo Monquero**  
Estudante de Engenharia de Software  
Desenvolvedor de Sistemas  
SCTEC — Trilha de Inteligência Artificial
