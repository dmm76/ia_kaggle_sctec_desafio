# ============================================
# Desafio Extra - Introdução à Inteligência Artificial
# Projeto: Análise de Reservas de Hotéis
# Dataset: Hotel Booking Demand
# Autor: Douglas Marcelo Monquero
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from pathlib import Path

sns.set_theme(style="whitegrid")

DATA_PATH = Path("hotel_bookings.csv")

df = pd.read_csv(DATA_PATH)

print("Primeiras linhas do dataset:")
print(df.head())

print("\nDimensão do dataset:")
print(f"O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas.")

print("\nInformações gerais:")
print(df.info())

print("\nValores nulos por coluna:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

# ============================================================
# 2. TRATAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================

print("\n" + "=" * 60)
print("2. TRATAMENTO E PREPARAÇÃO DOS DADOS")
print("=" * 60)

# Criando uma cópia para preservar o dataset original
df_clean = df.copy()

# Verificando duplicados antes do tratamento
duplicados_antes = df_clean.duplicated().sum()
print(f"\nRegistros duplicados antes do tratamento: {duplicados_antes}")

# Removendo registros duplicados
df_clean = df_clean.drop_duplicates()

duplicados_depois = df_clean.duplicated().sum()
print(f"Registros duplicados depois do tratamento: {duplicados_depois}")

# Tratamento de valores nulos
# company e agent são identificadores. Quando estão nulos, vamos considerar como 0,
# indicando ausência de empresa/agente associado à reserva.
df_clean["company"] = df_clean["company"].fillna(0)
df_clean["agent"] = df_clean["agent"].fillna(0)

# country nulo será tratado como "Unknown"
df_clean["country"] = df_clean["country"].fillna("Unknown")

# children nulo será tratado como 0
df_clean["children"] = df_clean["children"].fillna(0)

# Conversão de tipos
df_clean["children"] = df_clean["children"].astype(int)
df_clean["agent"] = df_clean["agent"].astype(int)
df_clean["company"] = df_clean["company"].astype(int)

# Convertendo reservation_status_date para data
df_clean["reservation_status_date"] = pd.to_datetime(
    df_clean["reservation_status_date"],
    errors="coerce"
)

# Criando coluna de total de hóspedes
df_clean["total_guests"] = (
    df_clean["adults"] + df_clean["children"] + df_clean["babies"]
)

# Criando coluna de total de noites
df_clean["total_nights"] = (
    df_clean["stays_in_weekend_nights"] + df_clean["stays_in_week_nights"]
)

# Removendo reservas sem hóspedes
linhas_antes = df_clean.shape[0]
df_clean = df_clean[df_clean["total_guests"] > 0]
linhas_depois = df_clean.shape[0]

print(f"\nRegistros removidos por não possuírem hóspedes: {linhas_antes - linhas_depois}")

# Verificando valores negativos em ADR
adr_negativos = (df_clean["adr"] < 0).sum()
print(f"Registros com ADR negativo: {adr_negativos}")

# Removendo ADR negativo, se existir
df_clean = df_clean[df_clean["adr"] >= 0]

# Tratamento simples de outliers em ADR
# Vamos limitar a análise a diárias de até 1000, pois valores muito altos podem distorcer os gráficos.
adr_outliers = (df_clean["adr"] > 1000).sum()
print(f"Registros com ADR acima de 1000: {adr_outliers}")

df_clean = df_clean[df_clean["adr"] <= 1000]

# Verificando valores nulos após tratamento
print("\nValores nulos após tratamento:")
print(df_clean.isnull().sum().sort_values(ascending=False).head(15))

print("\nDimensão do dataset após tratamento:")
print(f"O dataset tratado possui {df_clean.shape[0]} linhas e {df_clean.shape[1]} colunas.")

print("\nTipos de dados após tratamento:")
print(df_clean[["children", "agent", "company", "reservation_status_date", "total_guests", "total_nights"]].dtypes)

# ============================================================
# 3. ANÁLISE EXPLORATÓRIA DE DADOS
# ============================================================

print("\n" + "=" * 60)
print("3. ANÁLISE EXPLORATÓRIA DE DADOS")
print("=" * 60)

GRAFICOS_DIR = Path("graficos")
GRAFICOS_DIR.mkdir(exist_ok=True)

# Ordem correta dos meses
ordem_meses = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# ------------------------------------------------------------
# 3.1 Taxa geral de cancelamento
# ------------------------------------------------------------

taxa_cancelamento_geral = df_clean["is_canceled"].mean() * 100

print(f"\nTaxa geral de cancelamento: {taxa_cancelamento_geral:.2f}%")

# ------------------------------------------------------------
# 3.2 Reservas e cancelamentos por tipo de hotel
# ------------------------------------------------------------

reservas_por_hotel = df_clean.groupby("hotel")["is_canceled"].agg(
    total_reservas="count",
    total_cancelamentos="sum",
    taxa_cancelamento="mean"
).reset_index()

reservas_por_hotel["taxa_cancelamento"] = reservas_por_hotel["taxa_cancelamento"] * 100

print("\nReservas e cancelamentos por tipo de hotel:")
print(reservas_por_hotel)

plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x="hotel", hue="is_canceled")
plt.title("Reservas e cancelamentos por tipo de hotel")
plt.xlabel("Tipo de hotel")
plt.ylabel("Quantidade de reservas")
plt.legend(title="Cancelada", labels=["Não", "Sim"])
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "cancelamentos_por_tipo_hotel.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=reservas_por_hotel, x="hotel", y="taxa_cancelamento")
plt.title("Taxa de cancelamento por tipo de hotel")
plt.xlabel("Tipo de hotel")
plt.ylabel("Taxa de cancelamento (%)")
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "taxa_cancelamento_por_hotel.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 3.3 Distribuição de reservas por mês
# ------------------------------------------------------------

reservas_por_mes = df_clean.groupby("arrival_date_month").size().reindex(ordem_meses)

print("\nQuantidade de reservas por mês:")
print(reservas_por_mes)

plt.figure(figsize=(12, 6))
sns.countplot(data=df_clean, x="arrival_date_month", order=ordem_meses)
plt.title("Quantidade de reservas por mês")
plt.xlabel("Mês de chegada")
plt.ylabel("Quantidade de reservas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "reservas_por_mes.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 3.4 Taxa de cancelamento por mês
# ------------------------------------------------------------

cancelamento_por_mes = df_clean.groupby("arrival_date_month")["is_canceled"].mean().reindex(ordem_meses) * 100

print("\nTaxa de cancelamento por mês:")
print(cancelamento_por_mes)

plt.figure(figsize=(12, 6))
sns.barplot(x=cancelamento_por_mes.index, y=cancelamento_por_mes.values)
plt.title("Taxa de cancelamento por mês")
plt.xlabel("Mês de chegada")
plt.ylabel("Taxa de cancelamento (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "taxa_cancelamento_por_mes.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 3.5 Lead time versus cancelamento
# ------------------------------------------------------------

df_clean["lead_time_group"] = pd.cut(
    df_clean["lead_time"],
    bins=[-1, 7, 30, 90, 180, 365, df_clean["lead_time"].max()],
    labels=[
        "0-7 dias",
        "8-30 dias",
        "31-90 dias",
        "91-180 dias",
        "181-365 dias",
        "Acima de 365 dias"
    ]
)

cancelamento_por_lead_time = df_clean.groupby("lead_time_group", observed=False)["is_canceled"].mean() * 100

print("\nTaxa de cancelamento por faixa de antecedência da reserva:")
print(cancelamento_por_lead_time)

plt.figure(figsize=(12, 6))
sns.barplot(x=cancelamento_por_lead_time.index, y=cancelamento_por_lead_time.values)
plt.title("Taxa de cancelamento por antecedência da reserva")
plt.xlabel("Faixa de lead time")
plt.ylabel("Taxa de cancelamento (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "lead_time_cancelamento.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 3.6 Tarifa diária média por status de cancelamento
# ------------------------------------------------------------

adr_por_cancelamento = df_clean.groupby("is_canceled")["adr"].mean().reset_index()
adr_por_cancelamento["status"] = adr_por_cancelamento["is_canceled"].map({
    0: "Não cancelada",
    1: "Cancelada"
})

print("\nTarifa diária média por status da reserva:")
print(adr_por_cancelamento[["status", "adr"]])

plt.figure(figsize=(8, 5))
sns.barplot(data=adr_por_cancelamento, x="status", y="adr")
plt.title("Tarifa diária média por status da reserva")
plt.xlabel("Status da reserva")
plt.ylabel("ADR - Tarifa diária média")
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "adr_por_cancelamento.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 3.7 Top 10 países com mais reservas
# ------------------------------------------------------------

top_paises = df_clean["country"].value_counts().head(10)

print("\nTop 10 países com mais reservas:")
print(top_paises)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_paises.values, y=top_paises.index)
plt.title("Top 10 países com mais reservas")
plt.xlabel("Quantidade de reservas")
plt.ylabel("País")
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "top_paises_reservas.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 3.8 Cancelamento por tipo de cliente
# ------------------------------------------------------------

cancelamento_por_cliente = df_clean.groupby("customer_type")["is_canceled"].mean().sort_values(ascending=False) * 100

print("\nTaxa de cancelamento por tipo de cliente:")
print(cancelamento_por_cliente)

plt.figure(figsize=(10, 5))
sns.barplot(x=cancelamento_por_cliente.index, y=cancelamento_por_cliente.values)
plt.title("Taxa de cancelamento por tipo de cliente")
plt.xlabel("Tipo de cliente")
plt.ylabel("Taxa de cancelamento (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "cancelamento_por_tipo_cliente.png", dpi=300)
plt.close()

print("\nGráficos gerados com sucesso na pasta 'graficos'.")

# ============================================================
# 4. MODELAGEM PREDITIVA
# ============================================================

print("\n" + "=" * 60)
print("4. MODELAGEM PREDITIVA")
print("=" * 60)

# Criando uma cópia específica para modelagem
df_model = df_clean.copy()

# ------------------------------------------------------------
# 4.1 Feature Engineering
# ------------------------------------------------------------

# Convertendo mês textual para número
mes_para_numero = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

df_model["arrival_month_number"] = df_model["arrival_date_month"].map(mes_para_numero)

# Criando variáveis derivadas
df_model["has_agent"] = np.where(df_model["agent"] > 0, 1, 0)
df_model["has_company"] = np.where(df_model["company"] > 0, 1, 0)
df_model["is_family"] = np.where((df_model["children"] + df_model["babies"]) > 0, 1, 0)
df_model["has_previous_cancellations"] = np.where(df_model["previous_cancellations"] > 0, 1, 0)

# Agrupando países para reduzir excesso de categorias
top_20_countries = df_model["country"].value_counts().head(20).index
df_model["country_grouped"] = np.where(
    df_model["country"].isin(top_20_countries),
    df_model["country"],
    "Other"
)

# ------------------------------------------------------------
# 4.2 Seleção das variáveis do modelo
# ------------------------------------------------------------

# Importante:
# reservation_status e reservation_status_date NÃO serão usadas,
# pois representam o estado final da reserva e causariam vazamento de informação.
# A ideia é prever o cancelamento a partir de dados disponíveis da reserva.

numeric_features = [
    "lead_time",
    "arrival_date_year",
    "arrival_month_number",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_guests",
    "total_nights",
    "has_agent",
    "has_company",
    "is_family",
    "has_previous_cancellations"
]

categorical_features = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "country_grouped"
]

features = numeric_features + categorical_features

X = df_model[features]
y = df_model["is_canceled"]

print("\nDistribuição da variável-alvo:")
print(y.value_counts())
print("\nDistribuição percentual da variável-alvo:")
print(y.value_counts(normalize=True) * 100)

# ------------------------------------------------------------
# 4.3 Separação entre treino e teste
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTamanho do conjunto de treino: {X_train.shape[0]} registros")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} registros")

# ------------------------------------------------------------
# 4.4 Pipeline de pré-processamento e modelo
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numericas", "passthrough", numeric_features)
    ]
)

modelo = RandomForestClassifier(
    n_estimators=80,
    max_depth=14,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessamento", preprocessor),
        ("modelo", modelo)
    ]
)

print("\nTreinando o modelo Random Forest...")
pipeline.fit(X_train, y_train)

# ------------------------------------------------------------
# 4.5 Avaliação do modelo
# ------------------------------------------------------------

y_pred = pipeline.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)
relatorio = classification_report(
    y_test,
    y_pred,
    target_names=["Não cancelada", "Cancelada"]
)

print(f"\nAcurácia do modelo: {acuracia:.4f}")

print("\nMatriz de confusão:")
print(matriz)

print("\nRelatório de classificação:")
print(relatorio)

# Salvando matriz de confusão como imagem
fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Não cancelada", "Cancelada"]
)
disp.plot(ax=ax, values_format="d")
plt.title("Matriz de Confusão - Modelo Random Forest")
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / "matriz_confusao.png", dpi=300)
plt.close()

# Salvando relatório do modelo em arquivo de texto
with open("relatorio_modelo.txt", "w", encoding="utf-8") as arquivo:
    arquivo.write("RELATÓRIO DO MODELO PREDITIVO\n")
    arquivo.write("=" * 60)
    arquivo.write("\n\n")
    arquivo.write(f"Acurácia do modelo: {acuracia:.4f}\n\n")
    arquivo.write("Matriz de confusão:\n")
    arquivo.write(str(matriz))
    arquivo.write("\n\nRelatório de classificação:\n")
    arquivo.write(relatorio)

print("\nMatriz de confusão salva em: graficos/matriz_confusao.png")
print("Relatório do modelo salvo em: relatorio_modelo.txt")