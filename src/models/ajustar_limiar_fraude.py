
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURACOES
# ============================================================

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_DADOS = RAIZ / "data" / "processed" / "sinistros_seguros_tratados.csv"
PASTA_ARTIFATOS = RAIZ / "artifacts"
PASTA_ARTIFATOS.mkdir(parents=True, exist_ok=True)

CAMINHO_RESULTADOS = PASTA_ARTIFATOS / "ajuste_limiares_validacao.csv"
CAMINHO_MODELO = PASTA_ARTIFATOS / "modelo_fraudes_limiar.joblib"
CAMINHO_INFO = PASTA_ARTIFATOS / "info_modelo_limiar.json"

ALVO = "fraude_confirmada"
RANDOM_STATE = 42

# Limiar escolhido exclusivamente com o conjunto de validacao.
# Em caso de empate no F1, preferimos o maior limiar.
LIMIARES = np.arange(0.05, 0.96, 0.05)

mlflow.set_tracking_uri(f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}")
mlflow.set_experiment("prevencao_fraude_seguro_limiar")


# ============================================================
# LEITURA E DIVISAO DOS DADOS
# ============================================================

df = pd.read_csv(CAMINHO_DADOS)
df = df.drop_duplicates().copy()

X = df.drop(columns=[ALVO, "id_sinistro"], errors="ignore")
y = df[ALVO].astype(int)

# 60% treino, 20% validacao, 20% teste.
X_treino, X_temp, y_treino, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_validacao, X_teste, y_validacao, y_teste = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

numericas = X_treino.select_dtypes(include=["number"]).columns.tolist()
categoricas = X_treino.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

transformador_numerico = Pipeline(
    steps=[
        ("imputador", SimpleImputer(strategy="median")),
    ]
)

transformador_categorico = Pipeline(
    steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessador = ColumnTransformer(
    transformers=[
        ("numericas", transformador_numerico, numericas),
        ("categoricas", transformador_categorico, categoricas),
    ]
)

modelo = Pipeline(
    steps=[
        ("preprocessador", preprocessador),
        (
            "classificador",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

print("=" * 65)
print("AJUSTE DE LIMIAR COM VALIDACAO INDEPENDENTE")
print("=" * 65)
print(f"Total de registros: {len(df)}")
print(f"Treino: {len(X_treino)}")
print(f"Validacao: {len(X_validacao)}")
print(f"Teste reservado: {len(X_teste)}")
print(f"Fraudes na validacao: {int(y_validacao.sum())}")
print(f"Fraudes no teste: {int(y_teste.sum())}")

# O modelo aprende somente com o conjunto de treino.
modelo.fit(X_treino, y_treino)

# Escolha do limiar usando exclusivamente validacao.
prob_validacao = modelo.predict_proba(X_validacao)[:, 1]

resultados = []

for limiar in LIMIARES:
    previsoes = (prob_validacao >= limiar).astype(int)

    resultados.append(
        {
            "limiar": round(float(limiar), 2),
            "precisao": precision_score(
                y_validacao, previsoes, zero_division=0
            ),
            "recall": recall_score(
                y_validacao, previsoes, zero_division=0
            ),
            "f1": f1_score(
                y_validacao, previsoes, zero_division=0
            ),
            "taxa_sinalizados": float(previsoes.mean()),
            "quantidade_sinalizados": int(previsoes.sum()),
        }
    )

df_resultados = pd.DataFrame(resultados)

# Maior F1; em caso de empate, maior limiar.
df_resultados = df_resultados.sort_values(
    ["f1", "limiar"],
    ascending=[False, False],
)

melhor = df_resultados.iloc[0]
limiar_escolhido = float(melhor["limiar"])

# Salva todos os resultados da validacao.
df_resultados.sort_values("limiar").to_csv(
    CAMINHO_RESULTADOS,
    index=False,
)

print("\nRESULTADOS DA VALIDACAO")
print(
    df_resultados.sort_values(
        "limiar"
    ).round(4).to_string(index=False)
)

print("\nLIMIAR ESCOLHIDO PELA VALIDACAO")
print(f"Limiar: {limiar_escolhido:.2f}")
print(f"Precisao: {melhor['precisao']:.4f}")
print(f"Recall: {melhor['recall']:.4f}")
print(f"F1: {melhor['f1']:.4f}")
print(f"Sinistros sinalizados: {int(melhor['quantidade_sinalizados'])}")


# ============================================================
# AVALIACAO FINAL NO TESTE, SEM NOVO AJUSTE
# ============================================================

prob_teste = modelo.predict_proba(X_teste)[:, 1]
previsoes_teste = (prob_teste >= limiar_escolhido).astype(int)

matriz = confusion_matrix(y_teste, previsoes_teste)

metricas_teste = {
    "roc_auc_teste": float(roc_auc_score(y_teste, prob_teste)),
    "pr_auc_teste": float(average_precision_score(y_teste, prob_teste)),
    "precisao_teste": float(
        precision_score(y_teste, previsoes_teste, zero_division=0)
    ),
    "recall_teste": float(
        recall_score(y_teste, previsoes_teste, zero_division=0)
    ),
    "f1_teste": float(
        f1_score(y_teste, previsoes_teste, zero_division=0)
    ),
    "taxa_sinalizados_teste": float(previsoes_teste.mean()),
    "limiar_escolhido_validacao": limiar_escolhido,
}

print("\nRESULTADOS FINAIS NO TESTE")
for nome, valor in metricas_teste.items():
    print(f"{nome}: {valor:.4f}")

print("\nMatriz de confusao (teste):")
print(matriz)

# Registra em um experimento novo no MLflow.
with mlflow.start_run(run_name="random_forest_limiar_validacao"):
    mlflow.log_param("modelo", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("min_samples_leaf", 20)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("estrategia_divisao", "60_treino_20_validacao_20_teste")
    mlflow.log_param("criterio_limiar", "maior_f1_na_validacao")
    mlflow.log_param("limiar_escolhido", limiar_escolhido)

    for nome, valor in metricas_teste.items():
        mlflow.log_metric(nome, valor)

    mlflow.log_artifact(str(CAMINHO_RESULTADOS))
    mlflow.sklearn.log_model(
    modelo,
    name="modelo_fraudes",
    skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
)

# Salva o modelo e os metadados em arquivos separados.
joblib.dump(modelo, CAMINHO_MODELO)

informacoes = {
    "modelo": "RandomForestClassifier",
    "limiar": limiar_escolhido,
    "criterio": "maior F1 na validacao",
    "divisao": {
        "treino": 0.60,
        "validacao": 0.20,
        "teste": 0.20,
    },
    "metricas_teste": metricas_teste,
    "matriz_confusao": matriz.tolist(),
}

with open(CAMINHO_INFO, "w", encoding="utf-8") as arquivo:
    json.dump(informacoes, arquivo, ensure_ascii=False, indent=4)

print("\nArquivos salvos:")
print(CAMINHO_RESULTADOS)
print(CAMINHO_MODELO)
print(CAMINHO_INFO)
print("\nExperimento registrado no MLflow.")
