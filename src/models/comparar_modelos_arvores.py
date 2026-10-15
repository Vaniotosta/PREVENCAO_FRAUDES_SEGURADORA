
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURACAO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
CAMINHO_DADOS = (
    ROOT_DIR / "data" / "processed"
    / "sinistros_seguros_tratados.csv"
)

PASTA_MODELOS = ROOT_DIR / "artifacts"
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}")
mlflow.set_experiment("prevencao_fraude_seguro_arvores")

RANDOM_STATE = 42
LIMIAR = 0.50


# ============================================================
# PREPARACAO DOS DADOS
# ============================================================

df = pd.read_csv(CAMINHO_DADOS)
df = df.drop_duplicates().copy()

X_original = df.drop(
    columns=["fraude_confirmada", "id_sinistro"]
)
y = df["fraude_confirmada"].astype(int)

# Mesmo conjunto de teste para os cinco modelos
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_original,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Limite calculado somente com o conjunto de treino
limite_valor = X_treino["valor_sinistro"].quantile(0.90)


def criar_fatores(dados, limite):
    return pd.DataFrame({
        "sinistros_3_ou_mais":
            (dados["sinistros_anteriores"] >= 3).astype(int),
        "contratacao_recente":
            (dados["dias_desde_contratacao"] < 60).astype(int),
        "sinistro_valor_alto":
            (dados["valor_sinistro"] > limite).astype(int),
        "comunicacao_tardia":
            (dados["dias_para_comunicar"] > 10).astype(int),
        "ocorrencia_madrugada":
            (dados["hora_ocorrencia"] <= 4).astype(int),
    })


X_treino_fatores = criar_fatores(X_treino, limite_valor)
X_teste_fatores = criar_fatores(X_teste, limite_valor)

# ============================================================
# CINCO CONFIGURACOES
# ============================================================

experimentos = [
    (
        "arvore_prof_max3",
        DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "arvore_prof6",
        DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "random_forest_basica",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
    (
        "random_forest_profunda",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
    (
        "random_forest_regularizada",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=50,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
]


# ============================================================
# TREINAMENTO E REGISTRO NO MLFLOW
# ============================================================

resultados = []

print("=" * 65)
print("COMPARACAO DE MODELOS DE ARVORES PARA DETECCAO DE FRAUDES")
print("=" * 65)
print(f"Registros: {len(df)}")
print(f"Treino: {len(X_treino)}")
print(f"Teste: {len(X_teste)}")
print(f"Fraudes no teste: {int(y_teste.sum())}")
print(f"Limiar inicial: {LIMIAR:.2f}")

for nome, classificador in experimentos:
    print(f"\nTreinando: {nome}")

    pipeline = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("modelo", classificador),
    ])

    with mlflow.start_run(run_name=nome):
        pipeline.fit(X_treino_fatores, y_treino)

        probabilidades = pipeline.predict_proba(
            X_teste_fatores
        )[:, 1]

        previsoes = (probabilidades >= LIMIAR).astype(int)

        metricas = {
            "roc_auc_teste": roc_auc_score(
                y_teste, probabilidades
            ),
            "pr_auc_teste": average_precision_score(
                y_teste, probabilidades
            ),
            "precisao_teste": precision_score(
                y_teste, previsoes, zero_division=0
            ),
            "recall_teste": recall_score(
                y_teste, previsoes, zero_division=0
            ),
            "f1_teste": f1_score(
                y_teste, previsoes, zero_division=0
            ),
            "taxa_sinalizados": float(previsoes.mean()),
        }

        parametros = classificador.get_params()

        mlflow.log_param("modelo", nome)
        mlflow.log_param("limiar", LIMIAR)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_params({
            chave: str(valor)
            for chave, valor in parametros.items()
            if valor is None or isinstance(
                valor, (str, int, float, bool)
            )
        })
        mlflow.log_metrics(metricas)

        caminho_modelo = PASTA_MODELOS / f"{nome}.joblib"
        joblib.dump(pipeline, caminho_modelo)
        mlflow.log_artifact(
            str(caminho_modelo),
            artifact_path="modelo",
        )

        matriz = confusion_matrix(
            y_teste, previsoes, labels=[0, 1]
        )

        resultados.append({
            "modelo": nome,
            **metricas,
            "verdadeiros_negativos": int(matriz[0, 0]),
            "falsos_positivos": int(matriz[0, 1]),
            "falsos_negativos": int(matriz[1, 0]),
            "verdadeiros_positivos": int(matriz[1, 1]),
        })

        print("Métricas:", metricas)
        print("Matriz de confusão:")
        print(matriz)
        print(f"Modelo salvo: {caminho_modelo}")

# ============================================================
# COMPARACAO FINAL
# ============================================================

tabela = pd.DataFrame(resultados).sort_values(
    "pr_auc_teste", ascending=False
)

caminho_resultados = (
    ROOT_DIR / "artifacts" / "comparacao_modelos_arvores.csv"
)
tabela.to_csv(caminho_resultados, index=False)

print("\n" + "=" * 65)
print("RESULTADO FINAL - ORDENADO PELA PR-AUC")
print("=" * 65)
print(tabela.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nTabela salva em: {caminho_resultados}")
print("\nCinco experimentos concluidos.")
