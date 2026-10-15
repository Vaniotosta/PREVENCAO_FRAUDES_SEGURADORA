
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACAO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

CAMINHO_DADOS = (
    ROOT_DIR
    / "data"
    / "processed"
    / "sinistros_seguros_tratados.csv"
)

PASTA_MODELOS = ROOT_DIR / "artifacts"
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("prevencao_fraude_seguro_features")

RANDOM_STATE = 42
LIMIAR = 0.50


# ============================================================
# LEITURA E DIVISAO DOS DADOS
# ============================================================

df = pd.read_csv(CAMINHO_DADOS)
df = df.drop_duplicates().copy()

X = df.drop(columns=["fraude_confirmada", "id_sinistro"])
y = df["fraude_confirmada"].astype(int)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

colunas_numericas = X.select_dtypes(
    include=["number"]
).columns.tolist()

colunas_categoricas = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Pipeline numerico
pipeline_numerico = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
])

# Pipeline categorico
pipeline_categorico = Pipeline([
    ("imputador", SimpleImputer(strategy="most_frequent")),
    (
        "codificador",
        OneHotEncoder(handle_unknown="ignore"),
    ),
])

preprocessador = ColumnTransformer([
    ("numericas", pipeline_numerico, colunas_numericas),
    ("categoricas", pipeline_categorico, colunas_categoricas),
])

print("=" * 65)
print("MODELOS DE ARVORES COM VARIAVEIS ORIGINAIS")
print("=" * 65)
print("Registros:", len(df))
print("Treino:", len(X_treino))
print("Teste:", len(X_teste))
print("Fraudes no teste:", int(y_teste.sum()))
print("Variaveis numericas:", colunas_numericas)
print("Variaveis categoricas:", colunas_categoricas)
print("Limiar:", LIMIAR)


# ============================================================
# CINCO CONFIGURACOES
# ============================================================

experimentos = [
    (
        "features_arvore_prof3",
        DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "features_arvore_prof6",
        DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "features_rf_basica",
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
        "features_rf_profunda",
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
        "features_rf_regularizada",
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
# TREINAMENTO E AVALIACAO
# ============================================================

resultados = []

for nome, classificador in experimentos:
    print(f"\nTreinando: {nome}")

    modelo = Pipeline([
        ("preprocessador", preprocessador),
        ("classificador", classificador),
    ])

    with mlflow.start_run(run_name=nome):
        modelo.fit(X_treino, y_treino)

        probabilidades = modelo.predict_proba(X_teste)[:, 1]
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

        mlflow.log_param("modelo", nome)
        mlflow.log_param("limiar", LIMIAR)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_params({
            chave: str(valor)
            for chave, valor in classificador.get_params().items()
            if isinstance(valor, (str, int, float, bool))
            or valor is None
        })
        mlflow.log_metrics(metricas)

        caminho_modelo = PASTA_MODELOS / f"{nome}.joblib"
        joblib.dump(modelo, caminho_modelo)
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

        print("Metricas:", metricas)
        print("Matriz de confusao:")
        print(matriz)
        print("Modelo salvo:", caminho_modelo)


# ============================================================
# COMPARACAO FINAL
# ============================================================

tabela = pd.DataFrame(resultados).sort_values(
    "pr_auc_teste",
    ascending=False,
)

caminho_resultados = (
    PASTA_MODELOS / "comparacao_modelos_arvores_features.csv"
)
tabela.to_csv(caminho_resultados, index=False)

print("\n" + "=" * 65)
print("RESULTADO FINAL - VARIAVEIS ORIGINAIS")
print("=" * 65)
print(
    tabela.to_string(
        index=False,
        float_format=lambda valor: f"{valor:.4f}",
    )
)
print("\nTabela salva em:", caminho_resultados)
print("\nExperimentos concluidos.")
