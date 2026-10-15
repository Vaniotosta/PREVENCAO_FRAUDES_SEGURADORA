from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configurações
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

CAMINHO_DADOS = (
    ROOT_DIR
    / "data"
    / "processed"
    / "sinistros_seguros_tratados.csv"
)

PASTA_MODELOS = ROOT_DIR / "artifacts"
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

CAMINHO_MODELO = PASTA_MODELOS / "modelo_fraudes_balanceado.joblib"

LIMIAR = 0.55

# --------------------------------------------------
# 2. Carregar os dados
# --------------------------------------------------

df = pd.read_csv(CAMINHO_DADOS)
df = df.drop_duplicates().copy()

X = df.drop(columns=["fraude_confirmada", "id_sinistro"])
y = df["fraude_confirmada"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_dev, X_val, y_dev, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train,
)

# --------------------------------------------------
# 3. Criar os cinco fatores de risco
# --------------------------------------------------

def criar_fatores(dados, limite_valor):
    return pd.DataFrame(
        {
            "sinistros_3_ou_mais": (
                dados["sinistros_anteriores"] >= 3
            ).astype(int),
            "contratacao_recente": (
                dados["dias_desde_contratacao"] < 60
            ).astype(int),
            "sinistro_valor_alto": (
                dados["valor_sinistro"] > limite_valor
            ).astype(int),
            "comunicacao_tardia": (
                dados["dias_para_comunicar"] > 10
            ).astype(int),
            "ocorrencia_madrugada": (
                dados["hora_ocorrencia"] <= 4
            ).astype(int),
        }
    )


limite_valor = X_dev["valor_sinistro"].quantile(0.90)

X_dev_fatores = criar_fatores(X_dev, limite_valor)
X_val_fatores = criar_fatores(X_val, limite_valor)
X_test_fatores = criar_fatores(X_test, limite_valor)

# --------------------------------------------------
# 4. Treinar o modelo

# --------------------------------------------------
# ANALISE EXPLORATORIA DOS FATORES DE RISCO
# --------------------------------------------------

analise_fatores = X_dev_fatores.copy()
analise_fatores["fraude_confirmada"] = y_dev.values

print("\nFREQUENCIA DOS FATORES POR CLASSE")
print("=" * 60)

for fator in X_dev_fatores.columns:
    frequencias = analise_fatores.groupby(
        "fraude_confirmada"
    )[fator].mean()

    print(f"\nFator: {fator}")
    print(f"Sem fraude: {frequencias.loc[0]:.2%}")
    print(f"Com fraude: {frequencias.loc[1]:.2%}")

# --------------------------------------------------

modelo = Pipeline(
    steps=[
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

modelo.fit(X_dev_fatores, y_dev)

print("\nCOEFICIENTES APRENDIDOS PELO MODELO")
print("=" * 50)

coeficientes = modelo.named_steps["modelo"].coef_[0]

for fator, coeficiente in zip(X_dev_fatores.columns, coeficientes):
    print(f"{fator}: {coeficiente:.4f}")


# --------------------------------------------------
# 5. Avaliar na validação e no teste
# --------------------------------------------------

prob_val = modelo.predict_proba(X_val_fatores)[:, 1]
print("\nDISTRIBUICAO DAS PROBABILIDADES")
print("Mínima:", prob_val.min())
print("Mediana:", __import__("numpy").median(prob_val))
print("Máxima:", prob_val.max())
pred_val = (prob_val >= LIMIAR).astype(int)


print("\nCOMPARACAO DE LIMIARES NA VALIDACAO")
print("=" * 55)

for limiar_teste in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    pred_comparacao = (prob_val >= limiar_teste).astype(int)

    precisao = precision_score(
        y_val, pred_comparacao, zero_division=0
    )
    recall = recall_score(
        y_val, pred_comparacao, zero_division=0
    )
    f1 = f1_score(
        y_val, pred_comparacao, zero_division=0
    )

    print(
        f"Limiar: {limiar_teste:.2f} | "
        f"Precisao: {precisao:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"Sinalizados: {pred_comparacao.sum()}"
    )


prob_teste = modelo.predict_proba(X_test_fatores)[:, 1]
pred_teste = (prob_teste >= LIMIAR).astype(int)

metricas = {
    "roc_auc_teste": roc_auc_score(y_test, prob_teste),
    "pr_auc_teste": average_precision_score(y_test, prob_teste),
    "precisao_teste": precision_score(
        y_test, pred_teste, zero_division=0
    ),
    "recall_teste": recall_score(
        y_test, pred_teste, zero_division=0
    ),
    "f1_teste": f1_score(
        y_test, pred_teste, zero_division=0
    ),
    "roc_auc_validacao": roc_auc_score(y_val, prob_val),
    "pr_auc_validacao": average_precision_score(y_val, prob_val),
}

# --------------------------------------------------
# 6. Salvar o modelo com nome novo
# --------------------------------------------------

joblib.dump(modelo, CAMINHO_MODELO)

# --------------------------------------------------
# 7. Registrar uma nova execução no MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("prevencao_fraude_seguro")

with mlflow.start_run(run_name="retreinamento_logistica_fraudes"):
    mlflow.log_params(
        {
            "modelo": "Regressao Logistica",
            "limiar_classificacao": LIMIAR,
            "limite_valor_dev": float(limite_valor),
            "quantidade_features": 5,
            "origem_dados": "sinteticos",
            "random_state": 42,
            "tamanho_teste": 0.20,
            "tamanho_validacao_dentro_treino": 0.25,
        }
    )

    mlflow.log_metrics(metricas)

    mlflow.log_metric("casos_sinalizados_teste", int(pred_teste.sum()))
    mlflow.log_metric("fraudes_reais_teste", int(y_test.sum()))

    mlflow.log_artifact(str(CAMINHO_MODELO), artifact_path="modelo")

print("\nRETREINAMENTO CONCLUÍDO")
print("-" * 45)
print("Base utilizada:", CAMINHO_DADOS.name)
print("Registros:", len(df))
print("Desenvolvimento:", len(X_dev))
print("Validação:", len(X_val))
print("Teste:", len(X_test))
print("Limiar:", LIMIAR)
print("\nMétricas:")
for nome, valor in metricas.items():
    print(f"{nome}: {valor:.4f}")

print("\nMatriz de confusão do teste:")
print(confusion_matrix(y_test, pred_teste))
print("\nNovo modelo salvo em:", CAMINHO_MODELO)
print("Nova execução registrada no MLflow.")
