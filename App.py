
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuração
# --------------------------------------------------
st.set_page_config(
    page_title="Prevenção à Fraude em Seguros",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_FILE = BASE_DIR / "data" / "processed" / "sinistros_seguros_tratados.csv"
MODEL_FILE = ARTIFACTS_DIR / "modelo_fraudes.joblib"
METADATA_FILE = ARTIFACTS_DIR / "metadados_modelo.json"

st.title("🛡️ Prevenção à Fraude em Seguros")
st.caption("Projeto de portfólio de Ciência de Dados | Modelo experimental")

st.warning(
    "Este projeto utiliza dados sintéticos. O modelo é experimental, "
    "não está validado para uso real e não deve ser usado isoladamente "
    "para negar indenizações ou tomar decisões sobre segurados."
)

# --------------------------------------------------
# Verificação dos arquivos
# --------------------------------------------------
if not MODEL_FILE.exists() or not METADATA_FILE.exists():
    st.error("Arquivos do modelo não encontrados na pasta artifacts.")
    st.stop()

with open(METADATA_FILE, "r", encoding="utf-8") as arquivo:
    metadata = json.load(arquivo)

model = joblib.load(MODEL_FILE)

# --------------------------------------------------
# Visão geral dos dados
# --------------------------------------------------
st.header("1. Visão geral da base")

if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE)

    col1, col2, col3 = st.columns(3)

    col1.metric("Sinistros na base", f"{len(df):,}".replace(",", "."))
    col2.metric("Variáveis", len(df.columns))

    if "fraude_confirmada" in df.columns:
        taxa = df["fraude_confirmada"].mean() * 100
        col3.metric("Fraudes na base", f"{taxa:.2f}%")

    st.caption(
        "A proporção de fraudes descreve esta base sintética; "
        "não representa necessariamente a realidade do mercado."
    )
else:
    st.info("A base tratada não foi encontrada. A demonstração do modelo continua disponível.")

# --------------------------------------------------
# Resultados da avaliação final
# --------------------------------------------------
st.header("2. Avaliação do modelo")

st.write("Modelo selecionado: **Regressão Logística com cinco indicadores derivados**.")

metricas = st.columns(5)
metricas[0].metric("ROC-AUC", f"{metadata['roc_auc_teste']:.4f}")
metricas[1].metric("PR-AUC", f"{metadata['pr_auc_teste']:.4f}")
metricas[2].metric("Precisão", f"{metadata['precisao_teste']:.4f}")
metricas[3].metric("Recall", f"{metadata['recall_teste']:.4f}")
metricas[4].metric("F1-score", f"{metadata['f1_teste']:.4f}")

st.write(
    "Essas métricas vêm da avaliação final registrada no projeto. "
    "Elas não são recalculadas pelo aplicativo."
)

with st.expander("Entender as métricas"):
    st.markdown(
        """
        - **ROC-AUC:** capacidade de ordenar casos de fraude acima dos demais.
        - **PR-AUC:** equilíbrio entre precisão e recall, útil em classes desbalanceadas.
        - **Precisão:** proporção dos casos sinalizados que eram fraudes.
        - **Recall:** proporção das fraudes reais que foram identificadas.
        - **F1-score:** média harmônica entre precisão e recall.
        """
    )

# --------------------------------------------------
# Demonstração de análise de risco
# --------------------------------------------------
st.header("3. Demonstração interativa")

st.write(
    "Informe as características de um sinistro fictício para demonstrar "
    "como o modelo produz uma estimativa. Os campos abaixo não identificam pessoas."
)

with st.form("formulario_sinistro"):
    c1, c2 = st.columns(2)

    with c1:
        sinistros_anteriores = st.number_input(
            "Sinistros anteriores",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
        )
        dias_desde_contratacao = st.number_input(
            "Dias desde a contratação",
            min_value=0,
            max_value=10000,
            value=180,
            step=1,
        )
        valor_sinistro = st.number_input(
            "Valor do sinistro (R$)",
            min_value=0.0,
            value=3000.0,
            step=500.0,
        )

    with c2:
        dias_para_comunicar = st.number_input(
            "Dias para comunicar o sinistro",
            min_value=0,
            max_value=365,
            value=3,
            step=1,
        )
        hora_ocorrencia = st.slider(
            "Hora da ocorrência",
            min_value=0,
            max_value=23,
            value=14,
        )

    enviar = st.form_submit_button("Analisar sinistro fictício")

if enviar:
    limite_valor = float(metadata["limite_valor_dev"])

    fatores = pd.DataFrame(
        [{
            "sinistros_3_ou_mais": int(sinistros_anteriores >= 3),
            "contratacao_recente": int(dias_desde_contratacao < 60),
            "sinistro_valor_alto": int(valor_sinistro > limite_valor),
            "comunicacao_tardia": int(dias_para_comunicar > 10),
            "ocorrencia_madrugada": int(hora_ocorrencia <= 4),
        }]
    )

    probabilidade = float(model.predict_proba(fatores)[0, 1])
    limiar = float(metadata["limiar_classificacao"])

    st.subheader("Resultado da demonstração")
    st.metric("Probabilidade estimada pelo modelo", f"{probabilidade:.2%}")

    if probabilidade >= limiar:
        st.warning(
            "O modelo sinalizou este caso para análise adicional. "
            "Isso não confirma a existência de fraude."
        )
    else:
        st.info(
            "O modelo não sinalizou este caso pelo limiar escolhido. "
            "Isso não garante que não exista fraude."
        )

    st.caption(
        f"Limiar experimental utilizado: {limiar:.0%}. "
        "A estimativa depende dos padrões presentes nos dados sintéticos."
    )

# --------------------------------------------------
# Limitações
# --------------------------------------------------
st.header("4. Limitações e próximos passos")

st.markdown(
    """
    - A base é sintética e não comprova desempenho em sinistros reais.
    - As métricas obtidas são modestas; há risco de falsos alertas e fraudes não identificadas.
    - A saída deve servir apenas como demonstração educacional.
    - Próximos passos: rastreabilidade com MLflow, testes automatizados,
      monitoramento e avaliação com dados representativos e autorizados.
    """
)

st.caption("Portfólio educacional de Ciência de Dados.")
