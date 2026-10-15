# Prevenção de Fraudes em Seguros

🌐 **Acesse a aplicação online:** [Prevenção de Fraudes em Seguros — Streamlit](https://prevencaofraudesseguradora-izcqgczwj7ph3mbzbejydz.streamlit.app/)

Projeto de Ciência de Dados que explora dados de sinistros e demonstra como modelos de Machine Learning podem apoiar a identificação de ocorrências com maior risco de fraude.

## Objetivo

Desenvolver um fluxo experimental de análise de sinistros, desde a preparação dos dados até o treinamento, a avaliação dos modelos e a disponibilização de uma aplicação interativa.

## Tecnologias utilizadas

- Python
- Pandas e NumPy
- Scikit-learn
- Streamlit
- MLflow
- Matplotlib e Seaborn
- Joblib

## Modelo principal

O aplicativo utiliza um modelo de **Regressão Logística**. O projeto também inclui experimentos com árvores de decisão, Random Forest e ajuste de limiares de classificação.

### Métricas registradas no conjunto de teste

| Métrica | Resultado |
|---|---:|
| ROC-AUC | 0,5987 |
| PR-AUC | 0,0638 |
| Precisão | 0,0723 |
| Recall | 0,1713 |
| F1-score | 0,1016 |

Os resultados indicam desempenho limitado e oportunidades de melhoria. Em problemas de fraude, é importante considerar o desbalanceamento das classes e o equilíbrio entre a identificação de ocorrências suspeitas e os falsos positivos.

## Etapas desenvolvidas

- Preparação e tratamento dos dados.
- Treinamento e comparação de modelos.
- Avaliação por métricas de classificação.
- Experimentos com diferentes limiares.
- Persistência do modelo e dos metadados.
- Disponibilização de uma aplicação interativa com Streamlit.
- Acompanhamento de experimentos com MLflow.

## Estrutura do projeto

- `App.py` — aplicação interativa.
- `artifacts/` — modelo principal, metadados e resultados experimentais.
- `data/processed/` — dados tratados.
- `src/models/` — scripts de treinamento e comparação.
- `requirements.txt` — dependências Python.

## Como executar localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/Vaniotosta/PREVENCAO_FRAUDES_SEGURADORA.git
cd PREVENCAO_FRAUDES_SEGURADORA
```

### 2. Criar e ativar o ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Executar a aplicação

```bash
streamlit run App.py
```

## Dados e limitações

Os dados utilizados são sintéticos e destinados a estudo e demonstração. As métricas não comprovam a eficácia do modelo em operações reais de seguros.

Antes de qualquer uso operacional, seria necessária validação com dados reais autorizados, análise de vieses, definição de critérios de decisão e monitoramento contínuo. As previsões devem ser tratadas como apoio à análise, e não como prova definitiva de fraude.

## Sobre o projeto

Projeto de portfólio em Ciência de Dados, com foco em Machine Learning, avaliação de modelos e disponibilização de soluções analíticas.

**Aplicação online:** [Acessar no Streamlit](https://prevencaofraudesseguradora-izcqgczwj7ph3mbzbejydz.streamlit.app/)