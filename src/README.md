# Prevenção à Fraude em Seguros
### Machine Learning | Python | Scikit-learn | MLflow | Streamlit

## Sobre o projeto

Este projeto explora técnicas de Ciência de Dados e Machine Learning para investigar a identificação de possíveis fraudes em sinistros de seguros.

O trabalho contempla análise exploratória, preparação dos dados, engenharia de atributos, treinamento e comparação de modelos de classificação, avaliação de métricas e análise crítica dos resultados.

A solução está sendo estruturada para incluir uma aplicação interativa com Streamlit e rastreamento de experimentos com MLflow.

**Aviso:** este projeto utiliza dados sintéticos, gerados artificialmente para fins educacionais. Os resultados não representam o desempenho esperado em uma seguradora real, e o modelo não está validado para decisões operacionais.

## Objetivos

- Investigar padrões associados a possíveis fraudes.
- Preparar e analisar dados tabulares.
- Construir indicadores para apoiar a classificação.
- Comparar algoritmos de Machine Learning.
- Avaliar o impacto do desbalanceamento das classes.
- Analisar precisão, recall, F1-score, ROC-AUC e PR-AUC.
- Investigar o efeito dos limiares de classificação.
- Desenvolver uma estrutura reproduzível de experimentação.
- Apresentar os resultados por meio de uma aplicação interativa.

## Dados utilizados

A base contém 20.000 registros únicos de sinistros após a remoção de duplicatas e 14 colunas, incluindo a variável-alvo.

As informações disponíveis incluem:

- Idade do segurado.
- Tempo de relacionamento com a seguradora.
- Tipo de seguro.
- Região.
- Valor do sinistro.
- Valor da apólice.
- Quantidade de sinistros anteriores.
- Dias desde a contratação.
- Tipo de ocorrência.
- Canal de comunicação do sinistro.
- Hora da ocorrência.
- Dias para comunicar o sinistro.

### Variável-alvo

`fraude_confirmada`

A variável indica a classe de cada registro: sinistro classificado como fraude ou não fraude.

A base apresenta aproximadamente 4,52% de registros classificados como fraude, caracterizando um problema de classificação com classes desbalanceadas.

### Limitações da base

Os registros são artificiais. A variável-alvo foi gerada probabilisticamente a partir de indicadores simulados, com uma componente aleatória.

Essa característica limita a capacidade dos algoritmos de aprender relações fortes entre as variáveis explicativas e a classe de fraude.

Os resultados devem ser interpretados como experimentos educacionais, não como evidência de desempenho em dados reais.

## Metodologia

O projeto foi desenvolvido seguindo etapas de um fluxo de trabalho de Ciência de Dados.

### 1. Inspeção e preparação dos dados

- Leitura da base.
- Verificação de duplicatas.
- Investigação de valores ausentes e inconsistências.
- Preparação das variáveis para análise e modelagem.

### 2. Análise exploratória

- Análise da distribuição da variável-alvo.
- Inspeção das variáveis numéricas e categóricas.
- Investigação de possíveis indicadores associados à classe de fraude.

### 3. Separação dos dados

Os dados foram divididos em conjuntos de desenvolvimento, validação e teste.

A validação foi utilizada para comparar modelos e analisar limiares de decisão. O teste foi reservado para a avaliação final do modelo selecionado.

### 4. Engenharia de atributos

Foram construídos cinco indicadores binários:

- `sinistros_3_ou_mais`: três ou mais sinistros anteriores.
- `contratacao_recente`: contratação realizada há menos de 60 dias.
- `sinistro_valor_alto`: valor do sinistro acima do percentil 90 calculado no conjunto de desenvolvimento.
- `comunicacao_tardia`: comunicação realizada após mais de 10 dias.
- `ocorrencia_madrugada`: ocorrência registrada até as 4 horas.

Esses indicadores representam hipóteses simuladas e não devem ser interpretados como regras comprovadas de fraude no mercado de seguros.

### 5. Treinamento e comparação

Foram avaliados os seguintes modelos:

- Regressão Logística com cinco indicadores.
- Random Forest com cinco indicadores.
- Gradient Boosting com cinco indicadores.
- Regressão Logística com as variáveis originais.

### 6. Avaliação

Foram utilizadas métricas de classificação adequadas à análise de classes desbalanceadas:

- ROC-AUC.
- PR-AUC.
- Precisão.
- Recall.
- F1-score.
- Matriz de confusão.

Também foram investigados diferentes limiares de classificação para compreender o equilíbrio entre fraudes identificadas e falsos alertas.

## Comparação dos modelos

Resultados obtidos no conjunto de validação:

| Modelo | ROC-AUC | PR-AUC |
|---|---:|---:|
| Regressão Logística — cinco indicadores | 0,5870 | 0,0620 |
| Random Forest — cinco indicadores | 0,5785 | 0,0586 |
| Gradient Boosting — cinco indicadores | 0,5751 | 0,0591 |
| Regressão Logística — variáveis originais | 0,5198 | 0,0539 |

A Regressão Logística com cinco indicadores apresentou os melhores resultados entre os modelos comparados nessas métricas de validação.

Por esse motivo, foi selecionada como modelo de referência para a avaliação final.

A diferença entre os modelos foi pequena, e os resultados indicam capacidade limitada de discriminação.

## Avaliação final no conjunto de teste

A avaliação final utilizou o limiar de classificação de 0,06, escolhido durante a etapa de validação.

| Métrica | Resultado |
|---|---:|
| Total de sinistros | 4.000 |
| Fraudes reais | 181 |
| Casos sinalizados | 429 |
| Fraudes identificadas | 31 |
| Falsos alertas | 398 |
| Fraudes não identificadas | 150 |
| Casos corretamente não sinalizados | 3.421 |
| ROC-AUC | 0,5987 |
| PR-AUC | 0,0638 |
| Precisão | 0,0723 |
| Recall | 0,1713 |
| F1-score | 0,1016 |

### Interpretação dos resultados

O modelo identificou 31 das 181 fraudes existentes no conjunto de teste, correspondendo a um recall de aproximadamente 17,13%.

A precisão de 7,23% indica que uma pequena parcela dos sinistros sinalizados correspondia a fraudes. Foram registrados 398 falsos alertas e 150 fraudes não identificadas.

A ROC-AUC de 0,5987 demonstra capacidade limitada de discriminação entre as classes.

A PR-AUC de 0,0638 ficou acima da prevalência de fraudes no conjunto de teste, de aproximadamente 4,53%. Entretanto, essa diferença é modesta e não demonstra, por si só, utilidade operacional.

**Conclusão da avaliação:** a Regressão Logística foi o melhor modelo entre os algoritmos comparados na validação, mas seu desempenho final é insuficiente para recomendar uso operacional.

## Aplicação interativa — Streamlit

O projeto prevê uma interface interativa desenvolvida com Streamlit para apresentar os resultados da análise e facilitar a demonstração da solução.

A aplicação poderá incluir:

- Visão geral da base de dados.
- Distribuição das classes.
- Apresentação das métricas do modelo.
- Visualização da matriz de confusão.
- Demonstração das previsões e do limiar de classificação.
- Explicação das limitações e dos riscos associados aos resultados.

A implementação e as funcionalidades disponíveis deverão corresponder ao código efetivamente publicado no repositório.

As previsões são experimentais e não devem ser utilizadas para aprovar, rejeitar ou investigar sinistros reais sem validação apropriada.

## Rastreamento de experimentos — MLflow

O MLflow será utilizado para organizar e rastrear os experimentos de Machine Learning à medida que a integração for implementada.

A configuração poderá registrar:

- Nome e tipo do modelo.
- Parâmetros de treinamento.
- Métricas de validação.
- Métricas da avaliação final, identificadas separadamente.
- Artefatos relevantes.
- Modelo treinado e sua versão, quando configurado.

O objetivo é melhorar a rastreabilidade, facilitar a comparação entre experimentos e apoiar a reprodução dos resultados.

A integração deverá ser considerada concluída somente após a implementação e a verificação dos registros no MLflow.

## Tecnologias

- **Python:** linguagem de programação.
- **Pandas:** manipulação de dados.
- **NumPy:** operações numéricas.
- **Scikit-learn:** treinamento, preparação e avaliação dos modelos.
- **Jupyter Notebook:** experimentação e análise.
- **Streamlit:** construção da interface interativa.
- **MLflow:** rastreamento de experimentos e métricas.
- **Git e GitHub:** versionamento e publicação do projeto.

## Como executar

As instruções abaixo pressupõem que as dependências e os arquivos necessários estejam disponíveis no repositório.

### 1. Clonar o repositório

```bash
git clone URL_DO_REPOSITORIO
cd prevencao_fraude_seguro
```

Substitua `URL_DO_REPOSITORIO` pela URL real do repositório no GitHub.

### 2. Criar um ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar as dependências

Se o arquivo `requirements.txt` estiver presente:

```bash
pip install -r requirements.txt
```

### 4. Executar o notebook

Abra o Jupyter Notebook ou o VS Code e execute o notebook de modelagem disponível em `notebooks/02_modelagem_fraudes.ipynb`.

### 5. Executar a aplicação Streamlit

Após a criação do arquivo de entrada da aplicação e a configuração das dependências:

```bash
streamlit run app.py
```

O comando pressupõe que `app.py` esteja na raiz do projeto. Ajuste o caminho conforme a estrutura real do repositório.

### 6. Acompanhar experimentos no MLflow

Após a implementação do rastreamento:

```bash
mlflow ui
```

O comando inicia a interface local do MLflow, desde que a instalação e a configuração de armazenamento estejam corretas.

## Limitações e próximos passos

As principais limitações identificadas são:

- Uso de dados sintéticos.
- Relações limitadas entre os indicadores e a variável-alvo.
- Desbalanceamento entre as classes.
- Baixa precisão e recall na avaliação final.
- Quantidade elevada de falsos alertas.
- Ausência de validação em uma base real e independente de seguradora.
- Necessidade de avaliar custos operacionais e consequências dos erros.

Possíveis evoluções:

1. Implementar o rastreamento de experimentos com MLflow.
2. Desenvolver e testar a interface Streamlit.
3. Melhorar a organização do código e a reprodutibilidade.
4. Avaliar os custos de falsos positivos e falsos negativos.
5. Investigar estratégias adicionais de engenharia de atributos.
6. Validar a metodologia em dados reais, autorizados e protegidos, caso estejam disponíveis.
7. Avaliar estabilidade e monitoramento antes de qualquer implantação real.

## Conclusão

Este projeto demonstra a aplicação de etapas fundamentais de Ciência de Dados a um problema de classificação desbalanceada: preparação dos dados, engenharia de atributos, comparação de algoritmos, avaliação de métricas e interpretação crítica dos resultados.

O principal aprendizado é que a escolha de um algoritmo não garante bons resultados quando o sinal preditivo disponível é limitado.

A documentação das métricas e das limitações faz parte do processo de desenvolvimento responsável de soluções de Machine Learning.

**Status:** modelo experimental avaliado; integração com Streamlit e MLflow a ser concluída e verificada conforme o desenvolvimento.

**Finalidade:** educacional e demonstração de competências técnicas. Não é um sistema de detecção de fraudes pronto para produção.
