# Guia Técnico — Projeto de Prevenção de Fraudes em Seguradora

## 1. Objetivo do projeto

Este projeto de Ciência de Dados tem como objetivo desenvolver um modelo de classificação capaz de identificar sinistros de seguros com maior risco de fraude.

O modelo analisa características de cada sinistro e produz uma previsão que pode auxiliar uma equipe de investigação a priorizar os casos que merecem análise humana.

**Importante:** a base utilizada é sintética. Os resultados são experimentais e não comprovam desempenho em operações reais de seguradoras.

## 2. Fluxo de trabalho

O projeto segue etapas semelhantes às utilizadas em projetos de Machine Learning:

1. Preparação e tratamento dos dados.
2. Análise exploratória e compreensão do problema.
3. Criação e seleção de atributos.
4. Separação dos dados para treinamento e teste.
5. Treinamento de modelos.
6. Avaliação com métricas de classificação.
7. Ajuste do limiar de decisão.
8. Salvamento do modelo e dos metadados.
9. Criação de uma interface com Streamlit.
10. Versionamento do código e publicação no GitHub.

Esse fluxo se relaciona com o CRISP-DM, metodologia que organiza projetos de mineração de dados em etapas como entendimento do negócio, entendimento dos dados, preparação, modelagem, avaliação e implantação.

## 3. Definição do problema

A tarefa é uma **classificação binária**.

O modelo procura distinguir duas classes:

- Classe positiva: sinistro identificado como suspeito de fraude, conforme o rótulo utilizado na base.
- Classe negativa: sinistro classificado como não fraudulento.

O modelo aprende padrões a partir de exemplos históricos disponíveis no conjunto de treinamento.

Uma previsão não é uma confirmação de fraude. Em uma aplicação real, os casos sinalizados precisariam ser analisados por profissionais, com controles de qualidade, privacidade e prevenção de decisões injustas.

## 4. Engenharia de atributos

Engenharia de atributos (*feature engineering*) é o processo de transformar informações disponíveis em variáveis que possam ajudar o modelo a identificar padrões.

Neste projeto, foram utilizadas variáveis indicativas de comportamentos que podem merecer investigação, como:

- `sinistros_3_ou_mais`: indicador relacionado à quantidade de sinistros.
- `contratacao_recente`: indicador de contratação recente.
- `sinistro_valor_alto`: indicador de valor elevado do sinistro.
- `comunicacao_tardia`: indicador de comunicação tardia.
- `ocorrencia_madrugada`: indicador relacionado ao horário da ocorrência.

Esses atributos representam hipóteses de trabalho. Nenhuma dessas características, isoladamente, prova fraude.

Em um projeto real, seria necessário documentar a origem das variáveis, validar sua qualidade e verificar se elas estariam disponíveis no momento em que a previsão fosse realizada.

### Prevenção de vazamento de dados

O vazamento de dados (*data leakage*) acontece quando o modelo utiliza informações que não estariam disponíveis no momento da previsão ou que revelam indiretamente a resposta.

Para evitar esse problema, é importante separar treino e teste corretamente e garantir que as variáveis sejam calculadas apenas com informações legítimas e disponíveis naquele momento.

## 5. Desbalanceamento das classes

Em problemas de fraude, é comum haver muito mais casos legítimos do que casos fraudulentos.

Quando uma classe é rara, a acurácia pode ser enganosa. Um modelo que classifique todos os registros como legítimos pode alcançar uma acurácia alta e, ainda assim, não identificar nenhuma fraude.

Por isso, a avaliação deve considerar métricas específicas, como precisão, recall, F1 e PR-AUC.

Também é importante separar os dados de treinamento, validação e teste de forma adequada. Se os registros tiverem dependência temporal ou representarem os mesmos clientes, uma divisão aleatória pode produzir uma avaliação excessivamente otimista.

## 6. Modelos de Machine Learning

### 6.1 Regressão Logística

A Regressão Logística é um algoritmo utilizado em classificação binária. Apesar do nome, ela estima a probabilidade de uma observação pertencer à classe positiva.

O resultado normalmente fica entre zero e um. Um limiar de decisão transforma essa probabilidade em uma classe prevista.

Vantagens:

- É relativamente simples de interpretar.
- Permite analisar a direção e a magnitude dos coeficientes.
- Serve como modelo de referência para comparação.
- Pode ser eficiente em problemas com relações aproximadamente lineares no espaço dos atributos.

Limitações:

- Pode ter dificuldade em representar relações complexas sem transformações ou interações entre variáveis.
- Pode exigir tratamento adequado de escalas, categorias e multicolinearidade.
- Suas probabilidades precisam ser avaliadas quanto à calibração.

### 6.2 Árvore de Decisão

Uma Árvore de Decisão divide os dados por meio de regras sucessivas.

Por exemplo, uma árvore pode aprender que determinada combinação de características conduz a uma previsão de maior risco.

Vantagens:

- Pode representar relações não lineares.
- As regras podem ser relativamente fáceis de explicar.
- Não exige necessariamente padronização das variáveis numéricas.

Limitações:

- Árvores profundas podem se ajustar demais aos dados de treinamento (*overfitting*).
- Pequenas alterações nos dados podem gerar estruturas diferentes.
- Uma árvore complexa pode ser difícil de explicar.

Parâmetros importantes incluem `max_depth`, `min_samples_split` e `min_samples_leaf`.

### 6.3 Random Forest

Random Forest combina várias árvores de decisão. Cada árvore é treinada com variações dos dados e das variáveis consideradas, e as previsões são agregadas.

A ideia é reduzir a dependência de uma única árvore e obter um modelo mais robusto.

Parâmetros relevantes:

- `n_estimators`: quantidade de árvores.
- `max_depth`: profundidade máxima das árvores.
- `min_samples_leaf`: quantidade mínima de observações em uma folha.
- `max_features`: quantidade de atributos considerada nas divisões.
- `class_weight`: permite atribuir pesos diferentes às classes, quando configurado.

Um modelo mais complexo não é necessariamente melhor. A comparação deve ser feita em dados que não foram utilizados para ajustar o modelo.

## 7. Treinamento, validação e teste

O conjunto de treinamento é usado para aprender os parâmetros do modelo.

O conjunto de validação ajuda a comparar alternativas, escolher hiperparâmetros e ajustar decisões, como o limiar de classificação.

O conjunto de teste deve ser reservado para a avaliação final. Ele não deve orientar repetidamente as escolhas do projeto, pois isso compromete a independência da avaliação.

Quando são utilizadas validação cruzada ou busca de hiperparâmetros, todas as etapas de pré-processamento que aprendem informações dos dados devem ser ajustadas dentro de cada divisão de treinamento.

## 8. Métricas de classificação

### 8.1 Matriz de confusão

A matriz de confusão compara as classes reais com as classes previstas.

- Verdadeiro positivo (VP): caso positivo identificado corretamente.
- Verdadeiro negativo (VN): caso negativo identificado corretamente.
- Falso positivo (FP): caso negativo sinalizado incorretamente.
- Falso negativo (FN): caso positivo que o modelo não identificou.

Em prevenção a fraudes, falsos positivos podem gerar investigações desnecessárias, enquanto falsos negativos podem representar fraudes não detectadas.

### 8.2 Precisão (*Precision*)

A precisão mede a proporção de previsões positivas que realmente são positivas:

Precisão = VP / (VP + FP)

Uma precisão baixa significa que muitos casos sinalizados não pertencem à classe positiva.

### 8.3 Revocação (*Recall*)

O recall mede a proporção de casos positivos reais identificados pelo modelo:

Recall = VP / (VP + FN)

Um recall baixo indica que muitos casos positivos não foram identificados.

### 8.4 F1-score

O F1 combina precisão e recall por meio da média harmônica:

F1 = 2 × (Precisão × Recall) / (Precisão + Recall)

É útil quando se deseja equilibrar precisão e recall, mas não substitui uma análise dos custos reais de cada tipo de erro.

### 8.5 ROC-AUC

A ROC-AUC avalia a capacidade de ordenação do modelo entre exemplos positivos e negativos em diferentes limiares.

Um valor próximo de 0,5 geralmente indica discriminação semelhante à aleatória. Um valor maior indica melhor capacidade de separar as classes, embora não garanta que as probabilidades estejam calibradas.

### 8.6 PR-AUC

A área sob a curva Precisão-Recall é especialmente informativa quando a classe positiva é rara.

Ela evidencia a relação entre precisão e recall conforme o limiar muda. O valor de referência depende da prevalência da classe positiva e da definição exata do cálculo utilizado.

## 9. Resultados do modelo principal

Os metadados do modelo principal registram os seguintes resultados experimentais:

- Algoritmo: Regressão Logística.
- ROC-AUC: 0,5987.
- PR-AUC: 0,0638.
- Precisão: 0,0723.
- Recall: 0,1713.
- F1-score: 0,1016.
- Limiar registrado: 0,06.

Esses números indicam que o modelo ainda tem limitações importantes. A precisão de aproximadamente 7,23% significa que uma parcela pequena das previsões positivas correspondeu à classe positiva nos dados avaliados. O recall de aproximadamente 17,13% indica que o modelo identificou apenas parte dos casos positivos.

A interpretação final depende do conjunto em que as métricas foram calculadas, da prevalência da classe positiva e do procedimento usado para escolher o limiar.

Não se deve apresentar esses resultados como evidência de desempenho comprovado em uma seguradora real.

## 10. Ajuste do limiar de classificação

Um modelo probabilístico pode estimar, por exemplo, uma probabilidade de 0,08 para um registro.

Com um limiar de 0,50, esse registro não seria classificado como positivo. Com um limiar de 0,06, seria classificado como positivo.

Reduzir o limiar tende a aumentar a quantidade de casos sinalizados. Isso pode elevar o recall, mas também pode aumentar os falsos positivos e reduzir a precisão.

O limiar deve ser escolhido considerando:

- O custo de uma fraude não detectada.
- O custo de investigar um caso legítimo.
- A capacidade da equipe de investigação.
- A precisão e o recall observ