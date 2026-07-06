# Classificação Binária de Sinais de EEG para BCI (Dataset 2b)

Pipeline de processamento e classificação de sinais de EEG para
discriminação de imagética motora (mão esquerda vs. mão direita),
desenvolvida no âmbito do projeto de Iniciação Científica FAPESP
(processo 2024/17737-0).

## Estrutura do projeto

```
.
├── dataset/                   # Coloque aqui os arquivos .gdf baixados
├── results/
│   ├── tabela_resultados.csv  # Gerado automaticamente após a execução
│   └── figures/               # Figuras geradas automaticamente
├── src/
│   ├── config.py              # Parâmetros centrais do experimento
│   ├── preprocessing.py       # Carregamento, filtragem, remoção de EOG, epoching
│   ├── pipeline.py            # CSP, LDA, SVM, validação cruzada
│   ├── visualization.py       # Geração de todas as figuras
│   └── main.py                # Script principal (ponto de entrada)
├── requirements.txt
└── README.md
```

## Instalação

As dependências estão listadas em `requirements.txt`. Recomenda-se o uso
de um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Preparação dos dados

1. Baixe o Dataset 2b da BCI Competition IV em
   <https://www.bbci.de/competition/iv/#dataset2b> (arquivos `.gdf`).
2. Coloque todos os 45 arquivos (9 sujeitos × 5 sessões) na pasta
   `dataset/`, mantendo os nomes originais (`B0101T.gdf`, `B0102T.gdf`,
   ..., `B0905E.gdf`).

## Execução

```bash
cd src
python main.py
```

O script processa os 9 sujeitos sequencialmente. Cada sujeito demora
aproximadamente 1-3 minutos (a maior parte consumida pela busca em grade
do SVM); o tempo total estimado é de 15 a 30 minutos.

## Saídas geradas

- **`results/tabela_resultados.csv`**: acurácia, Kappa de Cohen e
  melhores hiperparâmetros do SVM para cada sujeito.
- **`results/figures/`**:
  - `sinal_bruto_vs_filtrado_S1.png`: efeito do filtro passa-banda.
  - `csp_patterns_S1.png`: padrões espaciais do CSP (topomapas).
  - `csp_scatter_S1.png`: dispersão das características CSP por classe.
  - `matriz_confusao_lda_S1.png` / `matriz_confusao_svm_S1.png`.
  - `comparacao_lda_svm.png`: acurácia e Kappa por sujeito (LDA vs. SVM).
  - `boxplot_validacao_cruzada.png`: distribuição do desempenho entre sujeitos.

## Metodologia (resumo)

1. **Carregamento**: arquivos `.gdf` lidos via MNE-Python; canais
   renomeados (3 EEG: C3, Cz, C4; 3 EOG).
2. **Filtragem**: passa-banda Butterworth de 4ª ordem, 8-30 Hz (ritmos Mu
   e Beta).
3. **Remoção de artefatos**: regressão linear EOG→EEG
   (`mne.preprocessing.EOGRegression`), adequada para o número reduzido
   de canais do dataset (ICA não é robusta com apenas 3 canais de EEG).
4. **Epoching**: janela de [+0.5s, +3.5s] relativa ao cue (769/770), para
   evitar contaminação por Potencial Evocado Visual (VEP).
5. **Extração de características**: Common Spatial Pattern (CSP),
   3 componentes, com log-variância.
6. **Classificação**: LDA e SVM (kernel RBF, hiperparâmetros otimizados
   via grid search), ambos encapsulados em `sklearn.pipeline.Pipeline`
   para evitar vazamento de dados na validação cruzada.
7. **Avaliação**: validação cruzada 10-fold sobre as sessões de treino
   (01T + 02T + 03T), reportando acurácia e coeficiente Kappa de Cohen
   (métrica oficial da BCI Competition IV). Os rótulos das sessões de
   avaliação (04E/05E) não estão disponíveis nos arquivos GDF originais.

## Observações

- Todos os parâmetros do experimento (faixa de filtro, janela de época,
  número de componentes CSP, grade de hiperparâmetros do SVM, etc.) estão
  centralizados em `src/config.py`.

## Autores

Carolina Elias de Almeida Américo — aluna de graduação  
Prof. Dr. Leonardo André Ambrosio — orientador  
Projeto de Iniciação Científica FAPESP, processo 2024/17737-0.
