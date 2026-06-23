# -*- coding: utf-8 -*-
"""
config.py

Arquivo central de configuração do projeto de classificação binária de
sinais de EEG (imagética motora de mão esquerda vs. mão direita), utilizando
o Dataset 2b da BCI Competition IV.

Manter todos os parâmetros aqui facilita a reprodutibilidade dos
experimentos e evita "números mágicos" espalhados pelo código.
"""

import os

# --------------------------------------------------------------------------
# Caminhos do projeto
# --------------------------------------------------------------------------
# Pasta onde estão os arquivos .gdf baixados do site da BCI Competition IV.
# Estrutura esperada: dataset/B0101T.gdf, dataset/B0102T.gdf, ...
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

# Pasta onde resultados (tabelas .csv e figuras) serão salvos.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# --------------------------------------------------------------------------
# Sujeitos e sessões do Dataset 2b
# --------------------------------------------------------------------------
# O Dataset 2b contém 9 sujeitos. Para cada sujeito há 5 sessões:
#   01T, 02T -> sessões de treinamento SEM realimentação visual (screening)
#   03T      -> sessão de treinamento COM realimentação visual (smiley)
#   04E, 05E -> sessões de avaliação COM realimentação visual (smiley)
#
# Seguindo o protocolo oficial da competição (Leeb et al., 2008), as três
# primeiras sessões (01T, 02T, 03T) possuem rótulos de classe disponíveis e
# são usadas para treinar o classificador. As sessões 04E e 05E são usadas
# para avaliação. No pacote de dados completo (pós-competição), todas as
# sessões já vêm com rótulos, o que permite uma avaliação fim-a-fim.
SUBJECT_IDS = list(range(1, 10))  # Sujeitos de 1 a 9

TRAIN_SESSION_SUFFIXES = ["01T", "02T", "03T"]
# Os arquivos GDF das sessões de avaliação (04E, 05E) não contêm os rótulos
# de classe nos eventos — ambos os eventos 781 e 783 aparecem em TODOS os
# trials independentemente da classe (são marcadores do protocolo de feedback
# online, não rótulos). Os rótulos verdadeiros foram distribuídos pelos
# organizadores em formato MATLAB separado (.mat). Por isso, a avaliação em
# conjunto de teste independente não é realizada nesta versão do código;
# a métrica principal é o Kappa de Cohen obtido pela validação cruzada sobre
# as sessões de treino (01T + 02T + 03T).
TEST_SESSION_SUFFIXES = []  # desativado — ver comentário acima

# Sessões com problema técnico conhecido no bloco de calibração de EOG
# (não afeta os trials de imagética motora, apenas o bloco inicial de 5 min
# usado para estimativa de artefatos oculares). Ver descrição oficial do
# dataset (Leeb et al., 2008), Tabela 1.
SESSIONS_WITHOUT_EOG_CALIBRATION = ["B0102T", "B0504E"]


def subject_filename(subject_id: int, suffix: str) -> str:
    """Monta o nome de arquivo .gdf a partir do número do sujeito e do
    sufixo de sessão (ex.: subject_id=1, suffix='01T' -> 'B0101T.gdf').
    """
    return f"B{subject_id:02d}{suffix}.gdf"


# --------------------------------------------------------------------------
# Canais
# --------------------------------------------------------------------------
# Nomenclatura dos canais como aparece nos arquivos .gdf originais da
# competição. Os 3 primeiros são EEG (montagem bipolar centrada em C3/Cz/C4)
# e os 3 últimos são EOG (montagem monopolar, ver Figura 2 da documentação).
RAW_EEG_CHANNEL_NAMES = ["EEG:C3", "EEG:Cz", "EEG:C4"]
RAW_EOG_CHANNEL_NAMES = ["EOG:ch01", "EOG:ch02", "EOG:ch03"]

# Nomes padronizados (mais limpos) que usaremos após o carregamento.
EEG_CHANNEL_RENAME_MAP = {
    "EEG:C3": "C3",
    "EEG:Cz": "Cz",
    "EEG:C4": "C4",
    "EOG:ch01": "EOG1",
    "EOG:ch02": "EOG2",
    "EOG:ch03": "EOG3",
}

EEG_CHANNELS = ["C3", "Cz", "C4"]
EOG_CHANNELS = ["EOG1", "EOG2", "EOG3"]

# --------------------------------------------------------------------------
# Códigos de eventos (conforme Tabela 2 da documentação oficial do dataset)
# --------------------------------------------------------------------------
EVENT_CUE_LEFT = 769    # Imagética motora: mão esquerda (classe 1) — sessões de treino
EVENT_CUE_RIGHT = 770   # Imagética motora: mão direita (classe 2) — sessões de treino
# Nas sessões de avaliação (04E, 05E), o cue/rótulo de classe é codificado
# com valores diferentes: 781 para mão esquerda e 783 para mão direita.
# Isso é padrão do Dataset 2b (ver Leeb et al., 2008, Tabela 2).
EVENT_CUE_LEFT_EVAL = 781    # rótulo de classe: mão esquerda, sessões de avaliação
EVENT_CUE_RIGHT_EVAL = 783   # rótulo de classe: mão direita, sessões de avaliação
EVENT_REJECTED_TRIAL = 1023  # Trial marcado como contendo artefato

# Mapeamento de classes para os rótulos usados pelo scikit-learn.
CLASS_LABELS = {EVENT_CUE_LEFT: "Mão Esquerda", EVENT_CUE_RIGHT: "Mão Direita"}

# --------------------------------------------------------------------------
# Parâmetros de pré-processamento
# --------------------------------------------------------------------------
SAMPLING_FREQUENCY = 250.0  # Hz (fixo para o Dataset 2b, ver documentação)

# Filtro passa-banda Butterworth. A faixa 8-30 Hz cobre os ritmos Mu (8-12Hz)
# e Beta (13-30Hz), que concentram a maior parte da informação relevante
# para ERD/ERS durante imagética motora (Pfurtscheller & Lopes da Silva,
# 1999).
BANDPASS_LOW_FREQ = 8.0
BANDPASS_HIGH_FREQ = 30.0
FILTER_ORDER_IIR = 4  # ordem do filtro Butterworth

# Frequência da rede elétrica na Áustria (local de gravação do dataset) é
# 50 Hz. O próprio dataset já é fornecido com filtro notch em 50 Hz
# (ver documentação oficial), mas mantemos a opção configurável aqui para
# eventual reaplicação em outros datasets com diferentes ruídos de rede.
NOTCH_FREQ = 50.0
APPLY_NOTCH_FILTER = False  # já vem filtrado de fábrica nesse dataset

# --------------------------------------------------------------------------
# Parâmetros de epoching (segmentação em épocas)
# --------------------------------------------------------------------------
# Os eventos 769/770 (cue onset) marcam o instante em que a seta indicativa
# da classe é mostrada na tela. Definimos a janela de análise relativa a
# esse instante.
#
# Janela escolhida: [+0.5s, +3.5s] após o cue.
#   - Evita os primeiros ~0.5s, período em que o Potencial Evocado Visual
#     (VEP) gerado pela própria seta (cue) ainda domina o sinal, o que
#     poderia levar o classificador a aprender a discriminar o estímulo
#     visual em vez da imagética motora em si.
#   - A imagética motora efetivamente ocorre entre +1s e +4s relativo ao
#     início do trial (= entre +0s e +3s relativo ao cue, que ocorre em
#     t=1s no timeline do trial). Mantemos uma janela de 3s dentro desse
#     intervalo, começando 0.5s após o cue para dar tempo de a dessincro-
#     nização (ERD) se estabelecer.
EPOCH_TMIN = 0.5   # segundos, relativo ao evento de cue (769/770)
EPOCH_TMAX = 3.5   # segundos, relativo ao evento de cue (769/770)

# Janela de baseline usada para normalização de algumas análises
# exploratórias (ex.: cálculo de ERD% relativo). Não é usada na extração
# de características via CSP, que opera diretamente sobre a época "crua"
# filtrada.
BASELINE_TMIN = -1.0
BASELINE_TMAX = 0.0

# Critério simples de rejeição de épocas por amplitude excessiva (artefato
# residual não capturado pela regressão de EOG). Valor em Volts, pois o MNE
# trabalha internamente em Volts (os dados brutos do GDF estão em microV,
# mas o MNE já faz a conversão automaticamente ao carregar).
REJECT_PEAK_TO_PEAK = dict(eeg=150e-6)  # 150 microV

# --------------------------------------------------------------------------
# Parâmetros de extração de características (CSP)
# --------------------------------------------------------------------------
# Número de componentes CSP. Como há apenas 3 canais de EEG no Dataset 2b
# (C3, Cz, C4), o número máximo de componentes CSP é 3. Usamos todas as
# componentes disponíveis; o próprio classificador linear (LDA/SVM) atribui
# pesos menores às componentes menos discriminativas.
CSP_N_COMPONENTS = 3

# --------------------------------------------------------------------------
# Parâmetros de validação cruzada e dos classificadores
# --------------------------------------------------------------------------
CV_N_SPLITS = 10
RANDOM_STATE = 42  # semente fixa para reprodutibilidade dos experimentos

# Grade de hiperparâmetros para a busca em grade (GridSearchCV) do SVM.
SVM_PARAM_GRID = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
}
SVM_KERNEL = "rbf"

# --------------------------------------------------------------------------
# Outros parâmetros
# --------------------------------------------------------------------------
VERBOSE_MNE = False  # silencia o log interno do MNE (deixamos nossos
                      # próprios prints mais legíveis no terminal)
