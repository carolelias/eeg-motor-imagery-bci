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
# Sessões 01T-03T têm rótulos de classe disponíveis e são usadas para treino.
# As sessões 04E/05E não embutem os rótulos nos arquivos GDF — foram
# distribuídos em formato .mat separado. Por isso, a avaliação é feita por
# validação cruzada sobre as sessões de treino.
SUBJECT_IDS = list(range(1, 10))

TRAIN_SESSION_SUFFIXES = ["01T", "02T", "03T"]
TEST_SESSION_SUFFIXES = []  # desativado — rótulos não disponíveis nos GDF

# Sessões com problema técnico no bloco de calibração de EOG
# (não afeta os trials de imagética motora).
SESSIONS_WITHOUT_EOG_CALIBRATION = ["B0102T", "B0504E"]


def subject_filename(subject_id: int, suffix: str) -> str:
    """Monta o nome de arquivo .gdf a partir do número do sujeito e do
    sufixo de sessão (ex.: subject_id=1, suffix='01T' -> 'B0101T.gdf').
    """
    return f"B{subject_id:02d}{suffix}.gdf"


# --------------------------------------------------------------------------
# Canais
# --------------------------------------------------------------------------
# Nomenclatura original dos arquivos .gdf (3 EEG bipolares + 3 EOG monopolares).
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
EVENT_CUE_LEFT = 769    # mão esquerda — sessões de treino
EVENT_CUE_RIGHT = 770   # mão direita — sessões de treino
# Nas sessões de avaliação os cues têm códigos diferentes (ver Tabela 2 da documentação).
EVENT_CUE_LEFT_EVAL = 781
EVENT_CUE_RIGHT_EVAL = 783
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

# O dataset já vem com filtro notch em 50 Hz aplicado.
NOTCH_FREQ = 50.0
APPLY_NOTCH_FILTER = False

# --------------------------------------------------------------------------
# Parâmetros de epoching (segmentação em épocas)
# --------------------------------------------------------------------------
# Janela [+0.5s, +3.5s] relativa ao cue: os primeiros 0.5s são descartados
# para evitar contaminação pelo VEP gerado pelo estímulo visual (a seta).
EPOCH_TMIN = 0.5
EPOCH_TMAX = 3.5

# Janela de baseline para análises exploratórias de ERD; não usada no CSP.
BASELINE_TMIN = -1.0
BASELINE_TMAX = 0.0

# Rejeição de épocas por amplitude excessiva (MNE usa Volts internamente).
REJECT_PEAK_TO_PEAK = dict(eeg=150e-6)  # 150 µV

# --------------------------------------------------------------------------
# Parâmetros de extração de características (CSP)
# --------------------------------------------------------------------------
# Com apenas 3 canais, o máximo de componentes CSP é 3.
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
