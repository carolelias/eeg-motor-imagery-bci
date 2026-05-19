import mne
import numpy as np
import warnings
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold

# Ignorar os avisos de padronização do MNE para limpar o terminal durante o loop
warnings.filterwarnings('ignore')

def avaliar_sessao_bci(file_path):
    """
    Função que recebe o caminho de um arquivo GDF, processa o sinal de EEG,
    extrai características com CSP e retorna a acurácia média do LDA.
    """
    try:
        # 1. Carregamento e Ajuste de Canais
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        raw.rename_channels({'EEG:C3': 'C3', 'EEG:Cz': 'Cz', 'EEG:C4': 'C4'})
        
        channel_types = {
            'C3': 'eeg', 'Cz': 'eeg', 'C4': 'eeg',
            'EOG:ch01': 'eog', 'EOG:ch02': 'eog', 'EOG:ch03': 'eog'
        }
        raw.set_channel_types(channel_types, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        # 2. Extração de Eventos
        events, event_id_dict = mne.events_from_annotations(raw, verbose=False)

        # 3. Pré-processamento (Filtro e ICA)
        raw.filter(l_freq=8., h_freq=30., fir_design='firwin', verbose=False)
        
        ica = mne.preprocessing.ICA(n_components=3, random_state=42, max_iter='auto', verbose=False)
        ica.fit(raw, verbose=False)
        eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_indices

        raw_clean = raw.copy()
        ica.apply(raw_clean, verbose=False)

        # 4. Epoching
        eventos_imagetica = {
            'Esquerda': event_id_dict.get('769'),
            'Direita': event_id_dict.get('770')
        }
        
        epochs = mne.Epochs(
            raw_clean, events, event_id=eventos_imagetica, 
            tmin=-0.5, tmax=3.5, picks='eeg',
            baseline=(-0.5, 0), preload=True, verbose=False
        )

        # 5. CSP e LDA (Classificação)
        X = epochs.get_data(copy=True)
        y = epochs.events[:, -1]

        csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
        X_features = csp.fit_transform(X, y)

        lda = LinearDiscriminantAnalysis()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(lda, X_features, y, cv=cv)

        return np.mean(scores) * 100, np.std(scores) * 100

    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None, None

# =====================================================================
# LOOP DE EXECUÇÃO PARA TODOS OS SUJEITOS
# =====================================================================
print("Iniciando o processamento em lote para os 9 sujeitos...\n")
print("-" * 50)
print(f"{'Sujeito':<10} | {'Arquivo':<15} | {'Acurácia Média':<20}")
print("-" * 50)

acuracias_gerais = []

# =====================================================================
# LOOP DE EXECUÇÃO PARA TODOS OS SUJEITOS (CORRIGIDO)
# =====================================================================
print("Iniciando o processamento em lote para os 9 sujeitos...\n")
print("-" * 50)
print(f"{'Sujeito':<10} | {'Arquivo':<15} | {'Acurácia Média':<20}")
print("-" * 50)

acuracias_gerais = []

# Loop do sujeito 1 ao 9
for sujeito_id in range(1, 10):
    # CORREÇÃO AQUI: Retiramos o '0' fixo após o 'B'. 
    # O {sujeito_id:02d} já garante que o número 1 vire '01' automaticamente.
    nome_arquivo = f'B{sujeito_id:02d}01T.gdf'
    caminho_arquivo = f'dataset/{nome_arquivo}'
    
    media, desvio = avaliar_sessao_bci(caminho_arquivo)
    
    if media is not None:
        print(f"Sujeito {sujeito_id:<2} | {nome_arquivo:<15} | {media:.2f}% (+/- {desvio:.2f}%)")
        acuracias_gerais.append(media)

print("-" * 50)
if acuracias_gerais: # Só calcula a média se a lista não estiver vazia
    print(f"ACURÁCIA GLOBAL (Média dos 9 sujeitos): {np.mean(acuracias_gerais):.2f}%")
print("-" * 50)