# -*- coding: utf-8 -*-
"""
preprocessing.py

Funções responsáveis por carregar os arquivos .gdf do Dataset 2b da BCI
Competition IV, padronizar os canais, aplicar filtragem temporal, remover
artefatos de EOG e segmentar o sinal em épocas (trials) de imagética
motora.

Cada função tem uma única responsabilidade, o que facilita testar e
depurar separadamente cada etapa da pipeline (carregamento, filtragem,
remoção de artefatos e epoching).
"""

import os
import warnings

import mne
import mne.preprocessing
import numpy as np

import config


def carregar_sessao_gdf(caminho_arquivo: str) -> mne.io.Raw:
    """Carrega um arquivo .gdf de uma sessão e padroniza os nomes e tipos
    de canal.

    Parameters
    ----------
    caminho_arquivo : str
        Caminho completo para o arquivo .gdf (ex.: 'dataset/B0101T.gdf').

    Returns
    -------
    raw : mne.io.Raw
        Objeto Raw do MNE com canais renomeados ('C3', 'Cz', 'C4', 'EOG1',
        'EOG2', 'EOG3') e tipos de canal corretamente atribuídos ('eeg' ou
        'eog').
    """
    if not os.path.isfile(caminho_arquivo):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_arquivo}. Verifique se o "
            f"dataset foi baixado e colocado na pasta correta "
            f"(config.DATASET_DIR = '{config.DATASET_DIR}')."
        )

    # Os eventos do Dataset 2b estão embutidos como anotações no próprio
    # arquivo GDF e são lidos posteriormente via mne.events_from_annotations.
    # Não há canal de estímulo dedicado neste dataset.
    raw = mne.io.read_raw_gdf(
        caminho_arquivo,
        eog=config.RAW_EOG_CHANNEL_NAMES,
        preload=True,
        verbose=config.VERBOSE_MNE,
    )

    # Renomeia os canais para nomes mais legíveis. Usamos apenas as chaves
    # de fato presentes no arquivo, pois alguns arquivos podem ter pequenas
    # variações na grafia dos nomes de canal entre sujeitos.
    rename_map = {
        canal_original: canal_novo
        for canal_original, canal_novo in config.EEG_CHANNEL_RENAME_MAP.items()
        if canal_original in raw.ch_names
    }
    raw.rename_channels(rename_map)

    # Garante que os tipos de canal estejam corretos (alguns arquivos GDF
    # carregam todos os canais inicialmente como 'eeg' por padrão).
    tipos_canal = {}
    for canal in config.EEG_CHANNELS:
        if canal in raw.ch_names:
            tipos_canal[canal] = "eeg"
    for canal in config.EOG_CHANNELS:
        if canal in raw.ch_names:
            tipos_canal[canal] = "eog"
    raw.set_channel_types(tipos_canal, verbose=config.VERBOSE_MNE)

    # Define explicitamente o esquema de referência do EEG. O Dataset 2b
    # já fornece os 3 canais de EEG como derivações BIPOLARES (C3, Cz, C4
    # medidos como diferença de potencial entre um par de eletrodos, e não
    # em relação a uma referência comum única -- ver documentação oficial
    # do dataset, seção "Data recording"). Por isso, não faz sentido
    # aplicar uma referência de média ou de eletrodo único adicional aqui;
    # usamos ref_channels=[] para informar ao MNE que os dados já estão em
    # sua forma de referência final, apenas suprimindo o erro
    # "No average reference for the EEG channels has been set" que o MNE
    # levanta ao tentar realizar operações (como o cálculo de covariância
    # dentro do CSP) sem que uma referência tenha sido explicitamente
    # declarada.
    raw.set_eeg_reference(ref_channels=[], verbose=config.VERBOSE_MNE)

    # Atribui a montagem padrão 10-20 para que os canais C3/Cz/C4 tenham
    # posições espaciais conhecidas. Isso é necessário para que, mais
    # adiante, o CSP.plot_patterns() consiga desenhar os mapas
    # topográficos dos padrões espaciais (ver visualization.py). Como o
    # arquivo .gdf original não traz coordenadas de eletrodo, usamos a
    # montagem padrão e ignoramos silenciosamente canais que não constam
    # nela (on_missing='warn' evita que a ausência de canais não-EEG,
    # como os de EOG, interrompa a execução).
    montagem = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montagem, on_missing="warn", verbose=config.VERBOSE_MNE)

    return raw


def filtrar_sinal(raw: mne.io.Raw) -> mne.io.Raw:
    """Aplica filtragem temporal passa-banda (e, opcionalmente, notch) ao
    sinal de EEG/EOG.

    O filtro passa-banda Butterworth (8-30 Hz) isola os ritmos Mu e Beta,
    relevantes para os fenômenos de ERD/ERS durante imagética motora. A
    filtragem é aplicada a todos os canais (EEG e EOG), pois a etapa
    seguinte de remoção de artefatos por regressão linear (ver
    `remover_artefatos_eog`) requer que ambos os sinais estejam na mesma
    faixa de frequência para que a correlação entre eles seja calculada de
    forma consistente.

    Parameters
    ----------
    raw : mne.io.Raw
        Sinal bruto (ainda não filtrado).

    Returns
    -------
    raw_filtrado : mne.io.Raw
        Cópia do sinal de entrada após a filtragem.
    """
    raw_filtrado = raw.copy()

    # iir_params define explicitamente um filtro Butterworth de ordem fixa,
    # em vez do filtro FIR padrão do MNE. Optamos pelo IIR Butterworth por
    # ser o filtro classicamente empregado na literatura de BCI baseada em
    # ERD/ERS (Pfurtscheller & Lopes da Silva, 1999) e por introduzir menor
    # atraso de grupo do que um FIR de ordem equivalente. O parâmetro
    # output='sos' (second-order sections) é usado em vez da representação
    # clássica por função de transferência ('ba'), pois é numericamente
    # mais estável, especialmente importante aqui já que o filtro é
    # aplicado em cascata (passa-banda = passa-alta + passa-baixa).
    iir_params = dict(order=config.FILTER_ORDER_IIR, ftype="butter", output="sos")

    raw_filtrado.filter(
        l_freq=config.BANDPASS_LOW_FREQ,
        h_freq=config.BANDPASS_HIGH_FREQ,
        method="iir",
        iir_params=iir_params,
        verbose=config.VERBOSE_MNE,
    )

    if config.APPLY_NOTCH_FILTER:
        raw_filtrado.notch_filter(
            freqs=config.NOTCH_FREQ, verbose=config.VERBOSE_MNE
        )

    return raw_filtrado


def remover_artefatos_eog(raw: mne.io.Raw) -> mne.io.Raw:
    """Remove artefatos oculares dos canais de EEG por regressão linear
    contra os canais de EOG.

    A documentação oficial do Dataset 2b recomenda explicitamente o uso de
    técnicas como filtragem passa-alta ou regressão linear para remoção de
    artefatos de EOG (Schlögl et al., 2007), e proíbe o uso direto dos
    canais de EOG como entrada do classificador. A abordagem de regressão
    linear (`mne.preprocessing.EOGRegression`) é apropriada para este
    dataset porque ele possui apenas 3 canais de EEG -- um número
    insuficiente para a aplicação robusta de ICA, que tipicamente requer
    muito mais canais do que componentes a serem estimadas para convergir
    de forma estável.

    Parameters
    ----------
    raw : mne.io.Raw
        Sinal já filtrado, contendo tanto os canais de EEG quanto os de
        EOG.

    Returns
    -------
    raw_limpo : mne.io.Raw
        Sinal de EEG com a contribuição estimada do EOG subtraída. Os
        canais de EOG são mantidos no objeto (úteis para inspeção visual),
        mas não devem ser usados na etapa de classificação.
    """
    raw_limpo = raw.copy()

    # EOGRegression estima, por mínimos quadrados, o quanto de cada canal
    # de EOG "contamina" cada canal de EEG, e subtrai essa contribuição
    # estimada do sinal de EEG. É equivalente, em espírito, ao método
    # clássico de Schlögl et al. (2007) referenciado na documentação
    # oficial do dataset.
    modelo_regressao = mne.preprocessing.EOGRegression(
        picks="eeg", picks_artifact="eog"
    )
    modelo_regressao.fit(raw_limpo)
    modelo_regressao.apply(raw_limpo, copy=False)

    return raw_limpo


def extrair_eventos_motor_imagery(raw: mne.io.Raw):
    """Extrai do objeto Raw apenas os eventos de cue de imagética motora
    (769 = mão esquerda, 770 = mão direita), descartando os demais tipos
    de evento (início de trial, início de novo bloco, etc.).

    Parameters
    ----------
    raw : mne.io.Raw
        Sinal com anotações de eventos (carregadas automaticamente pelo
        MNE a partir do arquivo .gdf).

    Returns
    -------
    events : np.ndarray, shape (n_eventos, 3)
        Array de eventos no formato do MNE (amostra, duração, código).
    event_id : dict
        Dicionário {'769': código_interno_769, '770': código_interno_770}
        mapeando os códigos de evento originais para os códigos internos
        atribuídos pelo MNE ao converter anotações em eventos.
    """
    todos_eventos, todos_event_id = mne.events_from_annotations(
        raw, verbose=config.VERBOSE_MNE
    )

    # As anotações do MNE para arquivos GDF da BCI Competition usam como
    # chave a representação em string do código do evento (ex.: '769'),
    # prefixada ocasionalmente por zeros à esquerda dependendo da versão
    # do MNE. Por isso, buscamos a chave de forma robusta.
    # Tenta primeiro os códigos das sessões de treino (769/770); se não
    # encontrar, tenta os códigos das sessões de avaliação (781/783).
    # Ambos marcam o mesmo momento funcional (onset do cue/rótulo de classe),
    # mas o Dataset 2b usa valores diferentes entre treino e avaliação.
    chave_esquerda = _buscar_chave_evento(todos_event_id, config.EVENT_CUE_LEFT)
    chave_direita = _buscar_chave_evento(todos_event_id, config.EVENT_CUE_RIGHT)

    if chave_esquerda is None:
        chave_esquerda = _buscar_chave_evento(todos_event_id, config.EVENT_CUE_LEFT_EVAL)
    if chave_direita is None:
        chave_direita = _buscar_chave_evento(todos_event_id, config.EVENT_CUE_RIGHT_EVAL)

    if chave_esquerda is None or chave_direita is None:
        raise RuntimeError(
            "Não foi possível localizar os eventos de cue (769/770 ou 781/783) nas "
            f"anotações do arquivo. Eventos disponíveis: {todos_event_id}"
        )

    # O MNE atribui códigos internos sequenciais ao converter anotações GDF
    # em eventos, e esses códigos variam entre arquivos de sessões diferentes
    # (ex.: "770" pode virar 11 num arquivo e 5 em outro). Se usarmos esses
    # códigos internos como valores no event_id e depois tentarmos concatenar
    # épocas de sessões diferentes, mne.concatenate_epochs lança ValueError
    # porque a mesma chave ("770") aponta para valores diferentes.
    # A solução é normalizar: substituir os códigos internos pelos códigos
    # originais do dataset (769 e 770), que são fixos e iguais em todas as
    # sessões.
    codigo_interno_esquerda = todos_event_id[chave_esquerda]
    codigo_interno_direita = todos_event_id[chave_direita]

    eventos_normalizados = todos_eventos.copy()
    eventos_normalizados[eventos_normalizados[:, 2] == codigo_interno_esquerda, 2] = config.EVENT_CUE_LEFT
    eventos_normalizados[eventos_normalizados[:, 2] == codigo_interno_direita, 2] = config.EVENT_CUE_RIGHT

    event_id_motor_imagery = {
        "769": config.EVENT_CUE_LEFT,
        "770": config.EVENT_CUE_RIGHT,
    }

    return eventos_normalizados, event_id_motor_imagery


def _buscar_chave_evento(event_id_dict: dict, codigo_evento: int):
    """Busca, de forma tolerante a formatação, a chave do dicionário
    `event_id_dict` que corresponde ao código numérico `codigo_evento`.

    O MNE converte anotações para strings com diferentes formatações
    dependendo da versão (ex.: '769', '769.0' ou, em tese, com zeros à
    esquerda). Em vez de comparar strings diretamente, convertemos cada
    chave para um número de ponto flutuante e comparamos numericamente,
    o que torna a busca robusta a qualquer uma dessas variações de
    formatação.
    """
    for chave in event_id_dict.keys():
        try:
            valor_numerico = float(chave)
        except (TypeError, ValueError):
            continue
        if valor_numerico == float(codigo_evento):
            return chave
    return None


def criar_epocas(raw_limpo: mne.io.Raw, events: np.ndarray, event_id: dict) -> mne.Epochs:
    """Segmenta o sinal contínuo em épocas (trials) de imagética motora.

    A janela temporal [EPOCH_TMIN, EPOCH_TMAX] é definida em config.py e
    é deslocada para frente em relação ao cue, de modo a evitar que o
    Potencial Evocado Visual (VEP) gerado pelo próprio estímulo (a seta na
    tela) contamine as características extraídas pelo CSP.

    Parameters
    ----------
    raw_limpo : mne.io.Raw
        Sinal já filtrado e com artefatos de EOG removidos.
    events : np.ndarray
        Array de eventos retornado por `extrair_eventos_motor_imagery`.
    event_id : dict
        Dicionário de mapeamento {'769': código, '770': código}.

    Returns
    -------
    epochs : mne.Epochs
        Épocas de EEG (apenas canais 'eeg', os canais de EOG são
        descartados nesta etapa pois já cumpriram seu papel na remoção de
        artefatos).
    """
    epochs = mne.Epochs(
        raw_limpo,
        events,
        event_id=event_id,
        tmin=config.EPOCH_TMIN,
        tmax=config.EPOCH_TMAX,
        picks="eeg",
        baseline=None,  # a normalização de baseline não é aplicada aqui,
                        # pois a janela de análise já foi deslocada
                        # especificamente para começar após o transiente
                        # visual do cue; uma correção de baseline usando
                        # um período anterior ao cue poderia reintroduzir
                        # a influência do VEP nas características CSP.
        reject=config.REJECT_PEAK_TO_PEAK,
        preload=True,
        verbose=config.VERBOSE_MNE,
    )
    return epochs


def processar_sessao_completa(caminho_arquivo: str) -> mne.Epochs:
    """Executa a pipeline completa de pré-processamento para um único
    arquivo de sessão: carregamento -> filtragem -> remoção de EOG ->
    epoching.

    Esta função de conveniência encapsula as quatro etapas anteriores,
    sendo o ponto de entrada usado pelo restante do projeto (ver
    pipeline.py).

    Parameters
    ----------
    caminho_arquivo : str
        Caminho completo para o arquivo .gdf da sessão.

    Returns
    -------
    epochs : mne.Epochs
        Épocas de EEG prontas para a extração de características (CSP).
    """
    raw = carregar_sessao_gdf(caminho_arquivo)
    raw_filtrado = filtrar_sinal(raw)
    raw_limpo = remover_artefatos_eog(raw_filtrado)
    events, event_id = extrair_eventos_motor_imagery(raw_limpo)
    epochs = criar_epocas(raw_limpo, events, event_id)
    return epochs


def carregar_e_concatenar_sessoes(caminhos_arquivos: list) -> mne.Epochs:
    """Processa múltiplos arquivos de sessão (ex.: 01T + 02T + 03T de um
    mesmo sujeito) e concatena as épocas resultantes em um único objeto.

    Sessões que não puderem ser carregadas ou processadas (ex.: arquivo
    ausente, ou sessão sem rótulos de classe disponíveis) são ignoradas
    com um aviso impresso no terminal, em vez de interromper toda a
    execução do programa.

    Parameters
    ----------
    caminhos_arquivos : list of str
        Lista de caminhos para os arquivos .gdf a serem processados e
        concatenados.

    Returns
    -------
    epochs_concatenadas : mne.Epochs or None
        Objeto único contendo as épocas de todas as sessões processadas
        com sucesso, ou None se nenhuma sessão pôde ser processada.
    """
    lista_epochs = []

    for caminho in caminhos_arquivos:
        nome_arquivo = os.path.basename(caminho)
        try:
            epochs = processar_sessao_completa(caminho)
            if len(epochs) == 0:
                print(f"  [aviso] {nome_arquivo}: nenhuma época válida "
                      f"após rejeição por artefato; sessão ignorada.")
                continue
            lista_epochs.append(epochs)
        except FileNotFoundError:
            print(f"  [aviso] {nome_arquivo}: arquivo não encontrado; "
                  f"sessão ignorada.")
        except RuntimeError as erro:
            print(f"  [aviso] {nome_arquivo}: {erro}; sessão ignorada.")
        except Exception as erro:  # pylint: disable=broad-except
            # Captura ampla intencional: preferimos seguir processando as
            # demais sessões/sujeitos do que interromper todo o experimento
            # por causa de uma única sessão problemática. Incluímos o nome
            # da classe da exceção (ex.: 'RuntimeError', 'ValueError') na
            # mensagem para facilitar o diagnóstico de problemas futuros,
            # já que exceções inesperadas do MNE costumam ter mensagens
            # textuais muito parecidas com simples avisos informativos.
            print(f"  [aviso] {nome_arquivo}: erro inesperado "
                  f"({type(erro).__name__}: {erro}); sessão ignorada.")

    if not lista_epochs:
        return None

    if len(lista_epochs) == 1:
        return lista_epochs[0]

    return mne.concatenate_epochs(lista_epochs, verbose=config.VERBOSE_MNE)
