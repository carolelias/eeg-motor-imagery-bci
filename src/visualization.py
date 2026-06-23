# -*- coding: utf-8 -*-
"""
visualization.py

Funções responsáveis por gerar as figuras do projeto:

1. Sinal de EEG bruto vs. filtrado (um sujeito exemplar).
2. Padrões espaciais do CSP (topomapas), mostrando a lateralização
   hemisférica esperada para imagética motora.
3. Dispersão (scatter) das duas primeiras componentes de características
   CSP, coloridas por classe.
4. Matrizes de confusão (LDA e SVM) para um sujeito exemplar.
5. Gráfico de barras comparando acurácia e Kappa entre LDA e SVM para
   todos os sujeitos.

Todas as figuras são salvas em disco (em config.FIGURES_DIR) no formato
PNG, em resolução adequada para inclusão direta no relatório LaTeX.
"""

import os

import matplotlib
matplotlib.use("Agg")  # backend não interativo: necessário para salvar
                        # figuras em ambientes sem display (servidores,
                        # execução em lote, etc.) sem lançar erros.
import matplotlib.pyplot as plt
import numpy as np

import config

# Paleta de cores consistente usada em todas as figuras do projeto.
COR_MAO_ESQUERDA = "#1f77b4"   # azul
COR_MAO_DIREITA = "#d62728"    # vermelho
COR_LDA = "#2ca02c"            # verde
COR_SVM = "#ff7f0e"            # laranja

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def _garantir_pasta_figuras():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)


def _salvar_figura(fig, nome_arquivo: str):
    _garantir_pasta_figuras()
    caminho = os.path.join(config.FIGURES_DIR, nome_arquivo)
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura salva em: {caminho}")


def plotar_sinal_bruto_vs_filtrado(raw_bruto, raw_filtrado, subject_id: int,
                                    duracao_segundos: float = 10.0,
                                    inicio_segundos: float = 30.0):
    """Plota um trecho do sinal de EEG (canal C3) antes e depois da
    filtragem passa-banda, permitindo visualizar o efeito da remoção de
    componentes de baixa e alta frequência.

    Parameters
    ----------
    raw_bruto : mne.io.Raw
        Sinal antes da filtragem.
    raw_filtrado : mne.io.Raw
        Sinal após a filtragem passa-banda (8-30 Hz).
    subject_id : int
        Identificador do sujeito, usado apenas no título da figura e no
        nome do arquivo salvo.
    duracao_segundos : float
        Duração, em segundos, do trecho de sinal a ser exibido.
    inicio_segundos : float
        Instante inicial, em segundos, do trecho a ser exibido.
    """
    canal = "C3"
    sfreq = raw_bruto.info["sfreq"]
    amostra_inicio = int(inicio_segundos * sfreq)
    amostra_fim = int((inicio_segundos + duracao_segundos) * sfreq)

    dados_bruto, tempos = raw_bruto.get_data(
        picks=canal, start=amostra_inicio, stop=amostra_fim, return_times=True
    )
    dados_filtrado, _ = raw_filtrado.get_data(
        picks=canal, start=amostra_inicio, stop=amostra_fim, return_times=True
    )

    # Converte de Volts (unidade interna do MNE) para microVolts, unidade
    # convencional em EEG.
    dados_bruto_uv = dados_bruto[0] * 1e6
    dados_filtrado_uv = dados_filtrado[0] * 1e6

    fig, eixos = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    eixos[0].plot(tempos, dados_bruto_uv, color="gray", linewidth=0.8)
    eixos[0].set_title(f"Canal {canal} -- Sinal bruto")
    eixos[0].set_ylabel("Amplitude (µV)")

    eixos[1].plot(tempos, dados_filtrado_uv, color=COR_LDA, linewidth=0.8)
    eixos[1].set_title(f"Canal {canal} -- Sinal filtrado "
                        f"({config.BANDPASS_LOW_FREQ:.0f}-"
                        f"{config.BANDPASS_HIGH_FREQ:.0f} Hz)")
    eixos[1].set_ylabel("Amplitude (µV)")
    eixos[1].set_xlabel("Tempo (s)")

    fig.suptitle(f"Sujeito {subject_id}: efeito da filtragem passa-banda")
    fig.tight_layout()

    _salvar_figura(fig, f"sinal_bruto_vs_filtrado_S{subject_id}.png")


def plotar_padroes_csp(pipeline_treinada, info_mne, subject_id: int):
    """Plota os padrões espaciais (spatial patterns) estimados pelo CSP
    como topomapas, exibindo como cada componente CSP pondera os três
    canais de EEG (C3, Cz, C4).

    Como o Dataset 2b possui apenas 3 canais, o topomapa terá resolução
    espacial limitada, mas ainda é informativo para ilustrar a
    lateralização hemisférica esperada entre as classes de imagética
    motora (maior peso em C3 para uma classe, em C4 para a outra).

    Parameters
    ----------
    pipeline_treinada : sklearn.pipeline.Pipeline
        Pipeline contendo uma etapa nomeada 'csp' já ajustada (fit).
    info_mne : mne.Info
        Objeto Info do MNE (contém posições de eletrodo), tipicamente
        obtido de `epochs.info`.
    subject_id : int
        Identificador do sujeito, usado no título e no nome do arquivo.
    """
    csp = pipeline_treinada.named_steps["csp"]

    fig, eixos = plt.subplots(1, config.CSP_N_COMPONENTS, figsize=(4 * config.CSP_N_COMPONENTS, 4))
    if config.CSP_N_COMPONENTS == 1:
        eixos = [eixos]

    csp.plot_patterns(
        info_mne,
        components=list(range(config.CSP_N_COMPONENTS)),
        ch_type="eeg",
        axes=eixos,
        show=False,
        colorbar=False,
    )
    fig.suptitle(f"Sujeito {subject_id}: padrões espaciais (CSP)")
    fig.tight_layout()

    _salvar_figura(fig, f"csp_patterns_S{subject_id}.png")


def plotar_dispersao_caracteristicas_csp(pipeline_treinada, X_treino, y_treino, subject_id: int):
    """Plota um gráfico de dispersão (scatter) das duas primeiras
    componentes de características extraídas pelo CSP (log-variância),
    coloridas de acordo com a classe verdadeira (mão esquerda / mão
    direita).

    Este gráfico permite visualizar, em duas dimensões, o quão separáveis
    as duas classes se tornam após a filtragem espacial do CSP -- uma
    inspeção visual direta da qualidade da extração de características
    antes mesmo de aplicar o classificador.

    Parameters
    ----------
    pipeline_treinada : sklearn.pipeline.Pipeline
        Pipeline contendo uma etapa 'csp' já ajustada.
    X_treino : np.ndarray
        Dados de treino (formato bruto, antes da extração CSP).
    y_treino : np.ndarray
        Rótulos de classe correspondentes.
    subject_id : int
        Identificador do sujeito.
    """
    csp = pipeline_treinada.named_steps["csp"]
    caracteristicas = csp.transform(X_treino)

    classes_unicas = np.unique(y_treino)
    rotulo_esquerda, rotulo_direita = classes_unicas.min(), classes_unicas.max()

    fig, eixo = plt.subplots(figsize=(6, 5))

    mascara_esquerda = y_treino == rotulo_esquerda
    mascara_direita = y_treino == rotulo_direita

    eixo.scatter(
        caracteristicas[mascara_esquerda, 0],
        caracteristicas[mascara_esquerda, 1],
        c=COR_MAO_ESQUERDA, label="Mão Esquerda", alpha=0.7, edgecolors="k", linewidths=0.3,
    )
    eixo.scatter(
        caracteristicas[mascara_direita, 0],
        caracteristicas[mascara_direita, 1],
        c=COR_MAO_DIREITA, label="Mão Direita", alpha=0.7, edgecolors="k", linewidths=0.3,
    )

    eixo.set_xlabel("Componente CSP 1 (log-variância)")
    eixo.set_ylabel("Componente CSP 2 (log-variância)")
    eixo.set_title(f"Sujeito {subject_id}: características CSP por classe")
    eixo.legend()
    fig.tight_layout()

    _salvar_figura(fig, f"csp_scatter_S{subject_id}.png")


def plotar_matriz_confusao(matriz_confusao: np.ndarray, nome_modelo: str, subject_id: int):
    """Plota a matriz de confusão de um classificador para um sujeito
    exemplar.

    Parameters
    ----------
    matriz_confusao : np.ndarray, shape (2, 2)
        Matriz de confusão retornada por
        `sklearn.metrics.confusion_matrix`.
    nome_modelo : str
        Nome do classificador (ex.: 'LDA' ou 'SVM'), usado no título.
    subject_id : int
        Identificador do sujeito.
    """
    rotulos = ["Mão Esquerda", "Mão Direita"]

    fig, eixo = plt.subplots(figsize=(4.5, 4))
    imagem = eixo.imshow(matriz_confusao, cmap="Blues")

    eixo.set_xticks([0, 1])
    eixo.set_yticks([0, 1])
    eixo.set_xticklabels(rotulos)
    eixo.set_yticklabels(rotulos)
    eixo.set_xlabel("Classe Predita")
    eixo.set_ylabel("Classe Verdadeira")
    eixo.set_title(f"Sujeito {subject_id} -- Matriz de Confusão ({nome_modelo})")

    valor_maximo = matriz_confusao.max()
    for i in range(matriz_confusao.shape[0]):
        for j in range(matriz_confusao.shape[1]):
            cor_texto = "white" if matriz_confusao[i, j] > valor_maximo / 2 else "black"
            eixo.text(j, i, str(matriz_confusao[i, j]), ha="center", va="center", color=cor_texto)

    fig.colorbar(imagem, ax=eixo, fraction=0.046, pad=0.04)
    fig.tight_layout()

    _salvar_figura(fig, f"matriz_confusao_{nome_modelo.lower()}_S{subject_id}.png")


def plotar_comparacao_classificadores(df_resultados):
    """Plota dois gráficos de barras lado a lado comparando, para cada
    sujeito, a acurácia (%) e o coeficiente Kappa obtidos pelos
    classificadores LDA e SVM no conjunto de teste.

    Parameters
    ----------
    df_resultados : pandas.DataFrame
        Tabela de resultados com (no mínimo) as colunas 'subject_id',
        'lda_teste_acuracia', 'svm_teste_acuracia', 'lda_teste_kappa' e
        'svm_teste_kappa'.
    """
    df_valido = df_resultados.dropna(subset=["lda_cv_acuracia", "svm_cv_acuracia"])

    if df_valido.empty:
        print("  [aviso] Nenhum resultado disponível; gráfico de comparação não foi gerado.")
        return

    sujeitos = df_valido["subject_id"].astype(str).values
    posicoes = np.arange(len(sujeitos))
    largura_barra = 0.35

    fig, (eixo_acc, eixo_kappa) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Subplot 1: Acurácia ---
    eixo_acc.bar(posicoes - largura_barra / 2, df_valido["lda_cv_acuracia"],
                 width=largura_barra, label="LDA", color=COR_LDA)
    eixo_acc.bar(posicoes + largura_barra / 2, df_valido["svm_cv_acuracia"],
                 width=largura_barra, label="SVM", color=COR_SVM)
    eixo_acc.axhline(50, color="gray", linestyle="--", linewidth=1, label="Nível de chance")
    eixo_acc.set_xticks(posicoes)
    eixo_acc.set_xticklabels(sujeitos)
    eixo_acc.set_xlabel("Sujeito")
    eixo_acc.set_ylabel("Acurácia (validação cruzada, %)")
    eixo_acc.set_title("Acurácia por sujeito (CV 10-fold)")
    eixo_acc.legend()
    eixo_acc.set_ylim(0, 100)

    # --- Subplot 2: Kappa ---
    eixo_kappa.bar(posicoes - largura_barra / 2, df_valido["lda_cv_kappa"],
                   width=largura_barra, label="LDA", color=COR_LDA)
    eixo_kappa.bar(posicoes + largura_barra / 2, df_valido["svm_cv_kappa"],
                   width=largura_barra, label="SVM", color=COR_SVM)
    eixo_kappa.axhline(0, color="gray", linestyle="--", linewidth=1, label="Nível de chance")
    eixo_kappa.set_xticks(posicoes)
    eixo_kappa.set_xticklabels(sujeitos)
    eixo_kappa.set_xlabel("Sujeito")
    eixo_kappa.set_ylabel("Coeficiente Kappa de Cohen")
    eixo_kappa.set_title("Kappa por sujeito (CV 10-fold)")
    eixo_kappa.legend()

    fig.suptitle("Comparação de desempenho: LDA vs. SVM (validação cruzada)")
    fig.tight_layout()

    _salvar_figura(fig, "comparacao_lda_svm.png")


def plotar_boxplot_validacao_cruzada(df_resultados):
    """Plota um boxplot comparando a distribuição das acurácias de
    validação cruzada (sobre o conjunto de treino) entre os classificadores
    LDA e SVM, agregando todos os sujeitos.

    Parameters
    ----------
    df_resultados : pandas.DataFrame
        Tabela de resultados com as colunas 'lda_cv_acuracia' e
        'svm_cv_acuracia'.
    """
    df_valido = df_resultados.dropna(subset=["lda_cv_acuracia", "svm_cv_acuracia"])

    if df_valido.empty:
        print("  [aviso] Nenhum resultado de validação cruzada disponível; "
              "boxplot não foi gerado.")
        return

    fig, eixo = plt.subplots(figsize=(5, 5))
    dados_boxplot = [df_valido["lda_cv_acuracia"].values, df_valido["svm_cv_acuracia"].values]

    caixas = eixo.boxplot(
        dados_boxplot,
        labels=["LDA", "SVM"],
        patch_artist=True,
        widths=0.5,
    )
    for caixa, cor in zip(caixas["boxes"], [COR_LDA, COR_SVM]):
        caixa.set_facecolor(cor)
        caixa.set_alpha(0.6)

    eixo.axhline(50, color="gray", linestyle="--", linewidth=1)
    eixo.set_ylabel("Acurácia (validação cruzada no treino, %)")
    eixo.set_title("Distribuição da acurácia entre os 9 sujeitos")
    fig.tight_layout()

    _salvar_figura(fig, "boxplot_validacao_cruzada.png")
