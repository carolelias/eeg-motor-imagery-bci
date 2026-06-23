# -*- coding: utf-8 -*-
"""
main.py

Ponto de entrada do projeto. Executa a pipeline completa de classificação
binária de imagética motora (mão esquerda vs. mão direita) para todos os
9 sujeitos do Dataset 2b da BCI Competition IV, comparando os
classificadores LDA e SVM.

Uso:
    python main.py

Pré-requisitos:
    - Os arquivos .gdf do Dataset 2b devem estar na pasta definida em
      config.DATASET_DIR (por padrão, uma pasta 'dataset/' na raiz do
      projeto), com os nomes originais (ex.: B0101T.gdf, B0102T.gdf, ...).
    - Bibliotecas necessárias: mne, numpy, scipy, scikit-learn, matplotlib,
      pandas. Ver requirements.txt.

Saídas geradas:
    - results/tabela_resultados.csv: tabela com as métricas de cada
      sujeito e de cada classificador.
    - results/figures/*.png: figuras descritas em visualization.py.
    - Relatório impresso no terminal com um resumo geral do experimento.
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd

import config
import pipeline
import preprocessing
import visualization

# Os avisos do MNE sobre, por exemplo, ausência de posições de eletrodo
# para topomapas de baixa densidade de canais são esperados neste dataset
# (apenas 3 canais de EEG) e não indicam um problema real na análise.
# Suprimimo-los para manter a saída do terminal legível, mas erros
# continuam sendo exibidos normalmente.
warnings.filterwarnings("ignore", category=UserWarning, module="mne")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


SUJEITO_EXEMPLAR_FIGURAS = 1  # sujeito usado para as figuras ilustrativas
                               # (sinal bruto/filtrado, padrões CSP,
                               # scatter de características, matrizes de
                               # confusão). Qualquer sujeito de 1 a 9 pode
                               # ser escolhido; o sujeito 1 é usado por
                               # convenção neste projeto.


def imprimir_cabecalho():
    print("=" * 70)
    print("CLASSIFICAÇÃO BINÁRIA DE SINAIS DE EEG PARA BCI")
    print("Imagética Motora: Mão Esquerda vs. Mão Direita")
    print("Dataset: BCI Competition IV -- Dataset 2b")
    print("Classificadores avaliados: LDA e SVM (kernel RBF)")
    print("=" * 70)
    print(f"Pasta do dataset: {config.DATASET_DIR}")
    print(f"Sujeitos a processar: {config.SUBJECT_IDS}")
    print(f"Banda de filtragem: {config.BANDPASS_LOW_FREQ}-"
          f"{config.BANDPASS_HIGH_FREQ} Hz")
    print(f"Janela de época: [{config.EPOCH_TMIN}s, {config.EPOCH_TMAX}s] "
          f"relativa ao cue")
    print(f"Componentes CSP: {config.CSP_N_COMPONENTS}")
    print(f"Validação cruzada: {config.CV_N_SPLITS}-fold")


def montar_dataframe_resultados(lista_resultados: list) -> pd.DataFrame:
    """Converte a lista de dicionários de resultado (um por sujeito) em
    uma tabela pandas, descartando objetos pesados/não tabulares (como as
    pipelines treinadas e os arrays de dados brutos) que não devem ser
    exportados para o arquivo .csv final.
    """
    colunas_para_tabela = [
        "subject_id", "n_epocas_treino",
        "lda_cv_acuracia", "lda_cv_desvio", "lda_cv_kappa", "lda_cv_kappa_desvio",
        "svm_cv_acuracia", "svm_cv_desvio", "svm_cv_kappa", "svm_cv_kappa_desvio",
        "svm_melhores_parametros", "erro",
    ]
    linhas = [{coluna: resultado.get(coluna) for coluna in colunas_para_tabela}
              for resultado in lista_resultados]
    return pd.DataFrame(linhas)


def imprimir_tabela_resumo(df_resultados: pd.DataFrame):
    print("\n" + "=" * 85)
    print("RESUMO DOS RESULTADOS POR SUJEITO (validação cruzada 10-fold sobre treino)")
    print("=" * 85)
    cabecalho = (
        f"{'Sujeito':<8} | {'Épocas':<7} | "
        f"{'LDA Acc (%)':<13} | {'LDA Kappa':<11} | "
        f"{'SVM Acc (%)':<13} | {'SVM Kappa':<11}"
    )
    print(cabecalho)
    print("-" * 85)

    for _, linha in df_resultados.iterrows():
        if pd.notna(linha["erro"]):
            print(f"{int(linha['subject_id']):<8} | [não processado: {linha['erro']}]")
            continue

        print(
            f"{int(linha['subject_id']):<8} | "
            f"{int(linha['n_epocas_treino']):<7} | "
            f"{linha['lda_cv_acuracia']:<7.2f} ± {linha['lda_cv_desvio']:<4.2f} | "
            f"{linha['lda_cv_kappa']:<5.3f} ± {linha['lda_cv_kappa_desvio']:<4.3f} | "
            f"{linha['svm_cv_acuracia']:<7.2f} ± {linha['svm_cv_desvio']:<4.2f} | "
            f"{linha['svm_cv_kappa']:<5.3f} ± {linha['svm_cv_kappa_desvio']:<5.3f}"
        )

    print("-" * 85)

    df_valido = df_resultados[df_resultados["erro"].isna()]
    if df_valido.empty:
        print("Nenhum sujeito foi processado com sucesso.")
        return

    print("\nMÉDIAS GERAIS (entre sujeitos processados com sucesso):")
    print(f"  LDA: acurácia = {df_valido['lda_cv_acuracia'].mean():.2f}% "
          f"(±{df_valido['lda_cv_acuracia'].std():.2f}%),  "
          f"Kappa = {df_valido['lda_cv_kappa'].mean():.3f} "
          f"(±{df_valido['lda_cv_kappa'].std():.3f})")
    print(f"  SVM: acurácia = {df_valido['svm_cv_acuracia'].mean():.2f}% "
          f"(±{df_valido['svm_cv_acuracia'].std():.2f}%),  "
          f"Kappa = {df_valido['svm_cv_kappa'].mean():.3f} "
          f"(±{df_valido['svm_cv_kappa'].std():.3f})")

    melhor_modelo = (
        "SVM" if df_valido["svm_cv_kappa"].mean() > df_valido["lda_cv_kappa"].mean()
        else "LDA"
    )
    print(f"\n  Com base no Kappa médio (CV), o classificador com melhor "
          f"desempenho foi: {melhor_modelo}.")


def gerar_figuras_sujeito_exemplar(resultado_exemplar: dict):
    """Gera o conjunto de figuras ilustrativas (sinal bruto/filtrado,
    padrões CSP, scatter de características e matrizes de confusão) para
    um único sujeito exemplar, evitando gerar centenas de figuras
    repetitivas para todos os 9 sujeitos.
    """
    subject_id = resultado_exemplar["subject_id"]
    print(f"\nGerando figuras ilustrativas para o sujeito {subject_id}...")

    # Figura 1: sinal bruto vs. filtrado. Carregamos a primeira sessão de
    # treino novamente (de forma independente do restante da pipeline)
    # apenas para fins de visualização, já que o objeto Raw bruto não é
    # mantido em memória após o epoching para economizar RAM durante o
    # processamento em lote dos 9 sujeitos.
    try:
        caminho_sessao_exemplar = (
            f"{config.DATASET_DIR}/"
            f"{config.subject_filename(subject_id, config.TRAIN_SESSION_SUFFIXES[0])}"
        )
        raw_bruto = preprocessing.carregar_sessao_gdf(caminho_sessao_exemplar)
        raw_filtrado = preprocessing.filtrar_sinal(raw_bruto)
        visualization.plotar_sinal_bruto_vs_filtrado(raw_bruto, raw_filtrado, subject_id)
    except Exception as erro:  # pylint: disable=broad-except
        print(f"  [aviso] Não foi possível gerar a figura de sinal "
              f"bruto/filtrado: {erro}")

    # Figura 2: padrões espaciais CSP.
    try:
        visualization.plotar_padroes_csp(
            resultado_exemplar["pipeline_lda"],
            resultado_exemplar["epochs_treino"].info,
            subject_id,
        )
    except Exception as erro:  # pylint: disable=broad-except
        print(f"  [aviso] Não foi possível gerar os padrões CSP: {erro}")

    # Figura 3: dispersão das características CSP.
    try:
        visualization.plotar_dispersao_caracteristicas_csp(
            resultado_exemplar["pipeline_lda"],
            resultado_exemplar["X_treino"],
            resultado_exemplar["y_treino"],
            subject_id,
        )
    except Exception as erro:  # pylint: disable=broad-except
        print(f"  [aviso] Não foi possível gerar o scatter de "
              f"características: {erro}")

    # Matrizes de confusão: não geradas porque a avaliação é por CV
    # (os rótulos das sessões de avaliação 04E/05E não estão disponíveis
    # nos arquivos GDF; ver comentário em config.TEST_SESSION_SUFFIXES).


def main():
    tempo_inicio_total = time.time()
    imprimir_cabecalho()

    lista_resultados = []

    for subject_id in config.SUBJECT_IDS:
        resultado = pipeline.avaliar_sujeito(subject_id, verbose=True)
        lista_resultados.append(resultado)

    # ------------------------------------------------------------------
    # Tabela consolidada de resultados
    # ------------------------------------------------------------------
    df_resultados = montar_dataframe_resultados(lista_resultados)
    imprimir_tabela_resumo(df_resultados)

    config_dir_results = config.RESULTS_DIR
    import os
    os.makedirs(config_dir_results, exist_ok=True)
    caminho_csv = os.path.join(config_dir_results, "tabela_resultados.csv")
    df_resultados.to_csv(caminho_csv, index=False)
    print(f"\nTabela de resultados salva em: {caminho_csv}")

    # ------------------------------------------------------------------
    # Figuras: comparação entre sujeitos e figuras do sujeito exemplar
    # ------------------------------------------------------------------
    print("\nGerando figuras de comparação entre classificadores...")
    visualization.plotar_comparacao_classificadores(df_resultados)
    visualization.plotar_boxplot_validacao_cruzada(df_resultados)

    resultado_exemplar = next(
        (r for r in lista_resultados if r["subject_id"] == SUJEITO_EXEMPLAR_FIGURAS and r["erro"] is None),
        None,
    )
    if resultado_exemplar is not None:
        gerar_figuras_sujeito_exemplar(resultado_exemplar)
    else:
        print(f"\n[aviso] Sujeito exemplar (S{SUJEITO_EXEMPLAR_FIGURAS}) não "
              f"pôde ser processado; figuras ilustrativas individuais não "
              f"foram geradas. Considere alterar SUJEITO_EXEMPLAR_FIGURAS "
              f"em main.py para outro sujeito disponível.")

    tempo_total = time.time() - tempo_inicio_total
    print(f"\n{'=' * 70}")
    print(f"Processamento concluído em {tempo_total / 60:.1f} minutos.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
        sys.exit(1)
