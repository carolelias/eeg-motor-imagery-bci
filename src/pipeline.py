# -*- coding: utf-8 -*-
"""
pipeline.py

Define as pipelines de classificação (CSP + LDA e CSP + SVM), a estratégia
de avaliação por validação cruzada sobre o conjunto de treino, e as funções
que orquestram o experimento completo para um único sujeito.

Nota: os rótulos de classe das sessões de avaliação (04E, 05E) não estão
disponíveis nos arquivos GDF originais (ver config.TEST_SESSION_SUFFIXES);
por isso, a métrica principal reportada é o Kappa de Cohen obtido pela
validação cruzada 10-fold sobre as sessões de treino (01T + 02T + 03T).

A métrica de desempenho oficial da BCI Competition IV é o coeficiente
Kappa de Cohen, por descontar a taxa de acerto esperada ao acaso. Reporta-
mos também a acurácia simples, por ser mais facilmente interpretável.
"""

import time

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import config
import preprocessing


def montar_pipeline_lda() -> Pipeline:
    """Monta a pipeline CSP + LDA.

    O CSP fica dentro da Pipeline para que os filtros espaciais sejam
    reestimados a cada fold da validação cruzada, evitando data leakage.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    """
    csp = CSP(
        n_components=config.CSP_N_COMPONENTS,
        reg=None,
        log=True,  # log-variância das componentes
        norm_trace=False,
    )
    lda = LinearDiscriminantAnalysis()

    pipeline = Pipeline([("csp", csp), ("lda", lda)])
    return pipeline


def montar_pipeline_svm() -> Pipeline:
    """Monta a pipeline CSP + SVM (kernel RBF).

    O StandardScaler é necessário porque o SVM com kernel RBF é sensível
    à escala das características.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    """
    from sklearn.preprocessing import StandardScaler

    csp = CSP(
        n_components=config.CSP_N_COMPONENTS,
        reg=None,
        log=True,
        norm_trace=False,
    )
    scaler = StandardScaler()
    svc = SVC(kernel=config.SVM_KERNEL, probability=False)

    pipeline = Pipeline([("csp", csp), ("scaler", scaler), ("svc", svc)])
    return pipeline


def validacao_cruzada_treino(pipeline: Pipeline, X: np.ndarray, y: np.ndarray):
    """Executa validação cruzada k-fold sobre o conjunto de TREINO.

    Retorna acurácia e coeficiente Kappa de Cohen. O Kappa é a métrica
    oficial da BCI Competition IV por descontar o acerto esperado ao acaso
    em um classificador aleatório (50% para duas classes balanceadas).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline de classificação já configurada (CSP + LDA ou CSP + SVM).
    X : np.ndarray, shape (n_epocas, n_canais, n_amostras)
        Dados de treino no formato exigido pelo `mne.decoding.CSP`.
    y : np.ndarray, shape (n_epocas,)
        Rótulos de classe correspondentes a cada época.

    Returns
    -------
    media_acuracia : float
        Acurácia média (em %) sobre os k folds.
    desvio_acuracia : float
        Desvio padrão da acurácia (em %) sobre os k folds.
    media_kappa : float
        Kappa de Cohen médio sobre os k folds.
    desvio_kappa : float
        Desvio padrão do Kappa sobre os k folds.
    """
    cv = KFold(
        n_splits=config.CV_N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )
    kappa_scorer = make_scorer(cohen_kappa_score)
    resultados = cross_validate(
        pipeline, X, y, cv=cv,
        scoring={"accuracy": "accuracy", "kappa": kappa_scorer},
    )
    accs = resultados["test_accuracy"]
    kappas = resultados["test_kappa"]
    return accs.mean() * 100.0, accs.std() * 100.0, kappas.mean(), kappas.std()


def ajustar_svm_com_grid_search(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Realiza busca em grade (grid search) dos hiperparâmetros C e gamma
    do SVM, usando validação cruzada interna sobre o conjunto de treino, e
    retorna a pipeline já ajustada (fit) com os melhores hiperparâmetros
    encontrados.

    Parameters
    ----------
    X : np.ndarray, shape (n_epocas, n_canais, n_amostras)
        Dados de treino.
    y : np.ndarray, shape (n_epocas,)
        Rótulos de treino.

    Returns
    -------
    melhor_pipeline : sklearn.pipeline.Pipeline
        Pipeline CSP + StandardScaler + SVM já treinada com os melhores
        hiperparâmetros encontrados na busca em grade.
    """
    pipeline_base = montar_pipeline_svm()
    cv_interno = KFold(
        n_splits=config.CV_N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )

    busca = GridSearchCV(
        pipeline_base,
        param_grid=config.SVM_PARAM_GRID,
        cv=cv_interno,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    busca.fit(X, y)
    return busca.best_estimator_, busca.best_params_


def avaliar_em_teste(pipeline_treinada: Pipeline, X_teste: np.ndarray, y_teste: np.ndarray):
    """Avalia uma pipeline já treinada sobre um conjunto de teste
    independente, calculando acurácia, coeficiente Kappa e matriz de
    confusão.

    Nota: não é chamada no fluxo atual porque os rótulos das sessões de
    avaliação (04E/05E) não estão nos arquivos GDF (ver config.py).

    Parameters
    ----------
    pipeline_treinada : sklearn.pipeline.Pipeline
        Pipeline já ajustada (fit) sobre o conjunto de treino.
    X_teste : np.ndarray
        Dados de teste.
    y_teste : np.ndarray
        Rótulos verdadeiros do conjunto de teste.

    Returns
    -------
    metrics : dict
        Dicionário com as chaves 'acuracia' (%), 'kappa' e
        'matriz_confusao' (np.ndarray 2x2).
    """
    y_predito = pipeline_treinada.predict(X_teste)

    acuracia = accuracy_score(y_teste, y_predito) * 100.0
    kappa = cohen_kappa_score(y_teste, y_predito)
    matriz_confusao = confusion_matrix(y_teste, y_predito)

    return {
        "acuracia": acuracia,
        "kappa": kappa,
        "matriz_confusao": matriz_confusao,
        "y_predito": y_predito,
    }


def avaliar_sujeito(subject_id: int, verbose: bool = True) -> dict:
    """Executa a pipeline experimental completa para um único sujeito:

    1. Carrega e concatena as sessões de treino (01T, 02T, 03T).
    2. Realiza validação cruzada no conjunto de treino para LDA e SVM
       (incluindo busca em grade de hiperparâmetros para o SVM).
    3. Treina os modelos finais no conjunto de treino completo (usado
       para geração de figuras e acesso aos filtros CSP ajustados).

    Parameters
    ----------
    subject_id : int
        Identificador do sujeito (1 a 9).
    verbose : bool
        Se True, imprime mensagens de progresso no terminal.

    Returns
    -------
    resultado : dict
        Dicionário com todas as métricas e metadados do experimento para
        este sujeito (ver chaves no corpo da função).
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"SUJEITO {subject_id}")
        print(f"{'=' * 70}")

    # ---------------------------------------------------------------
    # 1. Carregamento das sessões de treino
    # ---------------------------------------------------------------
    caminhos_treino = [
        f"{config.DATASET_DIR}/{config.subject_filename(subject_id, sufixo)}"
        for sufixo in config.TRAIN_SESSION_SUFFIXES
    ]

    if verbose:
        print(f"Carregando sessões de treino: {config.TRAIN_SESSION_SUFFIXES}...")

    epochs_treino = preprocessing.carregar_e_concatenar_sessoes(caminhos_treino)

    if epochs_treino is None or len(epochs_treino) == 0:
        if verbose:
            print(f"  [erro] Nenhuma sessão de treino pôde ser carregada "
                  f"para o sujeito {subject_id}. Pulando este sujeito.")
        return _resultado_vazio(subject_id, motivo="sem dados de treino")

    X_treino = epochs_treino.get_data(copy=True)
    y_treino = epochs_treino.events[:, -1]

    if verbose:
        n_esq = int(np.sum(y_treino == y_treino.min()))
        n_dir = int(np.sum(y_treino == y_treino.max()))
        print(f"  {len(epochs_treino)} épocas de treino carregadas "
              f"({n_esq} mão esquerda / {n_dir} mão direita).")

    # ---------------------------------------------------------------
    # 2. LDA: validação cruzada + treino final
    # ---------------------------------------------------------------
    if verbose:
        print("Treinando e validando classificador LDA...")

    t0 = time.time()
    pipeline_lda = montar_pipeline_lda()
    cv_acc_lda, cv_std_lda, cv_kappa_lda, cv_kstd_lda = validacao_cruzada_treino(
        pipeline_lda, X_treino, y_treino
    )
    pipeline_lda.fit(X_treino, y_treino)
    tempo_lda = time.time() - t0

    if verbose:
        print(f"  LDA -- acurácia (CV): {cv_acc_lda:.2f}% (+/- {cv_std_lda:.2f}%)  "
              f"kappa={cv_kappa_lda:.3f} (+/- {cv_kstd_lda:.3f})  [{tempo_lda:.1f}s]")

    # ---------------------------------------------------------------
    # 3. SVM: busca em grade + validação cruzada + treino final
    # ---------------------------------------------------------------
    if verbose:
        print("Treinando e validando classificador SVM (grid search)...")

    t0 = time.time()
    pipeline_svm_otima, melhores_parametros = ajustar_svm_com_grid_search(X_treino, y_treino)
    cv_acc_svm, cv_std_svm, cv_kappa_svm, cv_kstd_svm = validacao_cruzada_treino(
        pipeline_svm_otima, X_treino, y_treino
    )
    tempo_svm = time.time() - t0

    if verbose:
        print(f"  SVM -- melhores hiperparâmetros: {melhores_parametros}")
        print(f"  SVM -- acurácia (CV): {cv_acc_svm:.2f}% (+/- {cv_std_svm:.2f}%)  "
              f"kappa={cv_kappa_svm:.3f} (+/- {cv_kstd_svm:.3f})  [{tempo_svm:.1f}s]")

    return {
        "subject_id": subject_id,
        "n_epocas_treino": len(epochs_treino),
        # LDA
        "lda_cv_acuracia": cv_acc_lda,
        "lda_cv_desvio": cv_std_lda,
        "lda_cv_kappa": cv_kappa_lda,
        "lda_cv_kappa_desvio": cv_kstd_lda,
        "pipeline_lda": pipeline_lda,
        # SVM
        "svm_cv_acuracia": cv_acc_svm,
        "svm_cv_desvio": cv_std_svm,
        "svm_cv_kappa": cv_kappa_svm,
        "svm_cv_kappa_desvio": cv_kstd_svm,
        "svm_melhores_parametros": melhores_parametros,
        "pipeline_svm": pipeline_svm_otima,
        # Dados brutos (úteis para as figuras geradas em visualization.py)
        "X_treino": X_treino,
        "y_treino": y_treino,
        "epochs_treino": epochs_treino,
        "erro": None,
    }


def _resultado_vazio(subject_id: int, motivo: str) -> dict:
    """Retorna um dicionário de resultado "vazio" com todos os campos
    numéricos como NaN, usado quando um sujeito não pôde ser processado
    (ex.: arquivos ausentes). Mantém a estrutura do dicionário consistente
    para que o restante do pipeline (agregação de resultados, tabelas,
    etc.) não precise tratar este caso de forma especial em todo lugar.
    """
    return {
        "subject_id": subject_id,
        "n_epocas_treino": 0,
        "lda_cv_acuracia": np.nan,
        "lda_cv_desvio": np.nan,
        "lda_cv_kappa": np.nan,
        "lda_cv_kappa_desvio": np.nan,
        "lda_matriz_confusao": None,
        "pipeline_lda": None,
        "svm_cv_acuracia": np.nan,
        "svm_cv_desvio": np.nan,
        "svm_cv_kappa": np.nan,
        "svm_cv_kappa_desvio": np.nan,
        "svm_matriz_confusao": None,
        "svm_melhores_parametros": None,
        "pipeline_svm": None,
        "X_treino": None,
        "y_treino": None,
        "epochs_treino": None,
        "erro": motivo,
    }
