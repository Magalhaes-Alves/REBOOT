"""
Cross-validation experiment for the TimberStrike attack against LightGBM,
usando exclusivamente a API ORIGINAL de ``metrics.py`` (sem parâmetros
novos).  Compõe ``criar_mapa_tolerancia`` + ``emparelhar_reconstrucao_referencia``
diretamente para obter (a) a Reconstruction Accuracy das features e
(b) os índices do Hungarian matching, que reaproveitamos para a Label RA.

⚠ ATENÇÃO ⚠
O ``metrics.py`` original tem um bug em ``_calcular_taxa_erro_amostra``
(``max(1, num_cats)``/``max(1, num_conts)`` aplicados antes do cálculo de
``erro_total``).  Quando o dataset é 100% contínuo (Breast Cancer é!), o
denominador fica inflado em 1 e a RA reportada fica artificialmente
*alta* (uma reconstrução totalmente errada apareceria como ~16.7% em vez
de 0%).  Veja o final deste arquivo para o patch de uma única linha.
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, ParameterGrid

# Importações do projeto
from TIMBERSTRIKE.timberstrike_lgb import TimberStrikeLightGBM
from metrics.metrics import (
    criar_mapa_tolerancia,
    emparelhar_reconstrucao_referencia,
)

# ==========================================
# 1. Preparação dos Dados
# ==========================================
print("=== Preparando o Ambiente ===")
data = load_breast_cancer()
X = data.data
y = data.target

X = X[:150, :5]
y = y[:150]

# Subconjunto pequeno para agilizar o solver MILP
# Breast Cancer é 100% numérico
INDICES_CATEGORICOS: list = []

# Multiplicador de tolerância (estilo TabLeak/TimberStrike)
TOL_MULT = 0.319

# ==========================================
# 2. Configuração do Experimento (Grid e CV)
# ==========================================
param_grid = {
    'num_boost_round': [10, 50],
    'num_leaves': [3, 5],
    'max_depth': [-1, 8],
    'learning_rate': [0.1],
    'min_data_in_leaf': [1, 5],
    'lambda_l2': [0.0, 0.1]
}

n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

resultados_experimento = []
arquivo_saida = 'resultados_ataque_timberstrike_cv_balanceado.csv'

total_combinacoes = len(list(ParameterGrid(param_grid)))
print(f"Total de combinações a testar: {total_combinacoes}\n")


# ==========================================
# Helper local: RA das features + índices do Hungarian
# (replica a lógica de ``calcular_ra`` mas usando apenas funções
# públicas que JÁ existem no metrics.py original)
# ==========================================
def calcular_ra_e_indices(X_real, X_rec, indices_categoricos, tol_mult=0.319):
    """Retorna (ra_features, idx_real_emparelhados, idx_rec_emparelhados)."""
    X_real = np.asarray(X_real)
    X_rec = np.asarray(X_rec)

    n_real, n_rec = len(X_real), len(X_rec)

    # Equaliza tamanhos com o mesmo critério do calcular_ra original.
    if n_real != n_rec:
        tamanho_minimo = min(n_real, n_rec)
        np.random.seed(42)
        sub_real = np.random.choice(n_real, size=tamanho_minimo, replace=False)
        sub_rec = np.random.choice(n_rec, size=tamanho_minimo, replace=False)
        X_real_use = X_real[sub_real]
        X_rec_use = X_rec[sub_rec]
    else:
        sub_real = np.arange(n_real)
        sub_rec = np.arange(n_rec)
        X_real_use, X_rec_use = X_real, X_rec

    mapa_tolerancia = criar_mapa_tolerancia(
        X_real_use, indices_categoricos, tol_mult,
    )

    # emparelhar_reconstrucao_referencia(retornar_indices=True) JÁ existe
    # no metrics.py original.
    (_, array_erros, _, _,
     idx_linhas, idx_colunas) = emparelhar_reconstrucao_referencia(
        X_real_use, X_rec_use, mapa_tolerancia,
        retornar_indices=True, base_emparelhamento='all',
    )

    ra = 1.0 - float(np.mean(array_erros))
    # Devolve os índices no espaço ORIGINAL (antes da subamostragem)
    idx_real_final = sub_real[idx_linhas]
    idx_rec_final = sub_rec[idx_colunas]
    return ra, idx_real_final, idx_rec_final


# ==========================================
# 3. Loop de Validação Cruzada
# ==========================================
t0_global = time.time()

for idx_param, params_comb in enumerate(ParameterGrid(param_grid)):
    print(f"--- Combinação {idx_param + 1}/{total_combinacoes} ---")
    print(f"Parâmetros: {params_comb}")

    num_boost_round = params_comb.pop('num_boost_round')
    eta = params_comb['learning_rate']
    reg_lambda = params_comb['lambda_l2']

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'feature_pre_filter': False,
        'verbose': -1,
        'deterministic': True,
        'force_row_wise': True,
        **params_comb,
    }

    ra_features_folds = []
    ra_labels_folds = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        n_train, d_train = X_train.shape

        train_data = lgb.Dataset(X_train, label=y_train)
        booster = lgb.train(lgb_params, train_data, num_boost_round=num_boost_round)

        feature_bounds = [
            (float(np.min(X_train[:, f]) - 0.5), float(np.max(X_train[:, f]) + 0.5))
            for f in range(d_train)
        ]

        attacker = TimberStrikeLightGBM(
            booster=booster,
            n_features=d_train,
            feature_bounds=feature_bounds,
            learning_rate=eta,
            reg_lambda=reg_lambda,
            milp_time_limit=60,
            verbose=False,
        )

        try:
            X_rec, y_rec = attacker.attack()

            if len(X_rec) > 0:
                ra_feat, idx_real, idx_rec = calcular_ra_e_indices(
                    X_real=X_train,
                    X_rec=X_rec,
                    indices_categoricos=INDICES_CATEGORICOS,
                    tol_mult=TOL_MULT,
                )
                ra_features_folds.append(ra_feat)

                # Label RA: usa o MESMO emparelhamento da métrica de features
                label_acc = float(np.mean(y_train[idx_real] == y_rec[idx_rec]))
                ra_labels_folds.append(label_acc)
            else:
                ra_features_folds.append(np.nan)
                ra_labels_folds.append(np.nan)

        except Exception as e:
            print(f"  [!] Erro no ataque (Fold {fold}): {e}")
            ra_features_folds.append(np.nan)
            ra_labels_folds.append(np.nan)

    # Recoloca parâmetros extraídos no dict (para o log)
    params_comb['num_boost_round'] = num_boost_round
    params_comb['learning_rate'] = eta
    params_comb['lambda_l2'] = reg_lambda

    ra_feat_medio = float(np.nanmean(ra_features_folds))
    ra_feat_std = float(np.nanstd(ra_features_folds))
    ra_label_medio = float(np.nanmean(ra_labels_folds))
    ra_label_std = float(np.nanstd(ra_labels_folds))

    print(f"RA Features Médio: {ra_feat_medio * 100:.2f}% (±{ra_feat_std * 100:.2f}%)")
    print(f"RA Labels Médio:   {ra_label_medio * 100:.2f}% (±{ra_label_std * 100:.2f}%)\n")

    resultados_experimento.append({
        **params_comb,
        'ra_features_medio': ra_feat_medio,
        'ra_features_std': ra_feat_std,
        'ra_labels_medio': ra_label_medio,
        'ra_labels_std': ra_label_std,
    })

# ==========================================
# 4. Salvando Resultados Finais
# ==========================================
print(f"\nTempo total: {time.time() - t0_global:.1f}s")
print("=== Resultados Finais Consolidados ===")

df_resultados = pd.DataFrame(resultados_experimento)
df_resultados = df_resultados.sort_values(
    by='ra_features_medio', ascending=False
).reset_index(drop=True)

print(df_resultados.to_string())
df_resultados.to_csv(arquivo_saida, index=False)
print(f"\nResultados salvos com sucesso em: {arquivo_saida}")


# ==========================================================================
# PATCH RECOMENDADO PARA metrics.py
# --------------------------------------------------------------------------
# Em ``_calcular_taxa_erro_amostra``, SUBSTITUIR o bloco
#
#     # Evita divisão por zero
#     num_cats = max(1, num_cats)
#     num_conts = max(1, num_conts)
#     erro_total = (erro_cat + erro_cont) / (num_cats + num_conts)
#
# por
#
#     total_features = num_cats + num_conts
#     erro_total = 0.0 if total_features == 0 \
#                  else (erro_cat + erro_cont) / total_features
#     # max(1, ...) só é necessário no return detalhado:
#     if detalhado:
#         return erro_total, erro_cat / max(1, num_cats), erro_cont / max(1, num_conts)
#     return erro_total
#
# Sem isso, com dataset 100% contínuo (caso do Breast Cancer), os RAs
# ficam ~16.7% inflados (para 5 features), porque o denominador fica
# 1 + nº_contínuas em vez de nº_contínuas.
# ==========================================================================