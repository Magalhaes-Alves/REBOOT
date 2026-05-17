import time
import numpy as np
import pandas as pd
import lightgbm as lgb
# Alterado de load_breast_cancer para load_diabetes
from sklearn.datasets import load_diabetes 
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
print("=== Preparando o Ambiente com Dataset Diabetes ===")
data = load_diabetes()
X = data.data
y = data.target

# Binarização do target para compatibilidade com o ataque (classificação binária)
y = (y > np.mean(y)).astype(int)

# Subconjunto pequeno para agilizar o solver MILP
# Selecionando 100 amostras e as primeiras 5 features


# Dataset Diabetes é 100% numérico (contínuo)
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
arquivo_saida = 'resultados_ataque_timberstrike_diabetes.csv'

total_combinacoes = len(list(ParameterGrid(param_grid)))
print(f"Total de combinações a testar: {total_combinacoes}\n")


# ==========================================
# Helper local: RA das features + índices do Hungarian
# ==========================================
def calcular_ra_e_indices(X_real, X_rec, indices_categoricos, tol_mult=0.319):
    """Retorna (ra_features, idx_real_emparelhados, idx_rec_emparelhados)."""
    X_real = np.asarray(X_real)
    X_rec = np.asarray(X_rec)

    n_real, n_rec = len(X_real), len(X_rec)

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

    (_, array_erros, _, _,
     idx_linhas, idx_colunas) = emparelhar_reconstrucao_referencia(
        X_real_use, X_rec_use, mapa_tolerancia,
        retornar_indices=True, base_emparelhamento='all',
    )

    ra = 1.0 - float(np.mean(array_erros))
    idx_real_final = sub_real[idx_linhas]
    idx_rec_final = sub_rec[idx_colunas]
    return ra, idx_real_final, idx_rec_final


# ==========================================
# 3. Loop de Validação Cruzada
# ==========================================
t0_global = time.time()

for idx_param, params_comb in enumerate(ParameterGrid(param_grid)):
    print(f"--- Combinação {idx_param + 1}/{total_combinacoes} ---")
    
    # Cópia para não corromper o grid
    current_params = params_comb.copy()
    num_boost_round = current_params.pop('num_boost_round')
    eta = current_params['learning_rate']
    reg_lambda = current_params['lambda_l2']

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'feature_pre_filter': False,
        'verbose': -1,
        'deterministic': True,
        'force_row_wise': True,
        **current_params,
    }

    ra_features_folds = []
    ra_labels_folds = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        n_train, d_train = X_train.shape

        train_data = lgb.Dataset(X_train, label=y_train)
        booster = lgb.train(lgb_params, train_data, num_boost_round=num_boost_round)

        # Definição automática de limites baseada no novo dataset
        feature_bounds = [
            (float(np.min(X_train[:, f]) - 0.1), float(np.max(X_train[:, f]) + 0.1))
            for f in range(d_train)
        ]

        attacker = TimberStrikeLightGBM(
            booster=booster,
            n_features=d_train,
            feature_bounds=feature_bounds,
            learning_rate=eta,
            reg_lambda=reg_lambda,
            milp_time_limit=60, # Limite de tempo importante para o TimberStrike
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

                # Label RA
                label_acc = float(np.mean(y_train[idx_real] == y_rec[idx_rec]))
                ra_labels_folds.append(label_acc)
            else:
                ra_features_folds.append(np.nan)
                ra_labels_folds.append(np.nan)

        except Exception as e:
            print(f"  [!] Erro no ataque (Fold {fold}): {e}")
            ra_features_folds.append(np.nan)
            ra_labels_folds.append(np.nan)

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
print("=== Resultados Finais Consolidados (Diabetes) ===")

df_resultados = pd.DataFrame(resultados_experimento)
df_resultados = df_resultados.sort_values(
    by='ra_features_medio', ascending=False
).reset_index(drop=True)

print(df_resultados.to_string())
df_resultados.to_csv(arquivo_saida, index=False)