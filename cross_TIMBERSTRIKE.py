import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, ParameterGrid
from scipy.optimize import linear_sum_assignment

# Importando do seu arquivo timberstrike_lgb.py
from TIMBERSTRIKE.timberstrike_lgb import TimberStrikeLightGBM, reconstruction_accuracy

# ==========================================
# 1. Preparação dos Dados
# ==========================================
print("=== Preparando o Ambiente ===")
data = load_breast_cancer()
X = data.data
y = data.target

# Subconjunto pequeno para agilizar o solver MILP do TimberStrike
X = X[:150, :5]
y = y[:150]

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
arquivo_saida = 'resultados_ataque_timberstrike_cv.csv'
tol = 0.05 # Tolerância para a métrica de reconstrução das features

print(f"Total de combinações a testar: {len(list(ParameterGrid(param_grid)))}\n")

# ==========================================
# 3. Loop de Validação Cruzada
# ==========================================
for idx_param, params_comb in enumerate(ParameterGrid(param_grid)):
    print(f"--- Testando Combinação {idx_param + 1} ---")
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
        **params_comb 
    }
    
    ra_features_folds = []
    ra_labels_folds = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        n_train, d_train = X_train.shape
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Treinando o modelo
        booster = lgb.train(lgb_params, train_data, num_boost_round=num_boost_round)
        
        # O TimberStrike exige os limites (bounds) das features para inicializar o ataque
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
            milp_time_limit=60, # Limite de 60s por árvore para não travar o teste
            verbose=False
        )
        
        try:
            # Executando o ataque
            X_rec, y_rec = attacker.attack()
            
            if len(X_rec) > 0:
                # 1. Acurácia de Reconstrução das Features (RA)
                acc_feat, _ = reconstruction_accuracy(
                    X_train, X_rec, tol=tol, feature_ranges=feature_bounds
                )
                ra_features_folds.append(acc_feat)
                
                # 2. Acurácia de Reconstrução dos Rótulos (Label RA)
                spans = np.array([hi - lo for lo, hi in feature_bounds])
                cost = np.zeros((n_train, len(X_rec)))
                for i in range(n_train):
                    cost[i] = np.sum(np.abs(X_rec / spans - (X_train[i] / spans)), axis=1)
                
                row, col = linear_sum_assignment(cost)
                k = min(len(row), n_train, len(X_rec))
                label_acc = float(np.mean(y_train[row[:k]] == y_rec[col[:k]]))
                ra_labels_folds.append(label_acc)
            else:
                ra_features_folds.append(np.nan)
                ra_labels_folds.append(np.nan)
                
        except Exception as e:
            print(f"  [!] Erro no ataque (Fold {fold}): {e}")
            ra_features_folds.append(np.nan)
            ra_labels_folds.append(np.nan)
            
    # Recolocando parâmetros extraídos para salvar no log
    params_comb['num_boost_round'] = num_boost_round
    params_comb['learning_rate'] = eta
    params_comb['lambda_l2'] = reg_lambda
    
    # Consolidando resultados
    ra_feat_medio = np.nanmean(ra_features_folds)
    ra_feat_std = np.nanstd(ra_features_folds)
    ra_label_medio = np.nanmean(ra_labels_folds)
    ra_label_std = np.nanstd(ra_labels_folds)
    
    print(f"RA Features Médio: {ra_feat_medio * 100:.2f}% (±{ra_feat_std * 100:.2f}%)")
    print(f"RA Labels Médio:   {ra_label_medio * 100:.2f}% (±{ra_label_std * 100:.2f}%)\n")
    
    resultados_experimento.append({
        **params_comb,
        'ra_features_medio': ra_feat_medio,
        'ra_features_std': ra_feat_std,
        'ra_labels_medio': ra_label_medio,
        'ra_labels_std': ra_label_std
    })

# ==========================================
# 4. Salvando Resultados Finais
# ==========================================
print("=== Resultados Finais Consolidados ===")
df_resultados = pd.DataFrame(resultados_experimento)

# Ordenando do ataque mais bem-sucedido nas features para o menor
df_resultados = df_resultados.sort_values(by='ra_features_medio', ascending=False).reset_index(drop=True)

print(df_resultados.to_string())

# Salvando no arquivo CSV
df_resultados.to_csv(arquivo_saida, index=False)
print(f"\nResultados salvos com sucesso em: {arquivo_saida}")