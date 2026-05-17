import numpy as np
import pandas as pd
import lightgbm as lgb
# Alterado de load_breast_cancer para load_diabetes
from sklearn.datasets import load_diabetes 
from sklearn.model_selection import KFold, ParameterGrid
from REBOOT.REBOOT import REBOOT
from metrics.metrics import calcular_ra
from tests.utils import formatar_saida_ataque

# ==========================================
# 1. Preparação dos Dados
# ==========================================
print("=== Preparando o Ambiente com Dataset Diabetes ===")
# O dataset diabetes é composto por 10 variáveis numéricas (idade, IMC, pressão, etc.)
data = load_diabetes()
X = data.data
y = data.target

# Binarização do target: transformamos em 1 se estiver acima da média, caso contrário 0
# Isso mantém a compatibilidade com o lgb_params['objective'] = 'binary'
y = (y > np.mean(y)).astype(int)

# Mantendo o subconjunto pequeno para agilizar o Gurobi (Fase 4)
# Selecionamos as primeiras 5 features e 100 amostras para maior performance


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

print(f"Total de combinações de hiperparâmetros a testar: {len(list(ParameterGrid(param_grid)))}\n")

# ==========================================
# 3. Loop de Validação Cruzada sobre o Grid
# ==========================================
for idx_param, params_comb in enumerate(ParameterGrid(param_grid)):
    print(f"--- Testando Combinação {idx_param + 1} ---")
    
    # Cópia para não modificar o grid original durante o loop
    current_params = params_comb.copy()
    num_boost_round = current_params.pop('num_boost_round')
    eta = current_params['learning_rate']
    reg_lambda = current_params['lambda_l2']
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'lambda_l1': 0.0,
        'boost_from_average': True,
        'verbose': -1,
        **current_params 
    }
    
    ra_folds = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        N_total, M = X_train.shape
        train_data = lgb.Dataset(X_train, label=y_train)
        
        model = lgb.train(lgb_params, train_data, num_boost_round=num_boost_round)
        
        # O REBOOT é configurado com num_categoricas=0 conforme solicitado
        attacker = REBOOT(model, N_total, M, num_categoricas=0, num_continuas=M)
        
        try:
            y_bar_rec = attacker.phase_1()
            marginais = attacker.phase_2()
            folhas = attacker.phase_3(eta=eta, reg_lambda=reg_lambda)
            
            trees_info = attacker.model_dump.get('tree_info', [])
            
            if len(trees_info) > 2:
                amostras_reconstruidas = attacker.phase_4()
                
                X_train_real = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
                amostras_reconstruidas_formatadas = formatar_saida_ataque(amostras_reconstruidas, X_train_real)
                
                ra = calcular_ra(X_train_real, amostras_reconstruidas_formatadas, [])
                ra_folds.append(ra)
            else:
                ra_folds.append(np.nan)
                
        except Exception as e:
            print(f"  [!] Erro no ataque durante o fold {fold}: {e}")
            ra_folds.append(np.nan)
            
    ra_medio = np.nanmean(ra_folds)
    ra_std = np.nanstd(ra_folds)
    
    print(f"RA Médio da combinação: {ra_medio:.4f} (±{ra_std:.4f})\n")
    
    resultados_experimento.append({
        **params_comb,
        'ra_medio': ra_medio,
        'ra_std': ra_std
    })

# ==========================================
# 4. Resultados Finais
# ==========================================
df_resultados = pd.DataFrame(resultados_experimento)
df_resultados = df_resultados.sort_values(by='ra_medio', ascending=False).reset_index(drop=True)

print("\n=== Resultados Finais (Dataset Diabetes) ===")
print(df_resultados.to_string())

df_resultados.to_csv('resultados_ataque_reboot_diabetes.csv', index=False)