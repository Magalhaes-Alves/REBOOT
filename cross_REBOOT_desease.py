import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, ParameterGrid
from REBOOT.REBOOT import REBOOT
from metrics.metrics import calcular_ra
from tests.utils import formatar_saida_ataque

# ==========================================
# 1. Preparação dos Dados
# ==========================================
print("=== Preparando o Ambiente ===")
data = load_breast_cancer()
X = data.data
y = data.target

# Mantendo o subconjunto pequeno para agilizar o Gurobi (Fase 4)
X = X[:150, :5]
y = y[:150]

# ==========================================
# 2. Configuração do Experimento (Grid e CV)
# ==========================================
# Defina aqui os hiperparâmetros que deseja explorar
param_grid = {
    'num_boost_round': [10, 50],
    'num_leaves': [3, 5, 8],
    'max_depth': [-1, 8], # -1 permite crescimento irrestrito baseado no num_leaves
    'learning_rate': [0.05, 0.1],
    'min_data_in_leaf': [1, 3], # 1 é o pior cenário de privacidade
    'lambda_l2': [0.0, 0.1]
}

n_splits = 3  # Número de folds para o Cross-Validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lista para armazenar os resultados detalhados
resultados_experimento = []

print(f"Total de combinações de hiperparâmetros a testar: {len(list(ParameterGrid(param_grid)))}\n")

# ==========================================
# 3. Loop de Validação Cruzada sobre o Grid
# ==========================================
for idx_param, params_comb in enumerate(ParameterGrid(param_grid)):
    print(f"--- Testando Combinação {idx_param + 1} ---")
    print(f"Parâmetros: {params_comb}")
    
    # Separar os parâmetros de treinamento do LightGBM
    num_boost_round = params_comb.pop('num_boost_round')
    eta = params_comb['learning_rate']
    reg_lambda = params_comb['lambda_l2']
    
    # Parâmetros base do LightGBM combinados com o Grid atual
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'lambda_l1': 0.0,
        'boost_from_average': True,
        'verbose': -1,
        'bagging_fraction': 1.0,
        'feature_fraction': 1.0,
        **params_comb # Desempacota num_leaves, max_depth, learning_rate, etc.
    }
    
    ra_folds = [] # Armazena o RA de cada fold para esta configuração
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        N_total, M = X_train.shape
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Treinando o Modelo Alvo
        model = lgb.train(lgb_params, train_data, num_boost_round=num_boost_round)
        
        # Iniciando o Ataque
        attacker = REBOOT(model, N_total, M, num_categoricas=0, num_continuas=M)
        
        # Fases do ataque
        try:
            y_bar_rec = attacker.phase_1()
            marginais = attacker.phase_2()
            folhas = attacker.phase_3(eta=eta, reg_lambda=reg_lambda)
            
            trees_info = attacker.model_dump.get('tree_info', [])
            
            # Precisamos de árvores suficientes para a reconstrução (além do bias)
            if len(trees_info) > 2:
                amostras_reconstruidas = attacker.phase_4()
                
                # Cálculo das métricas
                X_train_real = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
                amostras_reconstruidas_formatadas = formatar_saida_ataque(amostras_reconstruidas, X_train_real)
                
                ra = calcular_ra(X_train_real, amostras_reconstruidas_formatadas, [])
                ra_folds.append(ra)
            else:
                # Modelo não cresceu o suficiente nesta configuração
                ra_folds.append(np.nan)
                
        except Exception as e:
            print(f"  [!] Erro no ataque durante o fold {fold}: {e}")
            ra_folds.append(np.nan)
            
    # Recolocando o num_boost_round para fins de registro no DataFrame final
    params_comb['num_boost_round'] = num_boost_round
    
    # Consolidando o resultado da validação cruzada para esta combinação
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
print("\n=== Resultados Finais Consolidados ===")
df_resultados = pd.DataFrame(resultados_experimento)

# Ordenando do maior RA (ataque mais bem sucedido) para o menor
df_resultados = df_resultados.sort_values(by='ra_medio', ascending=False).reset_index(drop=True)

print(df_resultados.to_string())

# Opcional: Salvar em CSV para análise futura ou inclusão no artigo/dissertação
df_resultados.to_csv('resultados_ataque_reboot_cv_2.csv', index=False)