import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, ParameterGrid
from REBOOT.REBOOT import REBOOT
from metrics.metrics import calcular_ra
from tests.utils import formatar_saida_ataque


print("=== Preparando o Ambiente ===")
data = load_breast_cancer()
X = data.data
y = data.target




param_grid = {
    'num_boost_round': [10, 50],
    'num_leaves': [3, 5, 8],
    'max_depth': [-1, 8], # -1 permite crescimento irrestrito baseado no num_leaves
    'learning_rate': [0.05, 0.1],
    'min_data_in_leaf': [1, 3], # 1 é o pior cenário de privacidade
    'lambda_l2': [0.0, 0.1]
}

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

resultados_timberstrike =[]
resultados_reboot =[]





