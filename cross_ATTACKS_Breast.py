import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold
from joblib import Parallel, delayed

# Imports locais
from TIMBERSTRIKE.timberstrike_lgb import TimberStrikeLightGBM
from metrics.metrics import calcular_ra, calcDistanceReconstruct, createScatterPlotReconstruction
from REBOOT.REBOOT import REBOOT

SEED = 42

# 1. Função de Carregamento (Mantida igual)
def load_target_dataset(dataset_name):
    # ... [Mesmo código da versão anterior para carregar os 4 datasets] ...
    print(f"\n[{dataset_name.upper()}] Carregando dataset...")
    if dataset_name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
    elif dataset_name == "pima_diabetes":
        data = fetch_openml(data_id=37, as_frame=True, parser='auto')
        X, y = data.data, data.target
        y = (y == 'tested_positive').astype(int)
    elif dataset_name == "magic_gamma":
        data = fetch_openml(data_id=1120, as_frame=True, parser='auto')
        X, y = data.data, data.target
        y = (y == 'h').astype(int) 
    elif dataset_name == "santander":
        if not os.path.exists("train.csv"):
            raise FileNotFoundError("Arquivo 'train.csv' não encontrado.")
        df = pd.read_csv("train.csv")
        X = df.drop(columns=['ID_code', 'target'])
        y = df['target']

    elif dataset_name == "nomao":
        # Baixa diretamente do OpenML (ID 1486)
        data = fetch_openml(data_id=1486, as_frame=True, parser='auto')
        X, y = data.data, data.target
        # Converte as classes ('1' e '2') para formato binário padrão 0 e 1
        y = (y == '1').astype(int)
    else:
        raise ValueError("Dataset desconhecido.")
        
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, y


# 2. O "Worker" - Função que será executada em paralelo
def process_fold(dataset_name, epsilon, fold, train_idx, test_idx, X, y, plot_dir, feature_names):
    print(f"[{dataset_name} | Eps: {epsilon}] Iniciando Fold {fold}...")
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    numerical_ranges = {col: [np.min(X[col]), np.max(X[col])] for col in feature_names}

    lgb_params = {
        "boosting_type": "gbdt", "objective": "binary", "metric": "binary_logloss",
        "num_leaves": 31, "max_depth": 6, "num_iterations": 50, "my_n_trees": 50,
        "lambda_l2": 0.1, "bagging_freq": 1, "bagging_fraction": 0.5, "max_bin": 64, 
        "balance_partition": 1, "geo_clip": 1, 'learning_rate': 0.05,
        "verbose": -1, "seed": SEED,
        "n_jobs": 1  # CRÍTICO: Impede que o LGBM brigue por núcleos com o Joblib
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    booster = lgb.train(lgb_params, train_data)

    f1 = f1_score(y_test, (booster.predict(X_test)>0.5).astype(int))

    X_original = np.hstack([X_train.values, y_train.values.reshape(-1, 1)])

    # --- ATAQUE 1: TimberStrike ---
    attacker_ts = TimberStrikeLightGBM(
        booster=booster, n_features=X_train.shape[1], feature_bounds=numerical_ranges,
        learning_rate=0.3, reg_lambda=1.0, milp_time_limit=600, verbose=False,
    )
    X_rec_ts, y_rec_ts = attacker_ts.attack()
    X_reconstruido_ts = np.hstack([X_rec_ts, y_rec_ts.reshape(-1, 1)])

    ra_ts, _ = calcular_ra(X_original, X_reconstruido_ts, [])
    dist_matrix_ts = calcDistanceReconstruct(pd.DataFrame(X_original), pd.DataFrame(X_reconstruido_ts))
    mean_dist_ts = np.mean(np.min(dist_matrix_ts, axis=1))

    # ATENÇÃO: Certifique-se de que a sua função `createScatterPlotReconstruction`
    # não está usando `n_jobs=-1` no TSNE, pelo mesmo motivo do LightGBM.
    createScatterPlotReconstruction(X_original, X_reconstruido_ts, f"{plot_dir}/TS_PCA_fold_{fold}", method='pca',name_datastet="PIMA")
    createScatterPlotReconstruction(X_original, X_reconstruido_ts, f"{plot_dir}/TS_TSNE_fold_{fold}", method='tsne',name_datastet="PIMA")

    # --- ATAQUE 2: REBOOT ---
    attacker_reboot = REBOOT(
        model=booster, N=X_train.shape[0], num_features=X_train.shape[1],
        num_categoricas=0, num_continuas=X_train.shape[1], feature_ranges=numerical_ranges
    )
    attacker_reboot.phase_1()
    attacker_reboot.phase_2()
    attacker_reboot.phase_3()

    trees_info = attacker_reboot.model_dump.get('tree_info', [])
    ra_reboot_val, mean_dist_reboot = np.nan, np.nan

    if len(trees_info) > 1:
        amostras_reconstruidas = attacker_reboot.phase_4()
        ra_reboot_val, _ = calcular_ra(X_original, amostras_reconstruidas, [])
        dist_matrix_reboot = calcDistanceReconstruct(pd.DataFrame(X_original), pd.DataFrame(amostras_reconstruidas))
        mean_dist_reboot = np.mean(np.min(dist_matrix_reboot, axis=1))

        createScatterPlotReconstruction(X_original, amostras_reconstruidas, f"{plot_dir}/REBOOT_PCA_fold_{fold}", method='pca',name_datastet="PIMA")
        createScatterPlotReconstruction(X_original, amostras_reconstruidas, f"{plot_dir}/REBOOT_TSNE_fold_{fold}", method='tsne',name_datastet="PIMA")

    # Retorna os resultados deste fold específico
    return {
        "Dataset": dataset_name, "Epsilon": epsilon, "f1_score":f1,"Fold": fold,
        "RA_TS": ra_ts, "Distance_TS": mean_dist_ts,
        "RA_REBOOT": ra_reboot_val, "Distance_REBOOT": mean_dist_reboot
    }

# 3. Execução Principal (Pipeline)
if __name__ == "__main__":
    datasets_to_run = ["breast_cancer"]
    epsilons = [0.5, 1.0, 5.0]
    kf = RepeatedKFold(n_repeats=2, n_splits=2, random_state=SEED)
    
    # NÚMERO DE JOBS: Ajuste conforme sua máquina. 
    N_JOBS = 3

    all_results = []
    os.makedirs("results", exist_ok=True)

    for dataset_name in datasets_to_run:
        try:
            X, y = load_target_dataset(dataset_name)
        except Exception as e:
            print(f"Pulando {dataset_name} devido a erro: {e}")
            continue
            
        feature_names = X.columns.tolist()
        
        # Cria uma lista de "tarefas" para este dataset
        tasks = []
        for epsilon in epsilons:
            plot_dir = f"plots/{dataset_name}/eps_{epsilon}"
            os.makedirs(plot_dir, exist_ok=True)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                # Adiciona os parâmetros de cada execução na lista de tarefas
                tasks.append((dataset_name, epsilon, fold, train_idx, test_idx, X, y, plot_dir, feature_names))

        print(f"\nDisparando {len(tasks)} tarefas em paralelo para {dataset_name} (Usando {N_JOBS} processos)...")
        
        # Executa as tarefas em paralelo
        dataset_results = Parallel(n_jobs=N_JOBS)(
            delayed(process_fold)(*task) for task in tasks
        )
        
        # Adiciona os resultados deste dataset à lista geral
        all_results.extend(dataset_results)

    # Salva o CSV consolidado
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results/DPBOOST_ATTACK_CONSOLIDATED_PIMA_FIXED.csv", index=False)
    print("\nExecução concluída e paralelizada! Resultados salvos em CSV.")