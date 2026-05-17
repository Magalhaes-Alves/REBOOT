import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from REBOOT.REBOOT import REBOOT
from metrics.metrics import calcular_ra
import pandas as pd
from tests.utils import formatar_saida_ataque

results_ra=[]

for boosting_rounds in [10, 50, 100]:
    for depth in[3,5,8,10]:
        print("=== Preparando o Ambiente ===")

        # 1. Carregar um dataset real, mas pequeno
        data = load_breast_cancer()
        X = data.data
        y = data.target


        # Vamos pegar apenas um subconjunto pequeno de features e amostras 
        # para que o solver Gurobi (Fase 4) rode rapidamente no teste
        X = X[:150, :5]  # 150 amostras, 5 features
        y = y[:150]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        N_total, M = X_train.shape
        print(f"Dataset de Treino: {N_total} amostras, {M} features.")
        print(f"Prevalência real da classe positiva: {np.mean(y_train):.4f}\n")

        # 2. Configurar e treinar o modelo alvo (LightGBM)
        # Usaremos parâmetros que facilitam a memorização (overfitting)
        # Isso demonstra o pior cenário de privacidade: folhas pequenas (min_data_in_leaf = 1)
        train_data = lgb.Dataset(X_train, label=y_train)

        eta = 0.1
        reg_lambda = 0.0

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': eta,
            'lambda_l1': 0.0,
            'lambda_l2': reg_lambda,
            'max_depth': 8,
            'num_leaves': depth,
            'min_data_in_leaf': 0, # Crítico para o ataque: permite folhas de 1 amostra
            'boost_from_average': True, # Necessário para a Fase 1
            'verbose': -1,
            'num_boost_round':50,
            'bagging_fraction': 1.0, # Sem bagging para facilitar a demonstração
            'feature_fraction': 1.0 # Sem subamostragem de features
        }

        print("=== Treinando o Modelo Alvo ===")
        # Treinando com apenas 3 árvores para simplificar a demonstração da Fase 4
        model = lgb.train(params, train_data)
        print("Treinamento concluído.\n")

        # 3. Executando o Ataque GBM-Recon
        print("=== Iniciando Ataque GBM-Recon ===")

        # Instancia o atacante simulando o conhecimento de N_total e M
        attacker = REBOOT(model, N_total, M,num_categoricas=0, num_continuas=M)

        # Fase 1
        y_bar_rec = attacker.phase_1()



        #print(f"Erro na prevalência recuperada: {abs(np.mean(y_train) - y_bar_rec)}")

        # Fase 2
        marginais = attacker.phase_2()

        # Fase 3 (usando os mesmos parâmetros matemáticos do treino)
        # No mundo real, eta e lambda são inferidos ou testados via força bruta
        folhas = attacker.phase_3(eta=eta, reg_lambda=reg_lambda)

        # Fase 4
        # Vamos usar as árvores 1 e 2 para refinar os limites (a árvore 0 é o bias)
        # Nota: No LightGBM real com boost_from_average, tree_info[0] é o intercept,
        # tree_info[1] é a primeira árvore de splits, tree_info[2] é a segunda.
        # indices_arvores_subsequentes = [2] 

        # Verifica se o modelo gerou árvores suficientes
        trees_info = attacker.model_dump.get('tree_info', [])
        if len(trees_info) > 2:
            amostras_reconstruidas = attacker.phase_4()
            
            # Exibir o nível de reconstrução das primeiras 3 amostras
            print("\n=== Resultados da Reconstrução ===")
            for i, amostra in enumerate(amostras_reconstruidas[:3]):
                print(f"Amostra Fictícia {i} | Rótulo Inferido: {amostra['label']}")
                print(f"  Limites de Features (Hipercubo):")
                for feat, limites in amostra['bounds'].items():
                    print(f"    Feature {feat}: [{limites[0]:.4f}, {limites[1]:.4f}]")
        else:
            print("\nO modelo não cresceu árvores suficientes para a Fase 4.")

        #amostras_reconstruidas = attacker._instantiate_samples_from_tree_1()

        # feature_ranges = attacker.get_feature_ranges_from_marginals()
        # X_rec, labels_rec = attacker.materialize_samples(feature_ranges)

        #print(X_rec, labels_rec)


        #print(amostras_reconstruidas)

        # Métricas de Reconstrução 

        X_train_real  = np.concatenate([X_train,y_train.reshape(-1,1)], axis=1)

        amostras_reconstruidas_formatadas = formatar_saida_ataque(amostras_reconstruidas, X_train_real)


        ra= calcular_ra(np.concat([X_train, y_train.reshape(-1,1)], axis=1),amostras_reconstruidas_formatadas, [],)



        print(f"RA: {ra}")

        results_ra.append((boosting_rounds,depth,ra))


print("\n=== Resultados Finais ===")

print(results_ra)

        





