import lightgbm as lgb
import numpy as np


def train_dpboost(X, y, **kwargs):
    
    train_data = lgb.Dataset(X, label=y)


    params = {
        **kwargs
    }

    print("Treinamento do modelo com parâmetros:", params)

    lgb_model = lgb.train(params, train_data)

    print("Modelo treinado com sucesso.")

    return lgb_model



def formatar_saida_ataque(saida_ataque: list, X_train: np.ndarray) -> np.ndarray:
    """
    Converte a saída do ataque em uma matriz NumPy.
    - Resolve intervalos [min, max] pegando o ponto médio.
    - Resolve limites infinitos (-inf ou inf) usando o min/max real do X_train.
    - Preenche features desconhecidas sorteando um valor aleatório da respectiva coluna no X_train.
    - Coloca o 'label' na última coluna.
    """
    # Estatísticas de referência para tratar os infinitos
    minimos_reais = np.min(X_train, axis=0)
    maximos_reais = np.max(X_train, axis=0)
    
    num_features = X_train.shape[1]
    matriz_reconstruida = []
    
    for amostra in saida_ataque:
        # Cria uma linha vazia com espaço para as features + 1 espaço para o label no final
        linha = np.zeros(num_features + 1)
        bounds = amostra.get('bounds', {})
        
        for f_idx in range(num_features):
            # Se o atacante descobriu os limites da feature
            if f_idx in bounds:
                limite_inferior, limite_superior = bounds[f_idx]
                
                if limite_inferior == -float('inf'):
                    limite_inferior = minimos_reais[f_idx]
                if limite_superior == float('inf'):
                    limite_superior = maximos_reais[f_idx]
                    
                # O palpite é o ponto médio do intervalo descoberto
                linha[f_idx] = (limite_inferior + limite_superior) / 2.0
                
            # Se o atacante NÃO descobriu a feature, faz amostragem aleatória
            else:
                linha[f_idx] = np.random.choice(X_train[:, f_idx])
                
        # Adiciona o Label (Y) na última coluna
        linha[-1] = amostra.get('label', 0)
        
        matriz_reconstruida.append(linha)
        
    return np.array(matriz_reconstruida)
    
    