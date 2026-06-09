import numpy as np
from scipy.optimize import linear_sum_assignment
import lightgbm as lgb
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plt


#Biblioteca que adapta as métricas de attack adaptado do Timberstrike
# - Reconstruction accuracy
# - Distância de pares
#

def criar_mapa_tolerancia(X_real: np.ndarray, indices_categoricos: list, tol_multiplicador: float = 0.319) -> list:
    
    desvios_padrao = np.std(X_real, axis=0)
    indices_continuos = [
        i for i in range(X_real.shape[1]) if i not in indices_categoricos
    ]
    desvios_continuos = desvios_padrao[indices_continuos]

    mapa_tolerancia = []
    ponteiro_continuo = 0

    for i in range(X_real.shape[1]):
        if i in indices_continuos:
            mapa_tolerancia.append(
                tol_multiplicador * desvios_continuos[ponteiro_continuo]
            )
            ponteiro_continuo += 1
        else:
            mapa_tolerancia.append("cat")

    return mapa_tolerancia


def _calcular_taxa_erro_amostra(
    amostra_real, amostra_reconstruida, mapa_tolerancia, detalhado=False
):
    
    erro_cat = 0
    erro_cont = 0
    num_cats = 0
    num_conts = 0

    for valor_real, valor_rec, tol in zip(amostra_real, amostra_reconstruida, mapa_tolerancia):

        if tol == "cat":
            erro_cat += 0 if str(valor_real) == str(valor_rec) else 1
            num_cats += 1
        elif not isinstance(tol, str):
            # Verifica se o valor reconstruído está dentro do intervalo [valor_real - tol, valor_real + tol]
            dentro_da_tolerancia = (
                (float(valor_real) - tol)
                <= float(valor_rec)
                <= (float(valor_real) + tol)
            )
            erro_cont += 0 if dentro_da_tolerancia else 1
            num_conts += 1
        else:
            raise TypeError(
                "O mapa de tolerância deve conter números ou a string 'cat'."
            )

    # Evita divisão por zero
    num_cats = max(1, num_cats)
    num_conts = max(1, num_conts)

    erro_total = (erro_cat + erro_cont) / (num_cats + num_conts)

    if detalhado:
        return erro_total, (erro_cat / num_cats), (erro_cont / num_conts)
    return erro_total


def calcular_taxa_erro_lote(
    X_real: np.ndarray,
    X_rec: np.ndarray,
    mapa_tolerancia: list,
    detalhado: bool = False,
):
    
    assert X_real.shape == X_rec.shape, (
        "Os dados reais e reconstruídos devem ter o mesmo formato."
    )

    erro_total_lote = 0
    erro_cat_lote = 0
    erro_cont_lote = 0
    num_amostras = X_real.shape[0]

    for linha_real, linha_rec in zip(X_real, X_rec):
        resultado = _calcular_taxa_erro_amostra(linha_real, linha_rec, mapa_tolerancia, detalhado=detalhado)
        if detalhado:
            erro_total_lote += resultado[0] / num_amostras
            erro_cat_lote += resultado[1] / num_amostras
            erro_cont_lote += resultado[2] / num_amostras
        else:
            erro_total_lote += resultado / num_amostras

    if detalhado:
        return erro_total_lote, erro_cat_lote, erro_cont_lote
    return erro_total_lote


def emparelhar_reconstrucao_referencia(
    lote_alvo: np.ndarray,
    lote_reconstruido: np.ndarray,
    mapa_tolerancia: list,
    retornar_indices: bool = False,
    base_emparelhamento: str = "all",
):
    
    assert base_emparelhamento in ["all", "cat", "cont"], ("Escolha válida: 'all', 'cat' ou 'cont'.")

    num_amostras = lote_alvo.shape[0]

    # Matrizes de custo (erro) para cada combinação possível de [linha_real x linha_reconstruida]
    matriz_custo_todos = np.zeros((num_amostras, num_amostras))
    matriz_custo_cat = np.zeros((num_amostras, num_amostras))
    matriz_custo_cont = np.zeros((num_amostras, num_amostras))

    # Preenche as matrizes de custo
    for i, linha_real in enumerate(lote_alvo):
        for j, linha_rec in enumerate(lote_reconstruido):
            custo_todos, custo_cat, custo_cont = _calcular_taxa_erro_amostra(linha_real, linha_rec, mapa_tolerancia, detalhado=True)
            matriz_custo_todos[i, j] = custo_todos
            matriz_custo_cat[i, j] = custo_cat
            matriz_custo_cont[i, j] = custo_cont

    # Aplica o Algoritmo Húngaro para encontrar a melhor combinação
    if base_emparelhamento == "all":
        indices_linhas, indices_colunas = linear_sum_assignment(matriz_custo_todos)
    elif base_emparelhamento == "cat":
        indices_linhas, indices_colunas = linear_sum_assignment(matriz_custo_cat)
    else:
        indices_linhas, indices_colunas = linear_sum_assignment(matriz_custo_cont)

    # Reordena o lote reconstruído e pega os menores custos de erro
    lote_rec_reordenado = lote_reconstruido[indices_colunas]
    custo_final_todos = matriz_custo_todos[indices_linhas, indices_colunas]
    custo_final_cat = matriz_custo_cat[indices_linhas, indices_colunas]
    custo_final_cont = matriz_custo_cont[indices_linhas, indices_colunas]

    if retornar_indices:
        return (
            lote_rec_reordenado,
            custo_final_todos,
            custo_final_cat,
            custo_final_cont,
            indices_linhas,
            indices_colunas,
        )
    return lote_rec_reordenado, custo_final_todos, custo_final_cat, custo_final_cont


def obter_k_features_menos_importantes_lightgbm(
    modelo, k: int, nomes_features: list = None, tipo_importancia: str = "split"
) -> list:
    """
    Retorna as 'k' features menos importantes de um modelo LightGBM treinado.

    Parâmetros:
    - modelo: O modelo LightGBM treinado (pode ser o lgb.LGBMClassifier da API Scikit-Learn ou lgb.Booster).
    - k: Número de features menos importantes a retornar.
    - nomes_features: Lista opcional com os nomes das colunas. Se omitido, tenta extrair do próprio modelo.
    - tipo_importancia: 'split' (frequência de uso) ou 'gain' (ganho de informação).

    Retorna:
    - Lista com os nomes das 'k' features menos importantes.
    """
    # 1. Padroniza o acesso ao objeto Booster do LightGBM
    if hasattr(modelo, "booster_"):
        # Caso o usuário tenha usado a API do Scikit-Learn (lgb.LGBMClassifier ou LGBMRegressor)
        booster = modelo.booster_
    elif isinstance(modelo, lgb.Booster):
        # Caso o usuário tenha usado a API nativa de treinamento do LightGBM
        booster = modelo
    else:
        raise ValueError("O modelo fornecido não é um modelo LightGBM válido.")

    # 2. Extrai os nomes das features (se não foram fornecidos)
    if nomes_features is None:
        nomes_features = booster.feature_name()

    # 3. Extrai as pontuações de importância
    # Ao contrário do XGBoost que retorna um dicionário, o LightGBM retorna um array NumPy
    importancias = booster.feature_importance(importance_type=tipo_importancia)

    # 4. Combina os nomes com as pontuações em um dicionário
    mapa_importancia = {nome: imp for nome, imp in zip(nomes_features, importancias)}

    # 5. Ordena de forma crescente (do menos importante para o mais importante)
    menos_importantes = sorted(mapa_importancia.items(), key=lambda x: x[1])[:k]

    return [feature[0] for feature in menos_importantes]




def calcDistanceReconstruct(X_original: pd.DataFrame, X_reconstruido: pd.DataFrame, metric= 'euclidean'):


    distances = pairwise_distances(X_reconstruido, X_original, metric=metric)

    return distances
    

def createScatterPlotReconstruction(X_original, X_reconstruido, name_file="Reconstruct", seed=42, method='pca', name_datastet= '-'):
    
    # 1. Validação de método e Nomenclatura Semântica
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=seed)
    elif method == 'tsne':
        # n_jobs=-1 acelera o t-SNE usando todos os núcleos do processador
        reducer = TSNE(n_components=2, random_state=seed, n_jobs=1) 
    else:
        raise ValueError("O método deve ser 'pca' ou 'tsne'.")

    # 2. Remoção do .copy() para poupar memória (np.concatenate já cria uma nova matriz)
    X = np.concatenate([X_original, X_reconstruido], axis=0)
    y = np.concatenate([np.zeros(len(X_original)), np.ones(len(X_reconstruido))])

    # Redução de dimensionalidade
    x_embedded = reducer.fit_transform(X)

    # 3. Melhorias Visuais no Plot
    plt.figure(figsize=(10, 8))

    # Uso de cores hexadecimais padrão (mais elegantes), ajuste no tamanho (s) e remoção da borda dos pontos
    plt.scatter(x_embedded[y == 0, 0], x_embedded[y == 0, 1], label="Original", c="#1f77b4", alpha=0.6, edgecolors='none', s=30)
    plt.scatter(x_embedded[y == 1, 0], x_embedded[y == 1, 1], label="Reconstructed", c="#d62728", alpha=0.6, edgecolors='none', s=30)

    # Adição de títulos, rótulos e grade
    plt.title(f"Data Reconstruction Analysis - {method.upper()} in {name_datastet}", fontsize=14)
    plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
    plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
    plt.legend(loc="best", framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)


    # 5. Salvando com Alta Resolução (300 DPI) e removendo bordas em branco
    plt.savefig(f"{name_file}.png", dpi=300, bbox_inches='tight')
    plt.close()



def calcular_ra(X_real: np.ndarray,X_rec: np.ndarray,indices_categoricos: list,tol_multiplicador: float = 0.319,) -> float:
    """
    Função principal e de alto nível que deve ser chamada para calcular o
    Reconstruction Accuracy (RA) do ataque TimberStrike.

    Retorna a Acurácia de Reconstrução em porcentagem (0 a 100%).
    """

    X_real = np.asarray(X_real)
    X_rec = np.asarray(X_rec)


    n_real = len(X_real)
    n_rec = len(X_rec)

    # Se os tamanhos forem diferentes (pode ocorrer dependendo do sistema FL atacado),
    # pegamos uma amostra aleatória igualitária (lower bound da acurácia)

    print(n_real, n_rec)
    if n_real != n_rec:
        tamanho_minimo = min(n_real, n_rec)
        np.random.seed(42)
        idx_real = np.random.choice(n_real, size=tamanho_minimo, replace=False)
        idx_rec = np.random.choice(n_rec, size=tamanho_minimo, replace=False)
        X_real = X_real[idx_real]
        X_rec = X_rec[idx_rec]
        print(
            f"[Aviso] Dados com tamanhos diferentes. Selecionadas {tamanho_minimo} amostras de cada."
        )

    # 1. Cria o mapa de tolerância
    mapa_tolerancia = criar_mapa_tolerancia(
        X_real, indices_categoricos, tol_multiplicador
    )

    # 2. Emparelha os dados reconstruídos com os reais da melhor forma possível
    lote_rec_reordenado, array_erros, _, _ = emparelhar_reconstrucao_referencia(
        X_real, X_rec, mapa_tolerancia, base_emparelhamento="all"
    )

    # 3. Calcula o RA (Reconstruction Accuracy)
    # array_erros contém a taxa de erro para cada linha.
    # Acurácia de uma linha é 1 - erro. A média disso tudo vezes 100 é o RA%.
    erro_medio = np.mean(array_erros)
    ra_score = (1.0 - erro_medio)

    return ra_score, lote_rec_reordenado
