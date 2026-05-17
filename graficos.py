import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os datasets
df_timber = pd.read_csv('resultados_ataque_timberstrike_cv.csv')
df_reboot = pd.read_csv('resultados_ataque_reboot_cv_2_desease.csv')

# Definir as colunas de hiperparâmetros que são comuns
hparams = ['lambda_l2', 'learning_rate', 'max_depth', 'min_data_in_leaf', 'num_leaves', 'num_boost_round']

# 2. Unir os datasets para comparação direta por configuração
df_comparacao = pd.merge(
    df_timber, 
    df_reboot, 
    on=hparams, 
    suffixes=('_timber', '_reboot')
)

# 3. Exibir as melhores configurações de cada ataque
print("Melhores resultados Timberstrike (por Features RA):")
print(df_timber.nlargest(5, 'ra_features_medio')[['ra_features_medio', 'ra_labels_medio'] + hparams])

print("\nMelhores resultados Reboot (por RA):")
print(df_reboot.nlargest(5, 'ra_medio')[['ra_medio'] + hparams])

# 4. Visualização: Comparação de Distribuição das Métricas
plt.figure(figsize=(10, 6))
data_to_plot = {
    'Timberstrike (Features)': df_comparacao['ra_features_medio'],
    'Timberstrike (Labels)': df_comparacao['ra_labels_medio'],
    'Reboot': df_comparacao['ra_medio']
}
sns.boxplot(data=pd.DataFrame(data_to_plot))
plt.title('Comparação da Acurácia de Reconstrução (RA) entre Ataques')
plt.ylabel('RA Médio')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('comparacao_ataques_box.png')

# 5. Visualização: Top 10 configurações (Comparação Lado a Lado)
# Vamos pegar as 10 configurações onde o Reboot teve melhor desempenho e ver o Timberstrike nelas
top_10_reboot = df_comparacao.nlargest(10, 'ra_medio')

top_10_reboot.plot(kind='bar', x='num_boost_round', y=['ra_features_medio', 'ra_labels_medio', 'ra_medio'], 
                   figsize=(12, 6))
plt.title('RA nas Top 10 Configurações do Reboot')
plt.ylabel('Acurácia de Reconstrução')
plt.xlabel('Número de Boost Rounds (como exemplo de eixo X)')
plt.legend(['Timber (Features)', 'Timber (Labels)', 'Reboot'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('comparacao_top10.png')

print("\nArquivos de análise e gráficos gerados com sucesso.")