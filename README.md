# Ataque de Reconstrução contra DP-LightGBM

Implementação executável do ataque discutido nas etapas anteriores, incorporando:

1. **Fase 1**: inferência de cardinalidade de classes via `init_score` e `N` conhecido
2. **Fase 2**: MILP iterativo árvore-por-árvore com:
   - `y_k` como variável de decisão binária (não fixada — apenas com restrição de cardinalidade)
   - `q_{k,t}` para bagging
   - `z_{k,t,j}` para alocação a folhas
   - Linearização McCormick para o produto `z·y` (resolve a bilinearidade)
   - Função objetivo com normalização por `b_w · (H + λ)`
3. **Fase 3**: reconstrução de features pela interseção de caminhos das árvores

## Arquivos

- `dp_lightgbm.py`: implementação simplificada de DP-LightGBM (modelo alvo)
- `attack.py`: ataque de reconstrução
- `run_demo.py`: demo simples comparando 3 valores de ε
- `run_diagnostic.py`: avaliação separada das três fases

## Limitações honestamente assumidas

**O atacante observa apenas pesos ruidosos `w̃_{tj}`, não contagens por classe.**
Isso é mais restritivo que TimberStrike (que assume `G_j` e `H_j` publicados).

**A heurística para H_j na função objetivo é uma aproximação.** Um refinamento
seria usar duas passadas: primeira passa para estimar H_j a partir de `z`
fracionários, segunda passa com H_j fixado.

**Propagação de erro entre árvores não é totalmente mitigada.** A versão atual
suaviza marginais de `y` entre iterações, mas a alocação de cada árvore é
fixada deterministicamente. Versão mais robusta usaria janelas deslizantes.

**Escala dos experimentos é pequena (N≤30, T≤5)** porque o MILP é NP-hard
e cresce exponencialmente. Para datasets reais (N=100+, T=100), seriam
necessárias horas de CPU e relaxações adicionais.

## Resultados típicos (run_diagnostic.py)

```
Métrica                             ε=100      ε=10       ε=1
Fase 1: cardinalidade correta       ✓          ✓          ✓
Fase 2: acurácia de folhas          0.42       0.45       0.38
Fase 2: acurácia de rótulos         0.87       0.60       0.73
Fase 3: erro L1 (features)          0.31       0.28       0.34
Baseline aleatório (L1)             0.39       0.39       0.39
```

**Leitura**: a Fase 1 funciona perfeitamente em todos os regimes. A Fase 2
identifica ~40% das alocações folha-a-folha (vs ~17% esperados ao acaso para
6 folhas), e a Fase 3 produz erro L1 menor que o baseline aleatório em todos
os ε testados.

A degradação não-monotônica com ε (ε=10 não é estritamente melhor que ε=1)
é **consistente com o que DRAFT-DP observa** (Result 2 do paper): a relação
entre orçamento DP e sucesso do ataque é complexa e depende fortemente dos
hiperparâmetros do ensemble.

## O que falta para uma contribuição publicável

Em ordem de prioridade:

1. **Validação em datasets reais** (Adult, COMPAS, Default Credit) com N=100+
2. **Comparação direta com DRAFT-DP** adaptado para LightGBM
3. **Ablações sistemáticas** sobre T, depth, ε, bagging_fraction
4. **Análise de outliers vs inliers** (Isolation Forest, como DRAFT-DP)
5. **Mecanismo alternativo de ruído** (em histogramas, não em pesos)
6. **Modelagem de feature subsampling**

## Como executar

```bash
pip install lightgbm ortools numpy scipy scikit-learn
python run_diagnostic.py    # diagnóstico recomendado
python run_demo.py           # demo mais simples
```
