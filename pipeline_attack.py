"""
cross_compare_attacks.py
========================

Comparação JUSTA entre TimberStrike, REBOOT e um baseline aleatório,
todos rodados sob exatamente o mesmo pipeline de avaliação.

Por que este script existe
--------------------------
Os scripts anteriores (`cross_TIMBERSTRIKE_RA.py` e `cross_REBOOT_desease_2.py`)
mediam coisas diferentes:
  * o TS rodava em X[:150, :5] enquanto o REBOOT rodava em X completo (569x30);
  * o REBOOT concatenava o label no X antes de calcular RA, o TS não;
  * o bug do `max(1, num_cats/cont)` em `_calcular_taxa_erro_amostra` inflava
    a RA do TS em ~8-17 p.p. e a do REBOOT em ~1-2 p.p.

Aqui:
  1. Os DOIS ataques enxergam o mesmo X_train/y_train (mesmo subset).
  2. A RA das features e a RA dos labels são reportadas SEPARADAMENTE,
     usando o mesmo emparelhamento Hungarian para ambos os ataques.
  3. O bug do metrics.py é corrigido via monkey-patch local — não toca
     no arquivo original.
  4. Um baseline aleatório (uniforme dentro do range de cada feature +
     classe majoritária para o label) entra como terceiro "ataque" para
     dar uma régua de "RA = X% é vazamento real?".

Saídas
------
  resultados_comparacao_long.csv     — 1 linha por (ataque, params, fold)
  resultados_comparacao_summary.csv  — agregado por (ataque, params)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, ParameterGrid

# === Imports do projeto =====================================================
from REBOOT.TIMBERSTRIKE.timberstrike_lgb import TimberStrikeLightGBM
from REBOOT.REBOOT.REBOOT import REBOOT
from REBOOT.tests.utils import formatar_saida_ataque
import REBOOT.metrics.metrics as _metrics_mod
from REBOOT.metrics.metrics import (
    criar_mapa_tolerancia,
    emparelhar_reconstrucao_referencia,
)


# ============================================================================
# 1. PATCH DO metrics.py (em runtime, não toca no arquivo)
# ============================================================================
def _calcular_taxa_erro_amostra_fixed(
    amostra_real, amostra_rec, mapa_tolerancia, detalhado=False,
):
    """
    Versão corrigida de `_calcular_taxa_erro_amostra`. A original calcula
    `erro_total = (erro_cat + erro_cont) / (max(1, num_cats) + max(1, num_conts))`,
    o que infla a RA quando o dataset tem 0 categóricas ou 0 contínuas
    (o denominador fica 1 + N em vez de N).
    """
    erro_cat = 0
    erro_cont = 0
    num_cats = 0
    num_conts = 0

    for v_real, v_rec, tol in zip(amostra_real, amostra_rec, mapa_tolerancia):
        if tol == "cat":
            erro_cat += 0 if str(v_real) == str(v_rec) else 1
            num_cats += 1
        elif not isinstance(tol, str):
            dentro = (float(v_real) - tol) <= float(v_rec) <= (float(v_real) + tol)
            erro_cont += 0 if dentro else 1
            num_conts += 1
        else:
            raise TypeError(
                "O mapa de tolerância deve conter números ou a string 'cat'."
            )

    total_feats = num_cats + num_conts
    erro_total = 0.0 if total_feats == 0 else (erro_cat + erro_cont) / total_feats

    if detalhado:
        return (
            erro_total,
            erro_cat / max(1, num_cats),
            erro_cont / max(1, num_conts),
        )
    return erro_total


# Aplica o monkey-patch uma única vez na importação deste módulo.
_metrics_mod._calcular_taxa_erro_amostra = _calcular_taxa_erro_amostra_fixed


# ============================================================================
# 2. AVALIAÇÃO UNIFICADA (mesma função para os 3 ataques)
# ============================================================================
def calcular_ra_features_e_indices(
    X_real: np.ndarray,
    X_rec: np.ndarray,
    indices_cat: list,
    tol_mult: float = 0.319,
):
    """
    RA das features + índices do Hungarian matching.
    NUNCA recebe o label — labels são calculados em `calcular_label_ra`
    usando os índices retornados aqui.
    """
    X_real = np.asarray(X_real, dtype=float)
    X_rec = np.asarray(X_rec, dtype=float)

    assert X_real.ndim == 2 and X_rec.ndim == 2, "X_real e X_rec devem ser 2D"
    assert X_real.shape[1] == X_rec.shape[1], (
        f"Número de features divergente: real={X_real.shape[1]} "
        f"rec={X_rec.shape[1]}"
    )

    n_real, n_rec = len(X_real), len(X_rec)
    if n_real != n_rec:
        n_min = min(n_real, n_rec)
        rng = np.random.RandomState(42)
        sub_real = rng.choice(n_real, size=n_min, replace=False)
        sub_rec = rng.choice(n_rec, size=n_min, replace=False)
        X_real_use, X_rec_use = X_real[sub_real], X_rec[sub_rec]
    else:
        sub_real = np.arange(n_real)
        sub_rec = np.arange(n_rec)
        X_real_use, X_rec_use = X_real, X_rec

    mapa_tol = criar_mapa_tolerancia(X_real_use, indices_cat, tol_mult)
    (
        _, array_erros, _, _, idx_l, idx_c,
    ) = emparelhar_reconstrucao_referencia(
        X_real_use,
        X_rec_use,
        mapa_tol,
        retornar_indices=True,
        base_emparelhamento="all",
    )
    ra = 1.0 - float(np.mean(array_erros))
    # Devolve índices no espaço ORIGINAL (antes da subamostragem)
    return ra, sub_real[idx_l], sub_rec[idx_c]


def calcular_label_ra(
    y_real: np.ndarray,
    y_rec: np.ndarray,
    idx_real_match: np.ndarray,
    idx_rec_match: np.ndarray,
) -> float:
    """Acurácia de label usando o emparelhamento Hungarian das features."""
    y_real = np.asarray(y_real)
    y_rec = np.asarray(y_rec)
    if len(idx_real_match) == 0 or len(y_rec) == 0:
        return float("nan")
    # Coage para int para comparação robusta (REBOOT às vezes devolve float).
    yr = y_real[idx_real_match].astype(int)
    yc = np.round(y_rec[idx_rec_match]).astype(int)
    return float(np.mean(yr == yc))


# ============================================================================
# 3. ADAPTADORES DE ATAQUE (interface unificada)
# ============================================================================
@dataclass
class AttackOutput:
    X_rec: np.ndarray  # shape (n, d)
    y_rec: np.ndarray  # shape (n,)
    attack_time_s: float
    info: dict


def run_timberstrike(
    booster, X_train, y_train, eta, reg_lambda, milp_time_limit=60,
) -> AttackOutput:
    n, d = X_train.shape
    feature_bounds = [
        (float(X_train[:, f].min() - 0.5), float(X_train[:, f].max() + 0.5))
        for f in range(d)
    ]
    attacker = TimberStrikeLightGBM(
        booster=booster,
        n_features=d,
        feature_bounds=feature_bounds,
        learning_rate=eta,
        reg_lambda=reg_lambda,
        milp_time_limit=milp_time_limit,
        verbose=False,
    )
    t0 = time.time()
    X_rec, y_rec = attacker.attack()
    X_rec = np.asarray(X_rec, dtype=float)
    y_rec = np.asarray(y_rec).ravel() if len(y_rec) else np.empty(0)
    return AttackOutput(
        X_rec=X_rec,
        y_rec=y_rec,
        attack_time_s=time.time() - t0,
        info={"n_reconstructed": len(X_rec)},
    )


def run_reboot(booster, X_train, y_train, eta, reg_lambda) -> AttackOutput:
    N, M = X_train.shape
    attacker = REBOOT(booster, N, M, num_categoricas=0, num_continuas=M)

    t0 = time.time()
    attacker.phase_1()
    attacker.phase_2()
    attacker.phase_3(eta=eta, reg_lambda=reg_lambda)

    trees_info = attacker.model_dump.get("tree_info", [])
    if len(trees_info) == 0:
        return AttackOutput(
            X_rec=np.empty((0, M)),
            y_rec=np.empty(0),
            attack_time_s=time.time() - t0,
            info={"n_reconstructed": 0, "reason": "no_trees"},
        )

    amostras_reconstruidas = attacker.phase_4()

    # `formatar_saida_ataque` recebe concat(X_real, y_real) como referência
    # e devolve as amostras no mesmo espaço — ou seja, shape (n, M+1).
    # Separamos usando M explícito (não [:, -1]) para ser robusto a qualquer
    # número extra de colunas que a função possa adicionar internamente.
    X_train_ref = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
    amostras_formatadas = np.asarray(
        formatar_saida_ataque(amostras_reconstruidas, X_train_ref),
        dtype=float,
    )

    n_cols = amostras_formatadas.shape[1]
    if n_cols < M + 1:
        raise ValueError(
            f"run_reboot: amostras_formatadas tem {n_cols} colunas, "
            f"esperava >= {M + 1} (M={M} features + 1 label)."
        )
    # As M primeiras colunas são as features; a coluna M é o label.
    # Qualquer coluna extra (n_cols > M+1) é descartada com aviso.
    if n_cols > M + 1:
        warnings.warn(
            f"run_reboot: amostras_formatadas tem {n_cols} colunas mas "
            f"M+1={M + 1} eram esperadas. As {n_cols - M - 1} colunas "
            "extras serão descartadas.",
            stacklevel=2,
        )

    return AttackOutput(
        X_rec=amostras_formatadas[:, :M],
        y_rec=amostras_formatadas[:, M],
        attack_time_s=time.time() - t0,
        info={
            "n_reconstructed": len(amostras_formatadas),
            "n_cols_raw": n_cols,
        },
    )


def run_random_baseline(
    booster_unused, X_train, y_train, eta_unused, reg_lambda_unused,
) -> AttackOutput:
    """
    Baseline: features sorteadas uniformemente dentro do range observado
    em X_train; label sempre na classe majoritária. NÃO usa o modelo.
    Serve de piso: se um ataque fica perto disso, ele não está aprendendo
    nada do modelo.
    """
    n, d = X_train.shape
    rng = np.random.RandomState(42)  # determinístico para reprodutibilidade
    t0 = time.time()
    X_rec = np.column_stack(
        [
            rng.uniform(X_train[:, f].min(), X_train[:, f].max(), size=n)
            for f in range(d)
        ]
    )
    majority = int(np.round(y_train.mean()))
    y_rec = np.full(n, majority, dtype=int)
    return AttackOutput(
        X_rec=X_rec,
        y_rec=y_rec,
        attack_time_s=time.time() - t0,
        info={"n_reconstructed": n, "majority_class": majority},
    )


# ============================================================================
# 4. CONFIGURAÇÃO DO EXPERIMENTO
# ============================================================================
# Subset usado por TODOS os ataques. Mantemos pequeno por causa do MILP do
# TimberStrike — mas o REBOOT recebe EXATAMENTE o mesmo subset, para que a
# única variável seja o ataque.
N_LIMIT = 150
D_LIMIT = 5

# Tolerância na RA (estilo TabLeak/TimberStrike).
TOL_MULT = 0.319

# Cross-validation.
N_SPLITS = 3
CV_SEED = 42

# Grid de hiperparâmetros (mesmo dos CSVs originais).
PARAM_GRID = {
    "num_boost_round": [10, 50],
    "num_leaves": [3, 5, 8],
    "max_depth": [-1, 8],
    "learning_rate": [0.05, 0.1],
    "min_data_in_leaf": [1, 3],
    "lambda_l2": [0.0, 0.1],
}

# Ataques a comparar.
ATTACKS: dict[str, Callable] = {
    "timberstrike": run_timberstrike,
    "reboot": run_reboot,
    "random": run_random_baseline,
}

# Arquivos de saída.
OUTFILE_LONG = "resultados_comparacao_long.csv"
OUTFILE_SUMMARY = "resultados_comparacao_summary.csv"


# ============================================================================
# 5. LOOP PRINCIPAL
# ============================================================================
def build_lgb_params(params_comb: dict, num_rounds: int) -> dict:
    """Mesmos parâmetros do LightGBM DP usados nos scripts originais."""
    return {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        # Parâmetros DP (DPBoost_2level)
        "boost_method": "DPBoost_2level",
        "total_budget": 50,
        "high_level_boost_round": 1,
        "inner_boost_round": 1,
        "balance_partition": 1,
        "geo_clip": 1,
        "bagging_freq": 1,
        "bagging_fraction": 0.5,
        "max_bin": 64,
        "verbose": -1,
        # Combinação atual
        "num_leaves": params_comb["num_leaves"],
        "max_depth": params_comb["max_depth"],
        "learning_rate": params_comb["learning_rate"],
        "min_data_in_leaf": params_comb["min_data_in_leaf"],
        "lambda_l2": params_comb["lambda_l2"],
        # `num_iterations` e `my_n_trees` precisam casar com num_rounds
        # para a variante DP funcionar como esperado.
        "num_iterations": num_rounds,
        "my_n_trees": num_rounds,
    }


def main() -> None:
    print("=== Preparando o ambiente ===")
    data = load_breast_cancer()
    X_full, y_full = data.data, data.target

    X = X_full[:N_LIMIT, :D_LIMIT]
    y = y_full[:N_LIMIT]
    INDICES_CAT: list = []  # Breast Cancer 100% numérico

    print(f"X.shape = {X.shape}   y.shape = {y.shape}")
    print(f"Classes: {np.bincount(y).tolist()}  (majority = {int(np.round(y.mean()))})")

    grid = list(ParameterGrid(PARAM_GRID))
    n_combinacoes = len(grid)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    print(
        f"\nGrid: {n_combinacoes} combinações × "
        f"{N_SPLITS} folds × {len(ATTACKS)} ataques "
        f"= {n_combinacoes * N_SPLITS * len(ATTACKS)} execuções"
    )

    rows_long: list[dict] = []
    t_global = time.time()

    for i_param, params_comb in enumerate(grid, start=1):
        print(f"\n[{i_param}/{n_combinacoes}] params={params_comb}")
        num_rounds = params_comb["num_boost_round"]
        eta = params_comb["learning_rate"]
        reg_lambda = params_comb["lambda_l2"]
        lgb_params = build_lgb_params(params_comb, num_rounds)

        for fold, (train_idx, _) in enumerate(kf.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_data = lgb.Dataset(X_tr, label=y_tr)
                booster = lgb.train(
                    lgb_params, train_data, num_boost_round=num_rounds,
                )

            for atk_name, atk_fn in ATTACKS.items():
                err = None
                try:
                    out = atk_fn(booster, X_tr, y_tr, eta, reg_lambda)
                    if len(out.X_rec) > 0:
                        ra_feat, idx_r, idx_c = calcular_ra_features_e_indices(
                            X_real=X_tr,
                            X_rec=out.X_rec,
                            indices_cat=INDICES_CAT,
                            tol_mult=TOL_MULT,
                        )
                        ra_lab = calcular_label_ra(y_tr, out.y_rec, idx_r, idx_c)
                        n_rec = out.info["n_reconstructed"]
                        atk_time = out.attack_time_s
                    else:
                        ra_feat, ra_lab, n_rec = float("nan"), float("nan"), 0
                        atk_time = out.attack_time_s
                except Exception as e:
                    ra_feat, ra_lab, n_rec, atk_time = (
                        float("nan"), float("nan"), 0, 0.0,
                    )
                    err = repr(e)
                    print(f"   [!] {atk_name} fold {fold}: {e}")

                rows_long.append(
                    {
                        "attack": atk_name,
                        **params_comb,
                        "fold": fold,
                        "ra_features": ra_feat,
                        "ra_labels": ra_lab,
                        "n_reconstructed": n_rec,
                        "attack_time_s": atk_time,
                        "error": err,
                    }
                )

            # Log curto por fold
            tail = rows_long[-len(ATTACKS):]
            log_bits = [
                f"{r['attack']}: feat={r['ra_features']:.3f} lab={r['ra_labels']:.3f}"
                if not np.isnan(r["ra_features"])
                else f"{r['attack']}: NaN"
                for r in tail
            ]
            print(f"   fold {fold}: " + " | ".join(log_bits))

    print(f"\nTempo total: {time.time() - t_global:.1f}s")

    # -----------------------------------------------------------------------
    # Long format (1 linha por execução)
    # -----------------------------------------------------------------------
    df_long = pd.DataFrame(rows_long)
    df_long.to_csv(OUTFILE_LONG, index=False)
    print(f"Salvo: {OUTFILE_LONG}  ({len(df_long)} linhas)")

    # -----------------------------------------------------------------------
    # Summary agregado por (attack, params)
    # -----------------------------------------------------------------------
    param_cols = list(PARAM_GRID.keys())
    df_summary = (
        df_long.groupby(["attack"] + param_cols, dropna=False)
        .agg(
            ra_features_mean=("ra_features", "mean"),
            ra_features_std=("ra_features", "std"),
            ra_labels_mean=("ra_labels", "mean"),
            ra_labels_std=("ra_labels", "std"),
            time_mean=("attack_time_s", "mean"),
            n_folds_ok=("ra_features", lambda s: int(s.notna().sum())),
        )
        .reset_index()
        .sort_values(["attack", "ra_features_mean"], ascending=[True, False])
    )
    df_summary.to_csv(OUTFILE_SUMMARY, index=False)
    print(f"Salvo: {OUTFILE_SUMMARY}  ({len(df_summary)} linhas)")

    # -----------------------------------------------------------------------
    # Print rápido do topo (e do baseline)
    # -----------------------------------------------------------------------
    print("\n=== TOP 3 por ataque (RA features médio) ===")
    for atk in ATTACKS:
        top = (
            df_summary[df_summary["attack"] == atk]
            .nlargest(3, "ra_features_mean")
            .drop(columns="attack")
        )
        print(f"\n--- {atk} ---")
        print(top.to_string(index=False))

    # Diferença média timberstrike - random e reboot - random (sinal de
    # quanto cada ataque aprende ALÉM do baseline).
    pivot = (
        df_long.groupby(["attack"] + param_cols)["ra_features"]
        .mean()
        .unstack("attack")
    )
    if {"timberstrike", "reboot", "random"}.issubset(pivot.columns):
        delta_ts = (pivot["timberstrike"] - pivot["random"]).mean()
        delta_rb = (pivot["reboot"] - pivot["random"]).mean()
        print("\n=== Vazamento médio acima do baseline aleatório ===")
        print(f"  TimberStrike - random : {delta_ts:+.4f}")
        print(f"  REBOOT       - random : {delta_rb:+.4f}")


if __name__ == "__main__":
    main()