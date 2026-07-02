"""
metrics.py
Evaluation utilities for the Polymarket insider detection experiment.

Functions:
  precision_at_k(scores_df, k)  — Precision@K
  compute_roc_auc(scores_df)    — ROC-AUC
  compute_pr_auc(scores_df)     — Average Precision (PR-AUC)
  recall_at_fpr(scores_df, fpr) — Recall at a given false positive rate
  evaluate_model(scores_df, label_col, score_col, name)  — full evaluation report
  ensemble_scores(if_df, ae_df) — combine both model scores
  leave_one_case_out(profiles, cases, train_fn, score_col) — LOCO test
"""

from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"


def precision_at_k(df: pd.DataFrame, score_col: str, label_col: str, k: int) -> float:
    """Fraction of top-K flagged traders with is_leaked_market == 1."""
    top_k = df.nlargest(k, score_col)
    return float(top_k[label_col].mean())


def recall_at_fpr(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    target_fpr: float = 0.01,
) -> float:
    """Recall at the threshold where FPR ≈ target_fpr."""
    y_true = df[label_col].values
    y_score = df[score_col].values
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
    # Find index closest to target_fpr
    idx = np.argmin(np.abs(fpr_arr - target_fpr))
    return float(tpr_arr[idx])


def evaluate_model(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "is_leaked_market",
    name: str = "model",
    k_values: list[int] | None = None,
) -> dict:
    """
    Full evaluation report. Returns dict with all metrics.
    Prints a formatted summary.
    """
    if k_values is None:
        k_values = [10, 20, 50, 100, 200]

    y_true = df[label_col].values
    y_score = df[score_col].values

    n_positive = y_true.sum()
    n_total = len(y_true)
    base_rate = n_positive / n_total

    roc_auc = roc_auc_score(y_true, y_score) if n_positive > 0 else float("nan")
    pr_auc = average_precision_score(y_true, y_score) if n_positive > 0 else float("nan")
    recall_1pct = recall_at_fpr(df, score_col, label_col, 0.01) if n_positive > 0 else float("nan")

    prec_at = {k: precision_at_k(df, score_col, label_col, k) for k in k_values}

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Positives (leaked): {n_positive}/{n_total}  (base rate={base_rate:.4f})")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  Recall@1%FPR: {recall_1pct:.4f}")
    print("  Precision@K:")
    for k, p in prec_at.items():
        lift = p / base_rate if base_rate > 0 else float("nan")
        print(f"    @{k:4d}: {p:.4f}  (lift={lift:.1f}×)")
    print()

    return {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall_at_1pct_fpr": recall_1pct,
        **{f"precision_at_{k}": v for k, v in prec_at.items()},
    }


def ensemble_scores(if_df: pd.DataFrame, ae_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine IF and AE scores by normalizing each to [0,1] and averaging.
    Returns a merged DataFrame with anomaly_score_ensemble column.
    """
    merged = if_df[["taker", "is_leaked_market", "case_id", "anomaly_score_if"]].merge(
        ae_df[["taker", "anomaly_score_ae"]], on="taker", how="inner"
    )

    # Min-max normalize each score
    def minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi == lo:
            return s * 0.0
        return (s - lo) / (hi - lo)

    merged["score_if_norm"] = minmax(merged["anomaly_score_if"])
    merged["score_ae_norm"] = minmax(merged["anomaly_score_ae"])
    merged["anomaly_score_ensemble"] = (merged["score_if_norm"] + merged["score_ae_norm"]) / 2
    merged["rank_ensemble"] = (
        merged["anomaly_score_ensemble"].rank(ascending=False, method="min").astype(int)
    )
    return merged


def leave_one_case_out(
    profiles: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], pd.DataFrame],
    score_col: str,
    label_col: str = "is_leaked_market",
) -> pd.DataFrame:
    """
    For each unique case_id in profiles, train on all OTHER cases' control traders,
    then score the held-out case's traders.

    train_fn: function that takes profiles DataFrame and returns scores DataFrame
              with columns: taker, {score_col}

    Returns DataFrame with per-case Precision@20 and ROC-AUC.
    """
    case_ids = profiles[profiles[label_col] == 1]["case_id"].unique()
    results = []

    for held_out in case_ids:
        print(f"\n[LOCO] Holding out case: {held_out}")

        # Training set: control traders + leaked traders from OTHER cases
        train_mask = (profiles["case_id"] != held_out) & (profiles[label_col] == 0)
        train_set = profiles[train_mask].copy()

        # Score on the held-out case's traders + control traders
        held_out_leaked = profiles[profiles["case_id"] == held_out]
        controls = profiles[profiles[label_col] == 0].sample(
            min(len(profiles[profiles[label_col] == 0]), 5_000), random_state=42
        )
        eval_set = pd.concat([held_out_leaked, controls], ignore_index=True)

        # Combine train + eval as the full "universe" for training
        full_for_train = pd.concat([train_set, eval_set], ignore_index=True)

        try:
            scored = train_fn(full_for_train)
        except Exception as e:
            print(f"  [error] Training failed for case {held_out}: {e}")
            continue

        scored_eval = scored[scored["taker"].isin(eval_set["taker"])].copy()
        scored_eval = scored_eval.merge(eval_set[["taker", label_col, "case_id"]], on="taker", how="left")

        if scored_eval[label_col].sum() == 0:
            print(f"  [skip] No positive labels in eval set for {held_out}")
            continue

        p20 = precision_at_k(scored_eval, score_col, label_col, 20)
        try:
            auc = roc_auc_score(scored_eval[label_col], scored_eval[score_col])
        except Exception:
            auc = float("nan")

        print(f"  Precision@20={p20:.3f}  ROC-AUC={auc:.3f}")
        results.append({"case_id": held_out, "precision_at_20": p20, "roc_auc": auc})

    return pd.DataFrame(results)


def load_and_evaluate() -> None:
    """Load saved score files and run full evaluation."""
    if_path = PROCESSED_DIR / "if_scores.parquet"
    ae_path = PROCESSED_DIR / "ae_scores.parquet"

    if not if_path.exists() or not ae_path.exists():
        print("[error] Score files not found. Run isolation_forest.py and autoencoder.py first.")
        return

    if_scores = pd.read_parquet(if_path)
    ae_scores = pd.read_parquet(ae_path)

    evaluate_model(if_scores, "anomaly_score_if", name="Isolation Forest")
    evaluate_model(ae_scores, "anomaly_score_ae", name="Autoencoder")

    ensemble = ensemble_scores(if_scores, ae_scores)
    evaluate_model(ensemble, "anomaly_score_ensemble", name="Ensemble (IF + AE)")


if __name__ == "__main__":
    load_and_evaluate()
