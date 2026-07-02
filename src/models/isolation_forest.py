"""
isolation_forest.py
Train an Isolation Forest anomaly detector on control-market traders,
then score all traders (control + leaked market traders).

Usage:
    python src/models/isolation_forest.py

Input:  data/processed/trader_profiles.parquet
Output:
    data/processed/if_scores.parquet  — trader + anomaly_score_if + rank_if
"""

import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURE_COLS = [
    "log_total_volume",
    "log_max_bet",
    "log_trade_count",
    "avg_bet_usdc",
    "bet_size_ratio",
    "direction_bias",
    "unique_markets",
    "market_concentration_hhi",
    "time_to_news_min",
    "time_span_hours",
]

IF_PARAMS = dict(
    n_estimators=300,
    contamination=0.01,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)


def run(profiles: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    """
    Train Isolation Forest on control traders and score everyone.
    Returns DataFrame with columns: taker, is_leaked_market, case_id,
                                    anomaly_score_if, rank_if
    """
    if profiles is None:
        path = PROCESSED_DIR / "trader_profiles.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Profiles not found at {path}. Run profile_builder.py first.")
        profiles = pd.read_parquet(path)

    X = profiles[FEATURE_COLS].values.astype(np.float32)

    # Fit scaler on control traders only
    control_mask = profiles["is_leaked_market"] == 0
    scaler = StandardScaler()
    scaler.fit(X[control_mask])
    X_scaled = scaler.transform(X)

    # Train on control traders only
    X_control = X_scaled[control_mask]
    print(f"Training Isolation Forest on {X_control.shape[0]:,} control traders ...")
    clf = IsolationForest(**IF_PARAMS)
    clf.fit(X_control)

    # Score all traders: higher = more anomalous
    raw_score = clf.decision_function(X_scaled)   # lower = more anomalous in sklearn
    anomaly_score = -raw_score                    # flip so higher = more anomalous

    result = profiles[["taker", "is_leaked_market", "case_id"]].copy()
    result["anomaly_score_if"] = anomaly_score
    result["rank_if"] = result["anomaly_score_if"].rank(ascending=False, method="min").astype(int)

    if save:
        out = PROCESSED_DIR / "if_scores.parquet"
        result.to_parquet(out, index=False)
        print(f"IF scores saved → {out}")
        # Also save model + scaler for reproducibility
        with open(PROCESSED_DIR / "isolation_forest_model.pkl", "wb") as f:
            pickle.dump({"model": clf, "scaler": scaler, "features": FEATURE_COLS}, f)

    return result


if __name__ == "__main__":
    result = run()
    top = result.nsmallest(20, "rank_if")[["taker", "is_leaked_market", "case_id", "anomaly_score_if", "rank_if"]]
    print("\nTop-20 anomalies:")
    print(top.to_string(index=False))
    leaked_in_top = result[result["rank_if"] <= 100]["is_leaked_market"].sum()
    total_top = 100
    print(f"\nPrecision@100 (IF): {leaked_in_top}/{total_top} = {leaked_in_top/total_top:.3f}")
