"""
profile_builder.py
Compute per-trader behavioral feature vectors from pre-news-window trades.

For each unique taker address, aggregates:
  - Bet size statistics (avg, max, total, spike ratio)
  - Direction bias (fraction of buys using D or taker_direction)
  - Portfolio concentration (HHI across markets)
  - Activity level (trade count, unique markets)
  - Timing relative to news release

The market_index uses 'market_id_prefix' for FFIC markets.
We match by condition_id.startswith(prefix) when joining.

Input:
  data/processed/market_index.parquet
  data/raw/polymarket/daily_aligned/*.parquet  (pre-filtered per-market files)

Output:
  data/processed/trades_filtered.parquet   — pre-news trades only
  data/processed/trader_profiles.parquet   — per-trader feature matrix
"""

import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
POLY_DIR = RAW_DIR / "polymarket" / "daily_aligned"

WINDOW_COL = "window_start_48h"   # primary; change to window_start_24h for sensitivity
EPS = 1e-9


def load_market_index() -> pd.DataFrame:
    idx_path = PROCESSED_DIR / "market_index.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Market index not found at {idx_path}. Run build_market_index.py first."
        )
    return pd.read_parquet(idx_path)


def filter_trades(market_index: pd.DataFrame, window_col: str) -> pd.DataFrame:
    """
    Load all per-market parquet files and keep only trades that:
      1. condition_id starts with a known FFIC market_id_prefix
      2. Fall within the pre-news window: window_start <= block_timestamp < news_timestamp
    """
    parquet_glob = str(POLY_DIR / "*.parquet")
    if not any(POLY_DIR.glob("*.parquet")):
        raise FileNotFoundError(
            f"No parquet files found at {POLY_DIR}. Run fetch_polymarket.py first."
        )

    con = duckdb.connect()
    con.register("market_index", market_index)

    print(f"  Filtering trades across {len(list(POLY_DIR.glob('*.parquet')))} files (window: {window_col}) ...")
    rows = []
    for p in POLY_DIR.glob("*.parquet"):
        query = f"""
        SELECT
            t.* EXCLUDE (resolved_at, category),
            m.is_leaked,
            m.case_id,
            m.case_title,
            m.news_timestamp,
            m.market_id_prefix,
            m.{window_col} AS window_start
        FROM read_parquet('{str(p)}') AS t
        INNER JOIN market_index AS m
            ON lower(t.condition_id) LIKE m.market_id_prefix || '%'
        WHERE
            t.block_timestamp >= m.{window_col}
            AND t.block_timestamp < m.news_timestamp
        """
        try:
            df_p = con.execute(query).df()
            if len(df_p) > 0:
                rows.append(df_p)
        except Exception as e:
            # If EXCLUDE fails because columns don't exist, try without EXCLUDE
            try:
                query_fallback = f"""
                SELECT
                    t.*,
                    m.is_leaked,
                    m.case_id,
                    m.case_title,
                    m.news_timestamp,
                    m.market_id_prefix,
                    m.{window_col} AS window_start
                FROM read_parquet('{str(p)}') AS t
                INNER JOIN market_index AS m
                    ON lower(t.condition_id) LIKE m.market_id_prefix || '%'
                WHERE
                    t.block_timestamp >= m.{window_col}
                    AND t.block_timestamp < m.news_timestamp
                """
                df_p = con.execute(query_fallback).df()
                if len(df_p) > 0:
                    rows.append(df_p)
            except Exception:
                continue
    con.close()
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"  {len(df):,} trades in pre-news windows")
    return df


def compute_hhi(series: pd.Series) -> float:
    """Herfindahl-Hirschman Index: volume concentration across markets."""
    total = series.sum()
    if total == 0:
        return 0.0
    shares = series / total
    return float((shares ** 2).sum())


def build_profiles(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-taker features. Returns one row per unique taker address.
    """
    # Determine direction column
    if "D" in trades.columns:
        dir_col = "D"
        buy_val: int | str = 1
    elif "taker_direction" in trades.columns:
        dir_col = "taker_direction"
        buy_val = "BUY"
    else:
        print("[warn] No direction column found — direction_bias will be 0.5", file=sys.stderr)
        trades = trades.copy()
        trades["_dir"] = 0
        dir_col = "_dir"
        buy_val = 1

    rows = []
    for taker, grp in trades.groupby("taker"):
        usdc = grp["usdc_amount"].fillna(0)
        total_vol = float(usdc.sum())
        avg_bet = float(usdc.mean())
        max_bet = float(usdc.max())
        n_trades = len(grp)

        unique_mkts = grp["condition_id"].nunique()
        direction_bias = float((grp[dir_col] == buy_val).mean())

        vol_by_market = grp.groupby("condition_id")["usdc_amount"].sum()
        hhi = compute_hhi(vol_by_market)

        news_ts = float(grp["news_timestamp"].iloc[0])
        time_to_news_min = (news_ts - float(grp["block_timestamp"].max())) / 60.0
        time_span_hours = (
            float(grp["block_timestamp"].max()) - float(grp["block_timestamp"].min())
        ) / 3600.0

        is_leaked = int(grp["is_leaked"].max())
        case_id = str(grp["case_id"].iloc[0]) if "case_id" in grp.columns else ""

        rows.append(
            {
                # Identity / labels
                "taker": taker,
                "is_leaked_market": is_leaked,
                "case_id": case_id,
                # Volume features
                "total_volume_usdc": total_vol,
                "avg_bet_usdc": avg_bet,
                "max_bet_usdc": max_bet,
                "bet_size_ratio": max_bet / (avg_bet + EPS),
                # Direction
                "direction_bias": direction_bias,
                # Portfolio
                "unique_markets": int(unique_mkts),
                "market_concentration_hhi": hhi,
                # Activity
                "trade_count": int(n_trades),
                # Timing
                "time_to_news_min": time_to_news_min,
                "time_span_hours": time_span_hours,
                # Log-transforms of heavy-tail features
                "log_total_volume": float(np.log1p(total_vol)),
                "log_max_bet": float(np.log1p(max_bet)),
                "log_trade_count": float(np.log1p(n_trades)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading market index ...")
    market_index = load_market_index()
    print(
        f"  {len(market_index)} markets "
        f"({market_index['is_leaked'].sum()} leaked, "
        f"{(market_index['is_leaked'] == 0).sum()} control)"
    )

    print("\nFiltering trades to pre-news windows ...")
    trades = filter_trades(market_index, WINDOW_COL)

    if len(trades) == 0:
        print(
            "[warn] No trades found. Ensure fetch_polymarket.py has been run "
            "and downloaded data for the target markets.",
            file=sys.stderr,
        )
        sys.exit(1)

    trades_out = PROCESSED_DIR / "trades_filtered.parquet"
    trades.to_parquet(trades_out, index=False)
    print(f"  Saved → {trades_out}")

    print("\nBuilding per-trader behavioral profiles ...")
    profiles = build_profiles(trades)
    print(f"  {len(profiles):,} unique traders")
    print(f"  Leaked-market traders: {profiles['is_leaked_market'].sum():,}")
    print(f"  Control traders:       {(profiles['is_leaked_market'] == 0).sum():,}")

    # Sanity checks
    assert profiles.isnull().sum().sum() == 0, "NaN values found in profiles — check pipeline"
    assert len(profiles) > 0, "No traders found — check trade filtering"

    feat_cols = [
        c for c in profiles.columns
        if c not in ("taker", "is_leaked_market", "case_id")
    ]
    profiles_out = PROCESSED_DIR / "trader_profiles.parquet"
    profiles.to_parquet(profiles_out, index=False)
    print(f"\nTrader profiles saved → {profiles_out}")
    print(f"  Feature columns ({len(feat_cols)}): {feat_cols}")


if __name__ == "__main__":
    main()
