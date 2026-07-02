"""
build_market_index.py
Construct the master market index combining:
  - FFIC leaked markets (ground truth)
  - Matched control markets (same category, similar resolution date, no leaks)

Output: data/processed/market_index.parquet

Schema:
  condition_id       str    — Polymarket condition_id
  is_leaked          int    — 1 = FFIC case, 0 = control
  case_id            str    — FFIC case ID (NaN for controls)
  case_title         str    — human-readable title
  category           str    — market category
  news_timestamp     int    — Unix seconds — news release / resolution time
  window_start_24h   int    — news_timestamp - 86400
  window_start_48h   int    — news_timestamp - 172800
  resolution_outcome str    — YES / NO / INVALID (NaN for controls)
"""

import json
import pathlib
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
FFIC_DIR = RAW_DIR / "ffic"
POLY_DIR = RAW_DIR / "polymarket" / "daily_aligned"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"

CONTROL_MULTIPLIER = 3   # aim for 3× as many control markets as leaked


def parse_timestamp(ts_str: str) -> int:
    """Parse ISO 8601 timestamp string to Unix seconds."""
    if not ts_str:
        return 0
    ts_str = ts_str.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return 0


def load_ffic_rows(ffic_dir: pathlib.Path) -> list[dict]:
    """Parse FFIC jsonl into flat per-market rows."""
    jsonl = ffic_dir / "ffic-v1.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"FFIC jsonl not found at {jsonl}. Run fetch_ffic.py first.")

    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            for mkt in case.get("markets", []):
                cid = (
                    mkt.get("condition_id_full")
                    or mkt.get("condition_id")
                    or mkt.get("market_id", "")
                )
                news_ts = parse_timestamp(
                    mkt.get("resolution_timestamp") or case.get("date", "")
                )
                rows.append(
                    {
                        "condition_id": cid.lower(),
                        "is_leaked": 1,
                        "case_id": case.get("case_id", ""),
                        "case_title": case.get("title", ""),
                        "category": case.get("category", ""),
                        "news_timestamp": news_ts,
                        "window_start_24h": news_ts - 86_400,
                        "window_start_48h": news_ts - 172_800,
                        "resolution_outcome": mkt.get("resolution_outcome", ""),
                    }
                )
    return rows


def load_polymarket_market_metadata(poly_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load market metadata from the already-downloaded filtered parquet files.
    Returns one row per unique condition_id with category and resolved_at.
    """
    if not poly_dir.exists() or not any(poly_dir.glob("*.parquet")):
        print(
            "[warn] No Polymarket parquet files found in data/raw/polymarket/daily_aligned/.\n"
            "       Run fetch_polymarket.py first, or market_index will have leaked markets only.",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=["condition_id", "category", "resolved_at"])

    con = duckdb.connect()
    parquet_glob = str(poly_dir / "*.parquet")
    df = con.execute(
        f"""
        SELECT
            lower(condition_id) AS condition_id,
            category,
            category_refined,
            ANY_VALUE(resolved_at) AS resolved_at
        FROM read_parquet('{parquet_glob}')
        GROUP BY 1, 2, 3
        """
    ).df()
    con.close()
    return df


def select_controls(
    leaked_rows: list[dict],
    all_markets: pd.DataFrame,
    leaked_ids: set[str],
    n_controls: int,
) -> list[dict]:
    """
    Select matched control markets:
      - Same category as a leaked market
      - resolved_at within ±30 days of the corresponding leaked market
      - Not in FFIC leaked set
    """
    if all_markets.empty:
        print("[warn] No Polymarket market metadata available — skipping control selection.")
        return []

    leaked_by_cat: dict[str, list[int]] = {}
    for row in leaked_rows:
        cat = row["category"]
        leaked_by_cat.setdefault(cat, []).append(row["news_timestamp"])

    controls: list[dict] = []
    seen: set[str] = set()
    WINDOW_SEC = 30 * 86_400  # ±30 days

    for _, mkt_row in all_markets.iterrows():
        cid = str(mkt_row["condition_id"]).lower()
        if cid in leaked_ids or cid in seen:
            continue
        cat = str(mkt_row.get("category", ""))
        if cat not in leaked_by_cat:
            continue

        res_at = mkt_row.get("resolved_at")
        if pd.isna(res_at) or res_at == 0:
            continue
        res_ts = int(res_at) if isinstance(res_at, (int, float)) else parse_timestamp(str(res_at))

        # Check if close to any leaked market in same category
        for leaked_ts in leaked_by_cat[cat]:
            if abs(res_ts - leaked_ts) <= WINDOW_SEC:
                seen.add(cid)
                controls.append(
                    {
                        "condition_id": cid,
                        "is_leaked": 0,
                        "case_id": "",
                        "case_title": "",
                        "category": cat,
                        "news_timestamp": res_ts,
                        "window_start_24h": res_ts - 86_400,
                        "window_start_48h": res_ts - 172_800,
                        "resolution_outcome": "",
                    }
                )
                break

        if len(controls) >= n_controls:
            break

    return controls


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FFIC rows ...")
    leaked_rows = load_ffic_rows(FFIC_DIR)
    leaked_ids = {r["condition_id"] for r in leaked_rows}
    print(f"  {len(leaked_rows)} leaked market rows ({len(leaked_ids)} unique condition_ids)")

    print("Loading Polymarket market metadata ...")
    all_markets = load_polymarket_market_metadata(POLY_DIR)
    print(f"  {len(all_markets)} markets found in downloaded data")

    target_controls = len(leaked_rows) * CONTROL_MULTIPLIER
    print(f"Selecting up to {target_controls} matched control markets ...")
    control_rows = select_controls(leaked_rows, all_markets, leaked_ids, target_controls)
    print(f"  {len(control_rows)} control markets selected")

    all_rows = leaked_rows + control_rows
    df = pd.DataFrame(all_rows)
    # Deduplicate (same condition_id could appear in multiple FFIC cases)
    df = df.drop_duplicates(subset=["condition_id", "is_leaked"])

    out = PROCESSED_DIR / "market_index.parquet"
    df.to_parquet(out, index=False)
    print(f"\nMarket index saved → {out}")
    print(f"  Leaked:  {df['is_leaked'].sum()}")
    print(f"  Control: {(df['is_leaked'] == 0).sum()}")
    print(f"  Total:   {len(df)}")


if __name__ == "__main__":
    main()
