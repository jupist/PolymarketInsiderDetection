"""
build_market_index.py
Construct the master market index combining:
  - FFIC leaked markets (ground truth) — matched via market_id_prefix
  - Matched control markets (same category, similar resolution date, no leaks)

Output: data/processed/market_index.parquet

Schema:
  market_id_prefix   str    — 8-char hex prefix (e.g. 0xc1b6d712)
  condition_id       str    — full condition ID (if resolved, else same as prefix)
  is_leaked          int    — 1 = FFIC case, 0 = control
  case_id            str    — FFIC case ID (empty for controls)
  case_title         str    — human-readable title
  category           str    — market category
  news_timestamp     int    — Unix seconds — news release / resolution time
  window_start_24h   int    — news_timestamp - 86400
  window_start_48h   int    — news_timestamp - 172800
  resolution_outcome str    — YES / NO / INVALID (empty for controls)
  trade_available    bool   — True if FFIC says trade history is available
"""

import json
import pathlib
import sys
from datetime import datetime, timezone

import pandas as pd

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
FFIC_DIR = RAW_DIR / "ffic"
POLY_DIR = RAW_DIR / "polymarket" / "daily_aligned"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"

CONTROL_MULTIPLIER = 3   # aim for 3× as many control markets as leaked


def parse_timestamp(ts_str: str) -> int:
    """Parse ISO 8601 date/datetime string or integer string to Unix seconds."""
    if not ts_str or pd.isna(ts_str):
        return 0
    ts_str = str(ts_str).strip().rstrip("Z")
    if ts_str.isdigit():
        return int(ts_str)
    try:
        if float(ts_str) > 1e8:
            return int(float(ts_str))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return 0


def load_ffic_rows(ffic_dir: pathlib.Path) -> list[dict]:
    """
    Parse FFIC jsonl and CSV into flat per-market rows.
    FFIC uses market_id_prefix (8-char hex) as the market identifier.
    """
    jsonl = ffic_dir / "ffic-v1.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"FFIC jsonl not found at {jsonl}. Run fetch_ffic.py first.")

    # Load resolved full condition IDs if available
    resolved_path = ffic_dir / "resolved_ids.json"
    resolved: dict[str, str] = {}
    if resolved_path.exists():
        resolved = {k.lower(): v.lower() for k, v in json.loads(resolved_path.read_text()).items()}

    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_date_ts = parse_timestamp(case.get("date", ""))

            for mkt in case.get("markets", []):
                prefix = mkt.get("market_id_prefix", "").lower()
                if not prefix:
                    continue

                # Resolution timestamp: prefer market-level, fall back to case date
                res_ts = parse_timestamp(mkt.get("resolution_date", "")) or case_date_ts

                # Full condition_id: use resolved if available, else use prefix
                full_cid = resolved.get(prefix, prefix)

                trade_avail = mkt.get("trade_history_available", False)
                # treat True and "partial" as available
                is_available = trade_avail is True or trade_avail == "partial"

                rows.append(
                    {
                        "market_id_prefix": prefix,
                        "condition_id": full_cid,
                        "is_leaked": 1,
                        "case_id": case.get("case_id", ""),
                        "case_title": case.get("title", ""),
                        "category": case.get("category", ""),
                        "news_timestamp": res_ts,
                        "window_start_24h": res_ts - 86_400,
                        "window_start_48h": res_ts - 172_800,
                        "resolution_outcome": mkt.get("resolution_outcome", ""),
                        "trade_available": is_available,
                    }
                )
    return rows


def load_polymarket_market_metadata(poly_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load market metadata from already-downloaded filtered parquet files.
    Returns one row per unique condition_id with category and resolved_at.
    """
    if not poly_dir.exists() or not any(poly_dir.glob("*.parquet")):
        print(
            "[warn] No Polymarket parquet files found in data/raw/polymarket/daily_aligned/.\n"
            "       Run fetch_polymarket.py first, or market_index will have leaked markets only.",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=["condition_id", "category", "resolved_at"])

    import duckdb
    con = duckdb.connect()
    rows = []
    for p in poly_dir.glob("*.parquet"):
        try:
            df_p = con.execute(
                f"""
                SELECT
                    lower(condition_id) AS condition_id,
                    ANY_VALUE(category) AS category,
                    ANY_VALUE(category_refined) AS category_refined,
                    ANY_VALUE(try_cast(resolved_at AS VARCHAR)) AS resolved_at,
                    ANY_VALUE(market_slug) AS market_slug
                FROM read_parquet('{str(p)}')
                GROUP BY 1
                """
            ).df()
            rows.append(df_p)
        except Exception as e:
            continue
    con.close()
    if not rows:
        return pd.DataFrame(columns=["condition_id", "category", "resolved_at"])
    df = pd.concat(rows, ignore_index=True)
    return df.drop_duplicates(subset=["condition_id"])


def select_controls(
    leaked_rows: list[dict],
    all_markets: pd.DataFrame,
    leaked_prefixes: set[str],
    n_controls: int,
) -> list[dict]:
    """
    Select matched control markets:
      - Same category as a leaked market
      - resolved_at within ±30 days of the corresponding leaked market
      - Not in FFIC leaked set (checked by prefix and full condition_id)
    """
    if all_markets.empty:
        print("[warn] No Polymarket market metadata available — skipping control selection.")
        return []

    leaked_by_cat: dict[str, list[int]] = {}
    for row in leaked_rows:
        cat = row["category"]
        leaked_by_cat.setdefault(cat, []).append(row["news_timestamp"])

    leaked_full_ids = {r["condition_id"] for r in leaked_rows}

    controls: list[dict] = []
    seen: set[str] = set()
    WINDOW_SEC = 30 * 86_400  # ±30 days

    for _, mkt_row in all_markets.iterrows():
        cid = str(mkt_row["condition_id"]).lower()

        # Skip if it looks like a leaked market
        is_leaked = (
            cid in leaked_full_ids
            or any(cid.startswith(pfx) for pfx in leaked_prefixes)
        )
        if is_leaked or cid in seen:
            continue

        cat = str(mkt_row.get("category", ""))
        if cat not in leaked_by_cat:
            continue

        res_at = mkt_row.get("resolved_at")
        if pd.isna(res_at) or not res_at:
            continue
        res_ts = int(float(res_at)) if isinstance(res_at, (int, float, str)) else 0

        for leaked_ts in leaked_by_cat[cat]:
            if abs(res_ts - leaked_ts) <= WINDOW_SEC:
                seen.add(cid)
                controls.append(
                    {
                        "market_id_prefix": cid[:10],
                        "condition_id": cid,
                        "is_leaked": 0,
                        "case_id": "",
                        "case_title": "",
                        "category": cat,
                        "news_timestamp": res_ts,
                        "window_start_24h": res_ts - 86_400,
                        "window_start_48h": res_ts - 172_800,
                        "resolution_outcome": "",
                        "trade_available": True,
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
    leaked_prefixes = {r["market_id_prefix"] for r in leaked_rows}
    print(f"  {len(leaked_rows)} leaked market rows ({len(leaked_prefixes)} unique prefixes)")

    # Filter to only markets with retrievable trade history
    available_rows = [r for r in leaked_rows if r["trade_available"]]
    print(f"  {len(available_rows)} with retrievable trade history")

    print("Loading Polymarket market metadata ...")
    all_markets = load_polymarket_market_metadata(POLY_DIR)
    print(f"  {len(all_markets)} markets found in downloaded data")

    target_controls = len(available_rows) * CONTROL_MULTIPLIER
    print(f"Selecting up to {target_controls} matched control markets ...")
    control_rows = select_controls(available_rows, all_markets, leaked_prefixes, target_controls)
    print(f"  {len(control_rows)} control markets selected")

    # Use all leaked rows (not just trade-available) for the index — filters happen in profile_builder
    all_rows = leaked_rows + control_rows
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["market_id_prefix", "is_leaked"])

    out = PROCESSED_DIR / "market_index.parquet"
    df.to_parquet(out, index=False)
    print(f"\nMarket index saved → {out}")
    print(f"  Leaked:  {df['is_leaked'].sum()}")
    print(f"  Control: {(df['is_leaked'] == 0).sum()}")
    print(f"  Total:   {len(df)}")


if __name__ == "__main__":
    main()
