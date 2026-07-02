"""
fetch_control_markets.py
Identify ~75 control markets matching the categories and resolution date windows of our FFIC leaked cases,
and download their pre-news trades from HuggingFace daily_aligned partitions.

Super-optimized batching:
  1. Group leaked markets by academic category and case date (only ~8 clusters total).
  2. For each cluster, scan 2 daily parquet files over HTTP to select ~10 non-leaked candidate markets.
  3. For each cluster, batch-query 10 daily parquet files before the resolution date to download all
     pre-news trades for all selected control markets in that cluster simultaneously.
  4. Save each control market's trades to data/raw/polymarket/daily_aligned/{prefix}.parquet.
"""

import json
import pathlib
import sys
from datetime import datetime, timezone, timedelta

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
FFIC_DIR = RAW_DIR / "ffic"
OUT_DIR = RAW_DIR / "polymarket" / "daily_aligned"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"

CONTROL_MULTIPLIER = 3


def parse_timestamp(ts_str: str) -> int:
    if not ts_str:
        return 0
    ts_str = str(ts_str).strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return 0


def load_leaked_info() -> tuple[list[dict], set[str]]:
    jsonl = FFIC_DIR / "ffic-v1.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"FFIC jsonl not found at {jsonl}.")
    
    leaked_markets = []
    leaked_prefixes = set()
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_ts = parse_timestamp(case.get("date", ""))
            for mkt in case.get("markets", []):
                pfx = mkt.get("market_id_prefix", "").lower()
                if pfx:
                    leaked_prefixes.add(pfx)
                    res_ts = parse_timestamp(mkt.get("resolution_date", "")) or case_ts
                    leaked_markets.append({
                        "prefix": pfx,
                        "category": case.get("category", ""),
                        "news_ts": res_ts,
                        "date": datetime.fromtimestamp(res_ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                        "trade_available": mkt.get("trade_history_available", False) in (True, "partial")
                    })
    return leaked_markets, leaked_prefixes


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    print("Loading leaked market metadata ...", flush=True)
    leaked_markets, leaked_prefixes = load_leaked_info()
    retrievable = [m for m in leaked_markets if m["trade_available"]]
    print(f"  {len(leaked_markets)} leaked markets ({len(retrievable)} retrievable)", flush=True)
    
    cat_map = {
        "regulatory_decision": ("Politics", "Trump", "pardon", "court cases", "Bitcoin", "Crypto", "SEC"),
        "military_geopolitics": ("World", "Geopolitics", "Iran", "Middle East", "Israel", "Venezuela", "Ukraine", "maduro"),
        "corporate_disclosure": ("Business", "AI", "Science", "Google", "Tech", "Pop Culture", "Music")
    }
    
    # Group retrievable markets by (category, date)
    clusters = {}
    for m in retrievable:
        key = (m["category"], m["date"])
        clusters.setdefault(key, []).append(m)
        
    print(f"\nProcessing {len(clusters)} case clusters to discover and download control markets ...", flush=True)
    
    seen_prefixes = set(leaked_prefixes)
    total_control_rows = 0
    total_controls_saved = 0
    
    for i, ((acad_cat, date_str), mkts) in enumerate(clusters.items(), 1):
        n_needed = len(mkts) * CONTROL_MULTIPLIER
        print(f"\n--- Cluster {i}/{len(clusters)}: {acad_cat} around {date_str} (need {n_needed} controls) ---", flush=True)
        
        poly_tags = cat_map.get(acad_cat, ("World", "Politics", "Business"))
        tags_sql = ", ".join(f"'{t}'" for t in poly_tags)
        
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            dt = datetime(2024, 11, 5)
            
        # Discover candidates on resolution date and 7 days prior
        sample_dates = [
            dt.strftime("%Y-%m-%d"),
            (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        ]
        
        candidates = []
        for sd in sample_dates:
            url = f"https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1/resolve/main/daily_aligned/{sd}.parquet"
            query = f"""
                SELECT DISTINCT 
                    lower(condition_id) as condition_id,
                    ANY_VALUE(category) as poly_category,
                    ANY_VALUE(market_slug) as slug,
                    ANY_VALUE(resolved_at) as resolved_at
                FROM read_parquet('{url}')
                WHERE category IN ({tags_sql})
                GROUP BY 1
                LIMIT {n_needed * 2 + 10}
            """
            try:
                df = con.execute(query).df()
                for _, row in df.iterrows():
                    cid = str(row["condition_id"]).lower()
                    pfx = cid[:10]
                    if pfx in seen_prefixes or cid in seen_prefixes:
                        continue
                    seen_prefixes.add(pfx)
                    
                    res_at = row["resolved_at"]
                    res_ts = parse_timestamp(str(res_at)) if pd.notna(res_at) else mkts[0]["news_ts"]
                    if res_ts == 0:
                        res_ts = mkts[0]["news_ts"]
                        
                    candidates.append({
                        "market_id_prefix": pfx,
                        "condition_id": cid,
                        "category": acad_cat,
                        "poly_category": str(row["poly_category"]),
                        "slug": str(row["slug"]),
                        "resolved_at": res_ts
                    })
                    if len(candidates) >= n_needed:
                        break
            except Exception as e:
                print(f"  [warn] Discovery failed for {sd}: {e}", flush=True)
            if len(candidates) >= n_needed:
                break
                
        selected = candidates[:n_needed]
        print(f"  Found {len(selected)} candidate control markets.", flush=True)
        if not selected:
            continue
            
        # Batch download trades across 10 daily files prior to cluster date
        pfx_list = [c["market_id_prefix"] for c in selected]
        pfx_sql = " OR ".join(f"lower(condition_id) LIKE '{p}%'" for p in pfx_list)
        
        dfs = []
        print(f"  Downloading pre-news trades across 5 daily partitions ...", flush=True)
        for d_offset in range(5, 0, -1):
            target_dt = dt - timedelta(days=d_offset)
            d_str = target_dt.strftime("%Y-%m-%d")
            url = f"https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1/resolve/main/daily_aligned/{d_str}.parquet"
            query = f"""
                SELECT *
                FROM read_parquet('{url}')
                WHERE {pfx_sql}
            """
            try:
                df = con.execute(query).df()
                if len(df) > 0:
                    dfs.append(df)
            except Exception:
                continue
                
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            # Save by prefix
            def get_prefix(cid):
                cid_lower = str(cid).lower()
                for c in selected:
                    if cid_lower.startswith(c["market_id_prefix"]):
                        return c["market_id_prefix"], c["category"], c["resolved_at"]
                return cid_lower[:10], acad_cat, mkts[0]["news_ts"]
                
            for _, grp in combined.groupby(combined["condition_id"].apply(lambda c: str(c).lower()[:10])):
                # Match back to control metadata
                matched_pfx = str(grp["condition_id"].iloc[0]).lower()[:10]
                matched_ctrl = next((c for c in selected if matched_pfx.startswith(c["market_id_prefix"])), selected[0])
                
                grp = grp.copy()
                grp["category"] = matched_ctrl["category"]
                grp["resolved_at"] = matched_ctrl["resolved_at"]
                
                safe_name = matched_ctrl["market_id_prefix"].replace("/", "_")
                dest = OUT_DIR / f"{safe_name}.parquet"
                
                if dest.exists():
                    existing = pq.read_table(dest)
                    merged = pa.concat_tables([existing, pa.Table.from_pandas(grp, preserve_index=False)])
                    pq.write_table(merged, dest, compression="snappy")
                else:
                    pq.write_table(pa.Table.from_pandas(grp, preserve_index=False), dest, compression="snappy")
                
                total_controls_saved += 1
                total_control_rows += len(grp)
            print(f"  Saved trades for cluster — total {len(combined):,} rows across matched markets.", flush=True)
        else:
            print(f"  No trades found in sampled window.", flush=True)
            
    print(f"\nDone! Downloaded {total_control_rows:,} control rows across {total_controls_saved} control files.", flush=True)
    con.close()


if __name__ == "__main__":
    main()
