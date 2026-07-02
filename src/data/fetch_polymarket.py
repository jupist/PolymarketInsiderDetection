"""
fetch_polymarket.py
Stream the Polymarket-v1 daily_aligned/ layer from HuggingFace and filter
to only the markets of interest (FFIC leaked markets + matched controls).

Strategy:
  1. Load FFIC cases to get leaked condition_ids
  2. Stream HuggingFace daily_aligned parquet files day by day using DuckDB
  3. Filter rows where condition_id is in target set
  4. Write filtered data to data/raw/polymarket/daily_aligned/

Memory usage: ~O(batch) — never loads the full 1.2B rows into memory.

Output: data/raw/polymarket/daily_aligned/{condition_id}.parquet (one file per market)
"""

import json
import pathlib
import sys
import time
from typing import Optional

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, list_repo_tree

REPO_ID = "TimeSeventeen/Polymarket-v1"
HF_PREFIX = f"datasets/{REPO_ID}/resolve/main/daily_aligned"

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
FFIC_DIR = RAW_DIR / "ffic"
OUT_DIR = RAW_DIR / "polymarket" / "daily_aligned"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"


def load_ffic_condition_ids(ffic_dir: pathlib.Path) -> set[str]:
    """Parse FFIC jsonl and return all condition_id_full values."""
    jsonl = ffic_dir / "ffic-v1.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"FFIC data not found at {jsonl}. Run fetch_ffic.py first.")

    condition_ids: set[str] = set()
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            for mkt in case.get("markets", []):
                cid = mkt.get("condition_id_full") or mkt.get("condition_id") or mkt.get("market_id")
                if cid:
                    # Normalize: some entries use short market_id (0x...) as condition_id
                    condition_ids.add(cid.lower())
    return condition_ids


def list_daily_aligned_files() -> list[str]:
    """Return sorted list of daily_aligned parquet URLs from HuggingFace."""
    fs = HfFileSystem()
    try:
        files = fs.glob(f"{REPO_ID}/daily_aligned/**/*.parquet", repo_type="dataset")
    except Exception as e:
        print(f"[warn] HfFileSystem glob failed ({e}), falling back to API listing", file=sys.stderr)
        files = []
    return sorted(files)


def fetch_and_filter_file(
    hf_url: str,
    target_ids: set[str],
    con: duckdb.DuckDBPyConnection,
) -> Optional[pd.DataFrame]:
    """
    Download a single daily_aligned parquet file via DuckDB HTTP read,
    filter to target condition_ids, and return a DataFrame (or None if empty).
    """
    # Build HTTPS URL for direct parquet access
    # HF path: TimeSeventeen/Polymarket-v1/daily_aligned/date=2024-10-01/part-0.parquet
    # URL:     https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1/resolve/main/daily_aligned/...
    rel_path = hf_url.replace(f"{REPO_ID}/", "")
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{rel_path}"

    # Build condition_id filter string for DuckDB
    ids_sql = ", ".join(f"'{cid}'" for cid in target_ids)
    query = f"""
        SELECT *
        FROM read_parquet('{url}')
        WHERE lower(condition_id) IN ({ids_sql})
           OR lower(condition_id) LIKE ANY (
               SELECT '%' || unnest([{ids_sql}]) || '%'
           )
    """
    # Simpler query — direct match on condition_id
    query = f"""
        SELECT *
        FROM read_parquet('{url}')
        WHERE lower(condition_id) IN ({ids_sql})
    """
    try:
        df = con.execute(query).df()
        if len(df) > 0:
            return df
    except Exception as e:
        print(f"  [skip] {rel_path}: {e}", file=sys.stderr)
    return None


def save_by_market(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Append rows to per-market parquet files."""
    if "condition_id" not in df.columns:
        return
    for cid, grp in df.groupby("condition_id"):
        safe_name = str(cid).replace("/", "_").replace(":", "_")[:64]
        dest = out_dir / f"{safe_name}.parquet"
        if dest.exists():
            existing = pq.read_table(dest)
            combined = pa.concat_tables([existing, pa.Table.from_pandas(grp)])
            pq.write_table(combined, dest, compression="snappy")
        else:
            pq.write_table(pa.Table.from_pandas(grp), dest, compression="snappy")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FFIC condition IDs ...")
    target_ids = load_ffic_condition_ids(FFIC_DIR)
    print(f"  {len(target_ids)} condition IDs from FFIC")
    # Save target IDs for reference
    (PROCESSED_DIR / "target_condition_ids.json").write_text(
        json.dumps(sorted(target_ids), indent=2)
    )

    print("\nListing daily_aligned files on HuggingFace ...")
    files = list_daily_aligned_files()
    if not files:
        print(
            "[error] No files found. Make sure huggingface_hub is installed "
            "and the dataset is accessible.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Found {len(files):,} parquet files")

    con = duckdb.connect()
    # Enable HTTP extension for remote parquet reads
    con.execute("INSTALL httpfs; LOAD httpfs;")

    total_rows = 0
    matched_files = 0
    print(f"\nStreaming and filtering {len(files):,} files ...")

    for i, hf_path in enumerate(files, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(files)}] processed so far — {total_rows:,} rows matched")

        df = fetch_and_filter_file(hf_path, target_ids, con)
        if df is not None and len(df) > 0:
            save_by_market(df, OUT_DIR)
            total_rows += len(df)
            matched_files += 1
            # Small delay to be polite to HF servers
            time.sleep(0.1)

    con.close()
    print(f"\nDone. {total_rows:,} rows across {matched_files} daily files → {OUT_DIR}")


if __name__ == "__main__":
    main()
