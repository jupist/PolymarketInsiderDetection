"""
fetch_polymarket.py
Stream the Polymarket-v1 daily_aligned/ layer from HuggingFace and filter
to only the markets of interest (FFIC leaked markets + matched controls).

Strategy:
  1. Load FFIC cases to get market_id_prefix values (and any resolved full condition_ids)
  2. Stream HuggingFace daily_aligned parquet files via DuckDB HTTP reads
  3. Filter rows where condition_id starts with any FFIC prefix OR matches full resolved IDs
  4. Write filtered data to data/raw/polymarket/daily_aligned/{prefix}.parquet

Memory usage: O(batch) — never loads the full 1.2B rows at once.

Output: data/raw/polymarket/daily_aligned/{market_id_prefix}.parquet
"""

import json
import pathlib
import sys
import time

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO_ID = "TimeSeventeen/Polymarket-v1"

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"
FFIC_DIR = RAW_DIR / "ffic"
OUT_DIR = RAW_DIR / "polymarket" / "daily_aligned"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"


def load_ffic_prefixes(ffic_dir: pathlib.Path) -> tuple[list[str], dict[str, str]]:
    """
    Parse FFIC jsonl and return:
      - list of market_id_prefix values
      - dict mapping prefix → full condition_id (if resolved)
    """
    jsonl = ffic_dir / "ffic-v1.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(
            f"FFIC data not found at {jsonl}. Run fetch_ffic.py first."
        )

    prefixes: list[str] = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            for mkt in case.get("markets", []):
                pfx = mkt.get("market_id_prefix", "")
                if pfx and pfx not in prefixes:
                    prefixes.append(pfx.lower())

    # Load resolved full IDs if available
    resolved_path = ffic_dir / "resolved_ids.json"
    resolved: dict[str, str] = {}
    if resolved_path.exists():
        resolved = {k.lower(): v.lower() for k, v in json.loads(resolved_path.read_text()).items()}

    return prefixes, resolved


def list_daily_aligned_files() -> list[str]:
    """Return sorted list of daily_aligned parquet file paths from HuggingFace repo."""
    fs = HfFileSystem()
    try:
        files = fs.glob(
            f"datasets/{REPO_ID}/daily_aligned/**/*.parquet",
        )
        return sorted(files)
    except Exception as e:
        print(f"[error] Could not list HF files: {e}", file=sys.stderr)
        return []


def build_filter_sql(prefixes: list[str], resolved: dict[str, str], url: str) -> str:
    """
    Build DuckDB SQL to filter daily_aligned parquet by condition_id.
    Uses:
      - Exact match on full condition_ids (if resolved)
      - startswith() LIKE match on market_id_prefix values
    """
    conditions: list[str] = []

    # Full condition_id matches
    full_ids = list(resolved.values())
    if full_ids:
        ids_sql = ", ".join(f"'{cid}'" for cid in full_ids)
        conditions.append(f"lower(condition_id) IN ({ids_sql})")

    # Prefix LIKE matches (condition_id starts with the 8-char prefix)
    for pfx in prefixes:
        pfx_clean = pfx.lower()
        conditions.append(f"lower(condition_id) LIKE '{pfx_clean}%'")

    where_clause = " OR ".join(conditions) if conditions else "1=0"

    return f"""
        SELECT *
        FROM read_parquet('{url}')
        WHERE {where_clause}
    """


def save_by_prefix(df: pd.DataFrame, out_dir: pathlib.Path, prefixes: list[str]) -> None:
    """Write rows to per-prefix parquet files (appending if file already exists)."""
    if "condition_id" not in df.columns:
        return

    def get_prefix(cid: str) -> str:
        cid_lower = str(cid).lower()
        for pfx in prefixes:
            if cid_lower.startswith(pfx):
                return pfx
        return cid_lower[:10]  # fallback

    df = df.copy()
    df["_prefix"] = df["condition_id"].apply(get_prefix)

    for pfx, grp in df.groupby("_prefix"):
        grp = grp.drop(columns=["_prefix"])
        safe_name = pfx.replace("/", "_")
        dest = out_dir / f"{safe_name}.parquet"
        if dest.exists():
            existing = pq.read_table(dest)
            combined = pa.concat_tables([existing, pa.Table.from_pandas(grp, preserve_index=False)])
            pq.write_table(combined, dest, compression="snappy")
        else:
            pq.write_table(pa.Table.from_pandas(grp, preserve_index=False), dest, compression="snappy")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FFIC market prefixes ...")
    prefixes, resolved = load_ffic_prefixes(FFIC_DIR)
    print(f"  {len(prefixes)} market_id_prefix values from FFIC")
    print(f"  {len(resolved)} resolved to full condition IDs")
    # Save for reference
    (PROCESSED_DIR / "ffic_prefixes.json").write_text(
        json.dumps({"prefixes": prefixes, "resolved": resolved}, indent=2)
    )

    print("\nListing daily_aligned files on HuggingFace ...")
    files = list_daily_aligned_files()
    if not files:
        print(
            "[error] No files found. Check huggingface_hub installation.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Found {len(files):,} parquet partitions")

    con = duckdb.connect()
    # Enable HTTP extension for remote parquet access
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET threads=4;")

    total_rows = 0
    matched_files = 0

    print(f"\nStreaming and filtering {len(files):,} files (this may take a while) ...")

    for i, hf_path in enumerate(files, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(files)}] {total_rows:,} rows matched in {matched_files} files")

        # Build HTTPS URL for direct parquet access
        # hf_path: datasets/TimeSeventeen/Polymarket-v1/daily_aligned/...parquet
        rel_path = hf_path.replace(f"datasets/{REPO_ID}/", "")
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{rel_path}"

        sql = build_filter_sql(prefixes, resolved, url)
        try:
            df = con.execute(sql).df()
            if len(df) > 0:
                save_by_prefix(df, OUT_DIR, prefixes)
                total_rows += len(df)
                matched_files += 1
                time.sleep(0.05)  # light throttle
        except Exception as e:
            print(f"  [skip] {rel_path}: {e}", file=sys.stderr)

    con.close()
    print(f"\nDone: {total_rows:,} rows across {matched_files} files → {OUT_DIR}")
    saved_files = list(OUT_DIR.glob("*.parquet"))
    print(f"  Saved {len(saved_files)} per-market parquet files")


if __name__ == "__main__":
    main()
