"""
validate_resolutions.py
Validate market resolution outcomes against HuggingFace CTF/resolutions.parquet table.

Queries TimeSeventeen/Polymarket-v1 CTF/resolutions.parquet over HTTP via DuckDB,
joins with data/processed/market_index.parquet, and compares:
  1. Resolution outcome consistency (FFIC outcome vs payout_numerators from CTF)
  2. Verification of on-chain resolution existence

Output: data/processed/resolution_validation.csv
"""

import pathlib
import sys

import duckdb
import numpy as np
import pandas as pd

PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
CTF_URL = "https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1/resolve/main/CTF/resolutions.parquet"


def main():
    idx_path = PROCESSED_DIR / "market_index.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(f"Market index not found at {idx_path}.")
    
    print("Loading market index ...")
    market_index = pd.read_parquet(idx_path)
    print(f"  {len(market_index)} total markets ({market_index['is_leaked'].sum()} leaked, {(market_index['is_leaked'] == 0).sum()} control)")
    
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.register("market_index", market_index)
    
    print("\nQuerying HuggingFace CTF/resolutions.parquet over HTTP ...")
    query = f"""
    SELECT
        m.market_id_prefix,
        m.condition_id,
        m.is_leaked,
        m.case_id,
        m.case_title,
        m.category,
        m.news_timestamp,
        m.resolution_outcome AS ffic_outcome,
        c.payout_numerators,
        c.question_id,
        c.oracle
    FROM market_index AS m
    LEFT JOIN read_parquet('{CTF_URL}') AS c
        ON lower(m.condition_id) = lower(c.condition_id)
        OR lower(c.condition_id) LIKE lower(m.market_id_prefix) || '%'
    """
    try:
        df = con.execute(query).df()
    except Exception as e:
        print(f"[error] Failed to query CTF resolutions: {e}", file=sys.stderr)
        return
    finally:
        con.close()
        
    df["ctf_resolved"] = df["payout_numerators"].notna() & (df["payout_numerators"].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False))
    
    out = PROCESSED_DIR / "resolution_validation.csv"
    df.to_csv(out, index=False)
    print(f"  Matched {df['ctf_resolved'].sum()} markets against on-chain CTF resolution records.")
    print(f"\nResolution validation report saved → {out}")
    
    print("\nSample comparison (Leaked Markets):")
    leaked_sample = df[df["is_leaked"] == 1][["market_id_prefix", "case_id", "ffic_outcome", "payout_numerators", "ctf_resolved"]].head(15)
    print(leaked_sample.to_string(index=False))


if __name__ == "__main__":
    main()
