"""
run_pipeline.py
Master executable pipeline for Polymarket Insider Detection.

Executes the end-to-end workflow:
  1. Data Acquisition (FFIC ground truth & Polymarket control/leaked trades)
  2. Market Index & Pre-News Window Definition
  3. Behavioral Profile Engineering (13-dim feature vectors)
  4. Anomaly Detection Modeling (Isolation Forest & PyTorch Autoencoder)
  5. Evaluation & CTF Resolution Validation

Usage:
    python run_pipeline.py               # Run pipeline (skips download if data exists)
    python run_pipeline.py --force-data  # Force re-download of raw datasets
    python run_pipeline.py --eval-only   # Skip training and run evaluation metrics only
"""

import argparse
import pathlib
import sys
import time

from src.data import build_market_index, fetch_control_markets, fetch_ffic, fetch_polymarket
from src.eval import metrics, validate_resolutions
from src.features import profile_builder
from src.models import autoencoder, isolation_forest

ROOT_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def print_step(step_num: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*60}\n")


def check_data_exists() -> bool:
    ffic_exists = (RAW_DIR / "ffic" / "ffic-v1.jsonl").exists()
    poly_exists = (RAW_DIR / "polymarket" / "daily_aligned").exists() and any(
        (RAW_DIR / "polymarket" / "daily_aligned").glob("*.parquet")
    )
    return ffic_exists and poly_exists


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Insider Detection Pipeline")
    parser.add_argument(
        "--force-data", action="store_true", help="Force re-download of raw datasets"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training and run evaluation metrics only"
    )
    args = parser.parse_args()

    start_time = time.time()
    print("\nStarting Polymarket Insider Detection Pipeline...")

    if args.eval_only:
        print_step(5, "Evaluation & Resolution Validation")
        metrics.load_and_evaluate()
        validate_resolutions.main()
        print(f"\nPipeline completed in {time.time() - start_time:.2f}s")
        return

    # Step 1: Data Acquisition
    print_step(1, "Data Acquisition (FFIC & Polymarket Trades)")
    if args.force_data or not check_data_exists():
        print("Fetching FFIC insider case inventory...")
        fetch_ffic.main()
        print("\nFetching Polymarket pre-filtered trade partitions...")
        fetch_polymarket.main()
        print("\nFetching matched control market trade candidates...")
        fetch_control_markets.main()
    else:
        print("[skip] Raw datasets already exist. Skipping download (use --force-data to re-fetch).")

    # Step 2: Market Index
    print_step(2, "Master Market Indexing & Pre-News Window Definition")
    build_market_index.main()

    # Step 3: Feature Engineering
    print_step(3, "Behavioral Profile Engineering")
    profile_builder.main()

    # Step 4: Model Training & Scoring
    print_step(4, "Anomaly Detection Modeling (IF & Autoencoder)")
    print("Running Isolation Forest baseline model...")
    isolation_forest.run()
    print("\nRunning PyTorch Autoencoder baseline model...")
    autoencoder.run()

    # Step 5: Evaluation
    print_step(5, "Evaluation & CTF Resolution Validation")
    metrics.load_and_evaluate()
    validate_resolutions.main()

    elapsed = time.time() - start_time
    print(f"\nEnd-to-end pipeline completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
