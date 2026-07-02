# Polymarket Insider Detection

Detecting informed (insider) trading on Polymarket using on-chain trade data and confirmed leak cases.

## Overview

This project builds an end-to-end pipeline to identify wallets that traded on non-public information on Polymarket. It uses:

- **[Polymarket-v1](https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1)** — 1.2B on-chain trades (2022–2026) with ground-truth buy/sell direction
- **[FFIC Inventory](https://github.com/ForesightFlow/datasets/tree/main/ffic-inventory)** — 8 confirmed insider-trading episodes across 24 Polymarket markets

## Approach

1. **Data acquisition** — filter Polymarket trades to known leaked markets + matched controls
2. **Window definition** — isolate pre-news trading (24h/48h before each leak event)
3. **Feature engineering** — per-trader behavioral profiles (bet size, direction bias, concentration, timing)
4. **Anomaly detection** — unsupervised models trained only on normal traders:
   - Dense Autoencoder (reconstruction error as anomaly score)
   - Isolation Forest (path-length anomaly score)
5. **Evaluation** — Precision@K, ROC-AUC, PR-AUC, and Leave-One-Case-Out generalization against FFIC ground truth

## Project Structure

```
data/
  raw/
    ffic/              # ForesightFlow Insider Cases (downloaded from GitHub)
    polymarket/        # Filtered daily_aligned/ parquet files (from HuggingFace)
  processed/
    market_index.parquet
    trades_filtered.parquet
    trader_profiles.parquet

notebooks/
  01_data_acquisition.ipynb
  02_eda.ipynb
  03_feature_engineering.ipynb
  04_autoencoder.ipynb
  05_isolation_forest.ipynb
  06_evaluation.ipynb

src/
  data/
    fetch_ffic.py
    fetch_polymarket.py
    build_market_index.py
  features/
    profile_builder.py
  models/
    autoencoder.py
    isolation_forest.py
  eval/
    metrics.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Download

```bash
python src/data/fetch_ffic.py
python src/data/fetch_polymarket.py
python src/data/build_market_index.py
```

## Run Pipeline

Open notebooks in order (01 → 06), or run scripts directly from `src/`.

## References

- Polymarket-v1: arXiv:2606.04217
- FFIC Inventory: arXiv:2605.00493 (Nechepurenko 2026)
- ILS Framework: arXiv:2605.00459
