"""
fetch_ffic.py
Download the ForesightFlow Insider Cases (FFIC) inventory from GitHub.

Files downloaded:
  - ffic-v1.jsonl   (one JSON object per case, 8 cases)
  - ffic-v1.csv     (flat per-market view, 32 rows)
  - README.md       (dataset documentation)

FFIC market schema uses 'market_id_prefix' (first 8 hex chars of condition_id).
We attempt to resolve full condition_ids via Polymarket's Gamma API.

Output: data/raw/ffic/
"""

import json
import pathlib
import sys
import time
import requests

FFIC_BASE = (
    "https://raw.githubusercontent.com/ForesightFlow/datasets/main/ffic-inventory"
)
FILES = [
    "ffic-v1.jsonl",
    "ffic-v1.csv",
    "README.md",
    "MANIFEST.json",
]
GAMMA_URL = "https://gamma-api.polymarket.com/markets"

OUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw" / "ffic"


def download(filename: str, out_dir: pathlib.Path) -> pathlib.Path:
    url = f"{FFIC_BASE}/{filename}"
    dest = out_dir / filename
    if dest.exists():
        print(f"  [skip] {filename} already exists")
        return dest
    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Saved → {dest} ({len(resp.content):,} bytes)")
    return dest


def load_cases(jsonl_path: pathlib.Path) -> list[dict]:
    cases = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def resolve_condition_ids(cases: list[dict], out_dir: pathlib.Path) -> dict[str, str]:
    """
    Resolve full condition_ids from market_id_prefix values.

    Strategy:
      1. Search Polymarket Gamma API by question text label
      2. Match result whose conditionId starts with the prefix
      3. Fall back to prefix-only matching in the daily_aligned dataset at query time

    Returns mapping: {market_id_prefix: full_condition_id}
    Saves to out_dir/resolved_ids.json for caching.
    """
    resolved_path = out_dir / "resolved_ids.json"
    if resolved_path.exists():
        print(f"  Loading cached resolved IDs from {resolved_path}")
        return json.loads(resolved_path.read_text())

    resolved: dict[str, str] = {}

    for case in cases:
        for mkt in case.get("markets", []):
            prefix = mkt.get("market_id_prefix", "")
            label = mkt.get("label", "")
            if not prefix or prefix in resolved:
                continue

            try:
                resp = requests.get(
                    GAMMA_URL,
                    params={"q": label, "limit": 10},
                    timeout=15,
                )
                if resp.ok:
                    data = resp.json()
                    market_list = data if isinstance(data, list) else data.get("markets", [])
                    for m in market_list:
                        cid = (m.get("conditionId") or m.get("condition_id", "")).lower()
                        mid = (m.get("id") or m.get("market_id", "")).lower()
                        prefix_clean = prefix.lower().lstrip("0x")
                        if (cid.startswith(prefix.lower()) or mid.startswith(prefix.lower())
                                or cid.lstrip("0x").startswith(prefix_clean)):
                            resolved[prefix] = cid
                            print(f"    {prefix} → {cid[:24]}...")
                            break
                time.sleep(0.3)  # polite rate-limiting
            except Exception as e:
                print(f"  [warn] Could not resolve {prefix} ({label[:40]}): {e}", file=sys.stderr)

    resolved_path.write_text(json.dumps(resolved, indent=2))
    return resolved


def print_summary(cases: list[dict]) -> None:
    print(f"\nFFIC Summary: {len(cases)} cases")
    total_markets = sum(len(c.get("markets", [])) for c in cases)
    available = sum(
        1 for c in cases
        for m in c.get("markets", [])
        if m.get("trade_history_available") is True
    )
    print(f"Total markets: {total_markets}  (trade history available: {available})")
    for c in cases:
        mkts = c.get("markets", [])
        avail = sum(1 for m in mkts if m.get("trade_history_available") is True)
        print(
            f"  {c['case_id']} | {c.get('date', '?')} | {c.get('category', '?')} "
            f"| {len(mkts)} market(s), {avail} retrievable | {c.get('title', '')[:55]}"
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FFIC files to {OUT_DIR}\n")
    for fname in FILES:
        try:
            download(fname, OUT_DIR)
        except requests.HTTPError as e:
            print(f"  [warn] Could not download {fname}: {e}", file=sys.stderr)

    jsonl_path = OUT_DIR / "ffic-v1.jsonl"
    if not jsonl_path.exists():
        print("[error] ffic-v1.jsonl not found — check network access", file=sys.stderr)
        sys.exit(1)

    cases = load_cases(jsonl_path)
    print_summary(cases)

    print("\nAttempting to resolve full condition IDs via Polymarket Gamma API ...")
    resolved = resolve_condition_ids(cases, OUT_DIR)
    if resolved:
        print(f"  {len(resolved)} prefixes resolved to full condition IDs")
    else:
        print(
            "  [note] No IDs resolved via API — prefix matching will be used "
            "when filtering the Polymarket dataset"
        )


if __name__ == "__main__":
    main()
