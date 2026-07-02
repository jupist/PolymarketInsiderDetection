"""
fetch_ffic.py
Download the ForesightFlow Insider Cases (FFIC) inventory from GitHub.

Files downloaded:
  - ffic-v1.jsonl   (one JSON object per case, 8 cases)
  - ffic-v1.csv     (flat per-market view, 24 rows)
  - README.md       (dataset documentation)

Output: data/raw/ffic/
"""

import json
import pathlib
import sys
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


def print_summary(cases: list[dict]) -> None:
    print(f"\nFFIC Summary: {len(cases)} cases")
    total_markets = sum(len(c.get("markets", [])) for c in cases)
    print(f"Total markets: {total_markets}")
    for c in cases:
        mkt_count = len(c.get("markets", []))
        print(
            f"  {c['case_id']} | {c.get('date', '?')} | {c.get('category', '?')} "
            f"| {mkt_count} market(s) | {c.get('title', '')[:60]}"
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
    if jsonl_path.exists():
        cases = load_cases(jsonl_path)
        print_summary(cases)
    else:
        print("[error] ffic-v1.jsonl not found — check network access", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
