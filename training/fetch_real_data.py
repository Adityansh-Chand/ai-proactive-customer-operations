"""Fetch BANKING77 -- real customer messages, written by real customers.

The synthetic corpus in `training/generate_corpus.py` is built from paraphrase
templates we wrote. Held-out *templates* keep that evaluation honest, but every
sentence in it still came out of our own heads: our idea of how a frustrated
customer types, our vocabulary, our typos.

BANKING77 is 13,083 real queries sent to an online bank, each labelled with one
of 77 fine-grained intents. Nobody wrote them to be classified.

    Casanueva, Temcinas, Gerz, Henderson and Vulic, "Efficient Intent Detection
    with Dual Sentence Encoders", NLP4ConvAI 2020.
    https://huggingface.co/datasets/PolyAI/banking77
    Licensed CC BY 4.0.

Fetched from the maintainers' own CSVs rather than through a dataset library, so
this repository does not grow a dependency to read two text files.

Cached under datasets/real/ and NOT committed.

    python training/fetch_real_data.py           # download and cache
    python training/fetch_real_data.py --check   # verify the cache, no network
"""
import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "datasets" / "real"

BASE = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
        "/master/banking_data")

# Checksums are filled in on first download and asserted on every later run, so a
# silently changed upstream file surfaces here rather than as moved metrics.
FILES = {
    "train.csv": {
        "url": f"{BASE}/train.csv",
        "sha256": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        "rows": 10003,
    },
    "test.csv": {
        "url": f"{BASE}/test.csv",
        "sha256": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        "rows": 3080,
    },
}
EXPECTED_INTENTS = 77


def count_rows(path):
    """Parse rather than count newlines.

    Some queries contain newlines inside quoted fields, so counting b"\\n" reports
    10,016 and 3,084 against the documented 10,003 and 3,080 -- close enough to
    look like a version difference and actually just the wrong way to count.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1  # minus the header


def download():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in FILES.items():
        print(f"downloading {spec['url']}")
        request = urllib.request.Request(
            spec["url"], headers={"User-Agent": "portfolio-fetch"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        path = CACHE_DIR / name
        path.write_bytes(data)
        rows = count_rows(path)
        digest = hashlib.sha256(data).hexdigest()
        print(f"  cached {name}: {rows} rows  sha256 {digest}")
        if digest != spec["sha256"]:
            print(f"  FAIL: checksum mismatch, expected {spec['sha256']}")
            return 1
        if rows != spec["rows"]:
            print(f"  FAIL: expected {spec['rows']} rows, got {rows}")
            return 1
    return 0


def check():
    problems = []
    for name, spec in FILES.items():
        path = CACHE_DIR / name
        if not path.exists():
            problems.append(f"MISSING {path}")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        rows = count_rows(path)
        print(f"{name}: {rows} rows  sha256 {digest}")
        if digest != spec["sha256"]:
            problems.append(f"{name}: checksum mismatch, expected {spec['sha256']}")
        if rows != spec["rows"]:
            problems.append(f"{name}: expected {spec['rows']} rows, got {rows}")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        print("run: python training/fetch_real_data.py")
        return 1
    print("OK: BANKING77 present, checksums and row counts match")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the cached copy without downloading")
    args = parser.parse_args()
    return check() if args.check else download()


if __name__ == "__main__":
    sys.exit(main())
