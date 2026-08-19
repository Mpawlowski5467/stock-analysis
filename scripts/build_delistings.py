"""Build the delisting/deregistration ledger from EDGAR master indexes.

  uv run python scripts/build_delistings.py 2011q1:2026q3

Prefer `ops.py delistings` for a routine refresh — same rebuild, but with the range
computed to the live quarter and the destroy-nothing guards around it (temp-file
commit, empty/shrunken scans rejected, previous ledger kept as .bak). This script is
the raw override for an explicit range.

MIND THE RANGE: the ledger keeps the earliest event per CIK across exactly the quarters
you pass and OVERWRITES the parquet with that result. It does not merge. Passing only
the quarters you think are new replaces fifteen years of history with those quarters.
"""

import sys

from stockscan.edgar.delistings import LEDGER_PATH, build_delisting_ledger
from stockscan.edgar.fsds import iter_quarters


def expand(args: list[str]) -> list[str]:
    quarters: list[str] = []
    for a in args:
        if ":" in a:
            start, end = a.split(":", 1)
            quarters.extend(iter_quarters(start, end))
        else:
            quarters.append(a)
    return quarters


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: build_delistings.py <quarter|start:end> [...]")
        return 1
    quarters = expand(argv)
    print(f"Scanning {len(quarters)} quarter(s) for delisting/deregistration events ...")
    n = build_delisting_ledger(quarters)
    print(f"Wrote {n:,} delisted/deregistered CIKs -> {LEDGER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
