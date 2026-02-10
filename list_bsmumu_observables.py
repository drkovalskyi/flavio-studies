#!/usr/bin/env python3
"""
List flavio b->s mu mu observables and group them into:
- B0->K*0mumu
- B+->Kmumu
- Bs->phimumu

Also split each channel into:
- differential branching fraction type observables (dBR, BR, etc)
- angular type observables (P5p, FL, AFB, S_i, etc)

Run:
  python list_bsmumu_observables.py

Optional:
  python list_bsmumu_observables.py --json out.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict

import flavio


CHANNEL_PATTERNS = {
    "B0->K*0mumu": [
        r"\bB0->K\*0mumu\b",
        r"\bB0->K\*mumu\b",
        r"\bBd->K\*0mumu\b",
    ],
    "B+->Kmumu": [
        r"\bB\+->Kmumu\b",
        r"\bB\+->K\+mumu\b",
    ],
    "Bs->phimumu": [
        r"\bBs->phimumu\b",
    ],
}

# Heuristics for splitting into "rate-like" vs "angular-like"
ANGULAR_KEYWORDS = [
    "P1", "P2", "P3", "P4p", "P5p", "P6p", "P8p",
    "S3", "S4", "S5", "S7", "S8", "S9",
    "A_FB", "AFB", "F_L", "FL", "A9", "A7", "A8",
    "J_", "dGamma", "A_T", "AT", "Q_",
]
RATE_KEYWORDS = [
    "dBR", "BR", "dG", "dGamma", "Gamma", "d^2BR", "d^2Gamma",
]


def matches_any(name: str, patterns) -> bool:
    for pat in patterns:
        if re.search(pat, name):
            return True
    return False


def classify_observable(name: str) -> str:
    # Prefer angular if any angular keyword is present
    for kw in ANGULAR_KEYWORDS:
        if kw in name:
            return "angular"
    # Otherwise rate-like if any rate keyword present
    for kw in RATE_KEYWORDS:
        if kw in name:
            return "rate"
    # Fallback: treat as "other"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="Write grouped observable names to this JSON file")
    args = ap.parse_args()

    obs_names = sorted(flavio.Observable.instances.keys())

    grouped = {ch: defaultdict(list) for ch in CHANNEL_PATTERNS.keys()}
    unassigned = []

    for o in obs_names:
        assigned = False
        for ch, pats in CHANNEL_PATTERNS.items():
            if matches_any(o, pats):
                grouped[ch][classify_observable(o)].append(o)
                assigned = True
        if not assigned:
            # Keep only b->s mumu related names for the unassigned list to avoid noise
            if "mumu" in o and ("b" in o or "B" in o):
                unassigned.append(o)

    # Pretty print
    for ch in CHANNEL_PATTERNS.keys():
        print("\n" + "=" * 80)
        print(ch)
        print("=" * 80)
        for cls in ["rate", "angular", "other"]:
            items = grouped[ch][cls]
            print(f"\n[{cls}]  ({len(items)})")
            for name in items:
                print("  " + name)

    print("\n" + "=" * 80)
    print(f"Unassigned but containing 'mumu' and B/b: ({len(unassigned)})")
    print("=" * 80)
    for name in unassigned[:200]:
        print("  " + name)
    if len(unassigned) > 200:
        print(f"  ... ({len(unassigned) - 200} more)")

    if args.json:
        out = {
            ch: {cls: grouped[ch][cls] for cls in grouped[ch].keys()}
            for ch in grouped.keys()
        }
        out["unassigned_mumu_B"] = unassigned
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON to: {args.json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
