#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import flavio
import yaml


def load_db():
    p = Path(flavio.__file__).resolve().parent / "data" / "measurements.yml"
    if not p.exists():
        raise FileNotFoundError(f"Could not find measurements.yml at {p}")
    db = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(db, dict):
        raise RuntimeError("measurements.yml did not parse as a dict")
    return db


def pub_key(db, mname):
    """Return a publication identifier for grouping (inspire > arxiv > mname)."""
    meta = db.get(mname)
    if isinstance(meta, dict):
        return meta.get("inspire") or meta.get("arxiv") or meta.get("eprint") or mname
    return mname


def format_pub(db, mnames):
    """Format publication info from the first measurement entry that has metadata."""
    for mname in mnames:
        meta = db.get(mname)
        if not isinstance(meta, dict):
            continue
        lines = []
        exp = meta.get("experiment")
        arxiv = meta.get("arxiv") or meta.get("eprint")
        inspire = meta.get("inspire")
        title = meta.get("description") or meta.get("title")
        if exp:
            lines.append(f"  experiment: {exp}")
        if title:
            lines.append(f"  title: {title}")
        if arxiv:
            lines.append(f"  arXiv: {arxiv}")
        if inspire:
            lines.append(f"  inspire: {inspire}")
        if lines:
            return "\n".join(lines)
    return "  (no metadata)"


def format_observables(keys):
    """Group keys by observable name; one line per observable with a list of bins."""
    by_obs = defaultdict(list)
    for key in keys:
        if isinstance(key, tuple):
            by_obs[key[0]].append(key[1:])
        else:
            by_obs[key]  # unbinned – just register it

    lines = []
    for obs in sorted(by_obs):
        bins = by_obs[obs]
        if bins:
            # deduplicate and sort
            bins = sorted(set(bins))
            bin_strs = [f"[{lo}, {hi}]" for lo, hi in bins]
            lines.append(f"    {obs}: {', '.join(bin_strs)}")
        else:
            lines.append(f"    {obs}")
    return "\n".join(lines)


def collect_by_publication(final_state, db):
    """For a final state, return {pub_id: (mnames, all_keys)} grouped by publication."""
    # First pass: collect per measurement
    meas_keys = {}
    for mname, m in flavio.Measurement.instances.items():
        try:
            keys = m.all_parameters
        except Exception:
            continue
        matched = [k for k in keys
                    if final_state in (k[0] if isinstance(k, tuple) else k)]
        if matched:
            meas_keys[mname] = matched

    # Second pass: group by publication
    pubs = defaultdict(lambda: ([], []))  # pub_id -> (mnames, keys)
    for mname, keys in meas_keys.items():
        pid = pub_key(db, mname)
        pubs[pid][0].append(mname)
        pubs[pid][1].extend(keys)

    return pubs


def main():
    targets = [
        "B0->K*mumu",
        "Bs->phimumu",
    ]

    db = load_db()
    print("Loaded measurements.yml entries:", len(db))

    for fs in targets:
        print("\n" + "=" * 100)
        print(f"Final state: {fs}")
        pubs = collect_by_publication(fs, db)
        print(f"N publications: {len(pubs)}")
        for pid in sorted(pubs):
            mnames, keys = pubs[pid]
            print(f"\n- {pid}")
            print(format_pub(db, mnames))
            print("  Observables:")
            print(format_observables(keys))


if __name__ == "__main__":
    main()
