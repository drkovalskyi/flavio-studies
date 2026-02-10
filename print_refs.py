#!/usr/bin/env python3
import sys
from pathlib import Path

import flavio
import yaml


def measurement_names_for_key(key):
    out = set()
    for mname, m in flavio.Measurement.instances.items():
        try:
            if key in m.all_parameters:
                out.add(mname)
        except Exception:
            pass
    return sorted(out)


def load_db():
    p = Path(flavio.__file__).resolve().parent / "data" / "measurements.yml"
    if not p.exists():
        raise FileNotFoundError(f"Could not find measurements.yml at {p}")
    db = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(db, dict):
        raise RuntimeError("measurements.yml did not parse as a dict of measurement_name -> metadata")
    return db


def print_measurement_ref(db, mname):
    meta = db.get(mname)
    if meta is None:
        print("  not found in measurements.yml")
        return
    if not isinstance(meta, dict):
        print("  unexpected entry type:", type(meta))
        return

    exp = meta.get("experiment")
    inspire = meta.get("inspire")
    arxiv = meta.get("arxiv") or meta.get("eprint")
    doi = meta.get("doi")
    title = meta.get("description") or meta.get("title")

    if exp:
        print("  experiment:", exp)
    if title:
        print("  title/desc:", title)
    if arxiv:
        print("  arXiv:", arxiv)
    if doi:
        print("  DOI:", doi)
    if inspire:
        print("  inspire:", inspire)
        print("  INSPIRE query: https://inspirehep.net/literature?sort=mostrecent&q=texkey%3A" + str(inspire))


def main():
    # Edit/extend as needed
    targets = [
        # K* examples
        ('<P5p>(B0->K*mumu)', 1.1, 2.5),
        ('<P5p>(B0->K*mumu)', 4.0, 6.0),
        ('<dBR/dq2>(B0->K*mumu)', 4.0, 6.0),

        # phi examples
        ('<FL>(Bs->phimumu)', 1.0, 6.0),
        ('<FL>(Bs->phimumu)', 15.0, 17.0),
        ('<dBR/dq2>(Bs->phimumu)', 15.0, 17.0),
    ]

    db = load_db()
    print("Loaded measurements.yml entries:", len(db))

    for key in targets:
        print("\n" + "=" * 100)
        print("Observable key:", key)
        mnames = measurement_names_for_key(key)
        print("N measurements:", len(mnames))
        for mn in mnames:
            print("\n-", mn)
            print_measurement_ref(db, mn)


if __name__ == "__main__":
    main()
