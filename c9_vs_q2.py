#!/usr/bin/env python3
import argparse
import math
import re
import time
import warnings
from collections import defaultdict
from importlib import import_module

import flavio
import yaml
from pathlib import Path


CHANNEL_PATTERNS = {
    "Kstar": [r"\bB0->K\*0?mumu\b", r"\bBd->K\*0?mumu\b"],
    "K":     [r"\bB\+->K\+?mumu\b"],
    "phi":   [r"\bBs->phimumu\b"],
}

# Curated list of publications to use per mode.
# Only these are included by default to avoid double-counting.
# Use --all-publications to override.
CURATED_PUBLICATIONS = {
    "Kstar": [
        "Aaij:2020nrf",       # LHCb Run 1+2 angular (FL, AFB, S/P-basis)
        "Aaij:2015oid",       # LHCb Run 1 (A7, A8, A9 — unique observables)
        "Aaij:2016flj",       # LHCb Run 1 BR
        "Aaboud:2018krd",     # ATLAS Run 2 angular
        "CMS:2024atz",        # CMS 13 TeV 140/fb angular (FL, P1–P8')
        "CMS:2017rzx",        # CMS 8 TeV angular (P1, P5' per-bin with correlations)
        "Khachatryan:2015isa", # CMS 8 TeV (FL, AFB, BR — different obs from 2017rzx)
        "CDF:2012qwd",        # CDF (FL, AFB, BR)
    ],
    "K": [
        "Aaij:2014pli",       # LHCb B+->Kmumu BR
        "CMS:2024aev",        # CMS 13 TeV 137/fb B+->Kmumu BR (R(K) paper)
    ],
    "phi": [
        "Aaij:2015esa",       # LHCb Run 1 Bs->phimumu angular + BR
        "Aaij:2021pkz",       # LHCb Run 1+2 Bs->phimumu BR
        "LHCb:2021xxq",       # LHCb Run 1+2 Bs->phimumu angular
        "CDF:2012qwd",        # CDF Bs->phimumu BR
    ],
}

ANGULAR_KEYWORDS = [
    "P1", "P2", "P3", "P4p", "P5p", "P6p", "P8p",
    "S3", "S4", "S5", "S7", "S8", "S9",
    "A_FB", "AFB", "F_L", "FL", "A9", "A7", "A8",
    "J_", "A_T", "AT", "Q_",
]
RATE_KEYWORDS = ["dBR", "BR", "dG", "Gamma", "d^2BR", "d^2Gamma"]


def _matches_any(name, patterns):
    return any(re.search(p, name) for p in patterns)


def _classify(name):
    for kw in ANGULAR_KEYWORDS:
        if kw in name:
            return "angular"
    for kw in RATE_KEYWORDS:
        if kw in name:
            return "rate"
    return "other"


def discover_base_names(mode, obs_kind):
    """Return list of observable base names for a given mode and obs_kind."""
    pats = CHANNEL_PATTERNS[mode]
    groups = defaultdict(list)
    for o in sorted(flavio.Observable.instances.keys()):
        if _matches_any(o, pats):
            groups[_classify(o)].append(o)
    names = []
    if obs_kind in ("angular", "both"):
        names.extend(groups["angular"])
    if obs_kind in ("rate", "both"):
        names.extend(groups["rate"])
    return names


def _load_measurements_db():
    p = Path(flavio.__file__).resolve().parent / "data" / "measurements.yml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def _pub_key(db, mname):
    meta = db.get(mname)
    if isinstance(meta, dict):
        return meta.get("inspire") or meta.get("arxiv") or meta.get("eprint") or mname
    return mname


def resolve_pub_to_measurement_names(pub_ids, db):
    """Return list of flavio measurement names that belong to any of the given publication IDs."""
    pub_set = set(pub_ids)
    names = []
    for mname in flavio.Measurement.instances:
        if _pub_key(db, mname) in pub_set:
            names.append(mname)
    return names


def find_publications_for_keys(keys_bin, db):
    """Return {pub_id: [list of obs keys]} for measurements that constrain any of keys_bin."""
    keys_set = set(keys_bin)
    pubs = defaultdict(set)
    for mname, m in flavio.Measurement.instances.items():
        try:
            overlap = keys_set & set(m.all_parameters)
        except Exception:
            continue
        if overlap:
            pid = _pub_key(db, mname)
            pubs[pid].update(overlap)
    return {pid: sorted(obs) for pid, obs in sorted(pubs.items())}


def _experiment_for_pub(db, pid):
    """Return experiment name for a publication id."""
    for mname, meta in db.items():
        if not isinstance(meta, dict):
            continue
        mpid = meta.get("inspire") or meta.get("arxiv") or meta.get("eprint")
        if mpid == pid:
            return meta.get("experiment", "?")
    return "?"


def check_overlap_warnings(keys_bin, db):
    """Detect same-experiment publications measuring the same observable in overlapping bins."""
    # Build: (experiment, obs_name, q2min, q2max) -> [pub_id, ...]
    keys_set = set(keys_bin)
    # pub_id -> experiment
    pub_exp = {}
    # pub_id -> set of keys
    pub_keys = defaultdict(set)
    for mname, m in flavio.Measurement.instances.items():
        try:
            overlap = keys_set & set(m.all_parameters)
        except Exception:
            continue
        if not overlap:
            continue
        pid = _pub_key(db, mname)
        if pid not in pub_exp:
            pub_exp[pid] = _experiment_for_pub(db, pid)
        pub_keys[pid].update(overlap)

    # Group pubs by experiment
    exp_pubs = defaultdict(list)
    for pid, exp in pub_exp.items():
        exp_pubs[exp].append(pid)

    warnings = []
    for exp, pids in exp_pubs.items():
        if len(pids) < 2:
            continue
        # For each pair of pubs from the same experiment, check for overlapping obs
        for i, p1 in enumerate(pids):
            for p2 in pids[i+1:]:
                shared = pub_keys[p1] & pub_keys[p2]
                if shared:
                    obs_names = sorted(set(
                        k[0] if isinstance(k, tuple) else k for k in shared
                    ))
                    warnings.append((exp, p1, p2, obs_names))
    return warnings


def print_bin_publications(q2lo, q2hi, keys_bin, db, curated_pubs=None):
    """Print which publications contribute to this bin."""
    all_pubs = find_publications_for_keys(keys_bin, db)
    obs_names = sorted(set(k[0] if isinstance(k, tuple) else k for k in keys_bin))
    print(f"  Observables: {', '.join(obs_names)}")

    if curated_pubs is not None:
        curated_set = set(curated_pubs)
        active = {p: o for p, o in all_pubs.items() if p in curated_set}
        skipped = {p: o for p, o in all_pubs.items() if p not in curated_set}
    else:
        active = all_pubs
        skipped = {}

    print(f"  Publications ({len(active)}):")
    for pid, obs in active.items():
        names = sorted(set(k[0] if isinstance(k, tuple) else k for k in obs))
        print(f"    {pid}: {', '.join(names)}")
    if skipped:
        print(f"  Skipped ({len(skipped)}):")
        for pid, obs in skipped.items():
            names = sorted(set(k[0] if isinstance(k, tuple) else k for k in obs))
            print(f"    {pid}: {', '.join(names)}")

    overlap_warnings = check_overlap_warnings(keys_bin, db)
    if overlap_warnings:
        print(f"  *** OVERLAP WARNINGS (all publications) ***")
        for exp, p1, p2, obs_names in overlap_warnings:
            print(f"    {exp}: {p1} & {p2} both measure {', '.join(obs_names)}")


def is_binned_key(k):
    return isinstance(k, tuple) and len(k) >= 3 and isinstance(k[0], str) and isinstance(k[1], (int, float)) and isinstance(k[2], (int, float))


def normalize_obs_key(k):
    if isinstance(k, str):
        return (k,)
    if isinstance(k, tuple) and len(k) >= 1 and isinstance(k[0], str):
        return k
    return None


def iter_measurement_constraint_keys(m):
    try:
        for k in m.all_parameters:
            nk = normalize_obs_key(k)
            if nk is not None:
                yield nk
    except Exception:
        return


def collect_constrained_obs_for_base_names(base_names, include_measurements=None):
    base_names = set(base_names)
    found = set()
    for m in flavio.Measurement.instances.values():
        if include_measurements is not None and m.name not in include_measurements:
            continue
        for obs in iter_measurement_constraint_keys(m):
            if obs and obs[0] in base_names:
                found.add(obs)
    return sorted(found)


def make_wc_delta_c9(c9_value):
    if hasattr(flavio, "WilsonCoefficients"):
        wc = flavio.WilsonCoefficients()
        wc.set_initial({"C9_bsmumu": float(c9_value)}, scale=4.8, eft="WET", basis="flavio")
        return wc
    from wilson import Wilson
    return Wilson({"C9_bsmumu": float(c9_value)}, scale=4.8, eft="WET", basis="flavio")


def grid_linspace(a, b, n):
    if n < 2:
        return [float(a)]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def argmax(xs):
    i_best = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[i_best]:
            i_best = i
    return i_best


def interp_x_at_y(x1, y1, x2, y2, ytarget):
    if y2 == y1:
        return 0.5 * (x1 + x2)
    t = (ytarget - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def estimate_best_and_sigma(c9_grid, logl):
    i0 = argmax(logl)
    c9_best = c9_grid[i0]
    ll_best = logl[i0]
    target = ll_best - 0.5

    left = None
    for i in range(i0, 0, -1):
        if (logl[i] >= target) and (logl[i - 1] < target):
            left = interp_x_at_y(c9_grid[i], logl[i], c9_grid[i - 1], logl[i - 1], target)
            break

    right = None
    for i in range(i0, len(c9_grid) - 1):
        if (logl[i] >= target) and (logl[i + 1] < target):
            right = interp_x_at_y(c9_grid[i], logl[i], c9_grid[i + 1], logl[i + 1], target)
            break

    sigma = float("nan")
    if left is not None and right is not None:
        sigma = 0.5 * (right - left)

    return c9_best, sigma, left, right, ll_best


def get_fastlikelihood_class():
    try:
        mod = import_module("flavio.statistics.likelihood")
        return getattr(mod, "FastLikelihood")
    except Exception:
        pass
    try:
        return getattr(flavio.statistics, "FastLikelihood")
    except Exception:
        pass
    raise RuntimeError("FastLikelihood not found in this flavio version.")


def build_fastlikelihood(name, observables, include_measurements=None,
                         threads=1, fast_N=100, fast_Nexp=5000):
    FastLikelihood = get_fastlikelihood_class()
    fl = FastLikelihood(
        name=name,
        par_obj=flavio.default_parameters,
        fit_parameters=[],
        nuisance_parameters="all",
        observables=observables,
        exclude_measurements=None,
        include_measurements=include_measurements,
    )

    # precompute
    if hasattr(fl, "make_measurement"):
        fl.make_measurement(N=int(fast_N), Nexp=int(fast_Nexp), threads=int(threads))
    elif hasattr(fl, "make_measurements"):
        fl.make_measurements(N=int(fast_N), Nexp=int(fast_Nexp), threads=int(threads))
    return fl


def eval_logl_fast(fl, wc):
    par = fl.parameters_central
    return float(fl.log_likelihood(par, wc))


def scan_c9_fast(fl, c9_grid, label="", report_every=10):
    t0 = time.time()
    out = []
    n = len(c9_grid)
    print(f"[{label}] scan start ({n} points)", flush=True)
    for i, c9 in enumerate(c9_grid, start=1):
        wc = make_wc_delta_c9(c9)
        out.append(eval_logl_fast(fl, wc))
        if report_every and (i % report_every == 0 or i == n):
            dt = time.time() - t0
            rate = i / dt if dt > 0 else float("inf")
            eta = (n - i) / rate if rate > 0 else float("inf")
            print(f"[{label}] {i}/{n} elapsed {dt:.1f}s eta {eta:.1f}s", flush=True)
    return out


def overlaps_any(q2min, q2max, veto_windows):
    for (a, b) in veto_windows:
        if (q2min < b) and (q2max > a):
            return True
    return False


def filter_to_contained_bin(keys, q2bin, veto_windows):
    """Select observable keys whose q2 range is contained within q2bin."""
    q2lo, q2hi = q2bin
    out = []
    for k in keys:
        if not is_binned_key(k):
            continue
        q2min = float(k[1])
        q2max = float(k[2])
        if q2min >= q2lo - 1e-9 and q2max <= q2hi + 1e-9:
            if veto_windows and overlaps_any(q2min, q2max, veto_windows):
                continue
            out.append(k)
    return out


def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.4g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fast-N", type=int, default=100)
    ap.add_argument("--fast-Nexp", type=int, default=5000)
    ap.add_argument("--c9min", type=float, default=-3.0)
    ap.add_argument("--c9max", type=float, default=+1.0)
    ap.add_argument("--npts", type=int, default=161)
    ap.add_argument("--report-every", type=int, default=10)

    ap.add_argument("--mode", choices=["Kstar", "K", "phi"], default="Kstar",
                    help="Which decay mode to use for the per-bin fit.")
    ap.add_argument("--obs-kind", choices=["angular", "rate", "both"], default="both",
                    help="Use angular observables, rate observables, or both.")

    ap.add_argument("--bins", default="0.1-0.98,1.1-2.5,2.5-4,4-6,15-17,17-19",
                    help="Comma-separated q2 bins, each as lo-hi in GeV^2.")
    ap.add_argument("--veto", default="8.68-14.18",
                    help="Comma-separated veto windows lo-hi. Use empty string to disable.")
    ap.add_argument("--all-publications", action="store_true",
                    help="Use all available measurements instead of the curated list.")
    ap.add_argument("--quiet-warnings", action="store_true")
    args = ap.parse_args()

    if args.quiet_warnings:
        warnings.filterwarnings("ignore", message=".*QCDF corrections should not be trusted.*")
        warnings.filterwarnings("ignore", message=".*predictions in the region of narrow charmonium resonances.*")

    base_names = discover_base_names(args.mode, args.obs_kind)

    meas_db = _load_measurements_db()

    if args.all_publications:
        curated_pubs = None
        include_meas = None
    else:
        curated_pubs = CURATED_PUBLICATIONS.get(args.mode, [])
        include_meas = resolve_pub_to_measurement_names(curated_pubs, meas_db)

    # Resolve to constrained keys (restricted to included measurements if curated)
    include_meas_set = set(include_meas) if include_meas is not None else None
    all_keys = collect_constrained_obs_for_base_names(base_names, include_meas_set)

    # Parse bins
    q2bins = []
    for part in args.bins.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split("-")
        q2bins.append((float(lo), float(hi)))

    veto_windows = []
    veto_str = args.veto.strip()
    if veto_str:
        for part in veto_str.split(","):
            part = part.strip()
            if not part:
                continue
            lo, hi = part.split("-")
            veto_windows.append((float(lo), float(hi)))

    c9_grid = grid_linspace(args.c9min, args.c9max, args.npts)

    print(f"\nMode={args.mode} obs_kind={args.obs_kind}", flush=True)
    if curated_pubs is not None:
        print(f"Publications: {', '.join(curated_pubs)}", flush=True)
    else:
        print(f"Publications: all (no filter)", flush=True)
    print(f"Bins={q2bins}", flush=True)
    print(f"Veto={veto_windows}", flush=True)
    print(f"Grid: [{args.c9min},{args.c9max}] npts={args.npts}", flush=True)

    results = []
    for (q2lo, q2hi) in q2bins:
        keys_bin = filter_to_contained_bin(all_keys, (q2lo, q2hi), veto_windows=veto_windows)
        if not keys_bin:
            print(f"\nBin {q2lo}-{q2hi}: no exact-edge constrained observables found, skipping.", flush=True)
            continue

        print(f"\nBin {q2lo}-{q2hi}: nobs={len(keys_bin)}", flush=True)
        print_bin_publications(q2lo, q2hi, keys_bin, meas_db, curated_pubs=curated_pubs)
        print(f"  Building FastLikelihood...", flush=True)
        t0 = time.time()
        fl = build_fastlikelihood(
            name=f"{args.mode}_{args.obs_kind}_{q2lo}_{q2hi}",
            observables=keys_bin,
            include_measurements=include_meas,
            threads=args.threads,
            fast_N=args.fast_N,
            fast_Nexp=args.fast_Nexp,
        )
        print(f"  Built in {time.time()-t0:.1f}s", flush=True)

        ll = scan_c9_fast(fl, c9_grid, label=f"{q2lo}-{q2hi}", report_every=args.report_every)
        c9_best, sig, left, right, _ = estimate_best_and_sigma(c9_grid, ll)

        results.append((q2lo, q2hi, len(keys_bin), c9_best, sig, left, right))

    print("\n" + "=" * 92)
    print("Delta C9_bsmumu per q2 bin (piecewise effective fit)")
    print("=" * 92)
    print("{:>10} {:>10} {:>6} {:>12} {:>10} {:>12} {:>12}".format("q2min", "q2max", "Nobs", "C9_best", "sigma", "68%_lo", "68%_hi"))
    for (q2lo, q2hi, nobs, c9_best, sig, left, right) in results:
        print("{:>10.2f} {:>10.2f} {:>6d} {:>12} {:>10} {:>12} {:>12}".format(
            q2lo, q2hi, nobs, fmt(c9_best), fmt(sig), fmt(left), fmt(right)
        ))

    print("\nNotes:")
    print("- This is an effective Delta C9 per bin (absorbs nonlocal charm).")
    print("- It uses all observables whose q2 bin is contained within the requested bin.")
    print("- Veto removes any bin overlapping the specified windows.")
    print("- For speed while iterating: try --fast-N 50 --fast-Nexp 1000 and fewer --npts.")


if __name__ == "__main__":
    main()
