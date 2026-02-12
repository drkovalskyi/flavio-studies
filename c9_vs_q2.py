#!/usr/bin/env python3
import argparse
import math
import re
import time
import warnings
from collections import defaultdict, OrderedDict
from importlib import import_module

import flavio
import yaml
from pathlib import Path
from scipy.stats import chi2 as scipy_chi2


def _json_safe(val):
    """Convert None / nan / inf to JSON-safe values."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


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


def classify_obs_key(key):
    """Return (channel, obs_type) for an observable key."""
    name = key[0] if isinstance(key, tuple) else key
    channel = "other"
    for ch, pats in CHANNEL_PATTERNS.items():
        if _matches_any(name, pats):
            channel = ch
            break
    obs_type = _classify(name)
    return channel, obs_type


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


def _filter_measurements_for_obs(obs_keys, include_meas):
    """Filter measurement names to only those constraining at least one of obs_keys.

    This prevents FastLikelihood from failing when a measurement in include_meas
    doesn't constrain any of the observables in the sub-group.
    """
    obs_set = set(obs_keys)
    filtered = []
    for mname in include_meas:
        try:
            m = flavio.Measurement[mname]
        except (KeyError, Exception):
            continue
        try:
            params = set(m.all_parameters)
        except Exception:
            continue
        if obs_set & params:
            filtered.append(mname)
    return filtered if filtered else None


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


def check_overlap_warnings(keys_bin, db, include_meas=None):
    """Detect same-experiment publications measuring the same observable in overlapping bins."""
    # Build: (experiment, obs_name, q2min, q2max) -> [pub_id, ...]
    keys_set = set(keys_bin)
    # pub_id -> experiment
    pub_exp = {}
    # pub_id -> set of keys
    pub_keys = defaultdict(set)
    for mname, m in flavio.Measurement.instances.items():
        if include_meas is not None and mname not in include_meas:
            continue
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


def print_bin_publications(q2lo, q2hi, keys_bin, db, curated_pubs=None, include_meas=None):
    """Print which publications contribute to this bin."""
    all_pubs = find_publications_for_keys(keys_bin, db)
    obs_names = sorted(set(k[0] if isinstance(k, tuple) else k for k in keys_bin))
    print(f"  Observables: {_compact_obs_names(obs_names)}")

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
        print(f"    {pid}: {_compact_obs_names(names)}")
    if skipped:
        print(f"  Skipped ({len(skipped)}):")
        for pid, obs in skipped.items():
            names = sorted(set(k[0] if isinstance(k, tuple) else k for k in obs))
            print(f"    {pid}: {_compact_obs_names(names)}")

    overlap_warnings = check_overlap_warnings(keys_bin, db, include_meas=include_meas)
    if overlap_warnings:
        print(f"  *** POSSIBLE OVERLAP WARNINGS ***")
        for exp, p1, p2, obs_names in overlap_warnings:
            print(f"    {exp}: {p1} & {p2} both measure {_compact_obs_names(obs_names)}")


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


def _parabolic_sigma(c9_grid, logl, i0):
    """Estimate sigma from a parabolic fit to 3 points around the maximum.

    Returns sigma or nan if curvature is non-negative (flat/rising edges).
    """
    if i0 <= 0 or i0 >= len(c9_grid) - 1:
        return float("nan")
    x0, x1, x2 = c9_grid[i0 - 1], c9_grid[i0], c9_grid[i0 + 1]
    y0, y1, y2 = logl[i0 - 1], logl[i0], logl[i0 + 1]
    # second derivative of parabola through 3 equally-spaced points
    h = x1 - x0
    if h == 0:
        return float("nan")
    d2 = (y2 - 2 * y1 + y0) / (h * h)
    if d2 >= 0:
        return float("nan")
    return 1.0 / math.sqrt(-d2)


def estimate_best_and_sigma(c9_grid, logl):
    i0 = argmax(logl)
    c9_best = c9_grid[i0]
    ll_best = logl[i0]
    at_boundary = (i0 == 0) or (i0 == len(c9_grid) - 1)
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

    # Parabolic fallback when crossing method fails
    if math.isnan(sigma):
        sigma = _parabolic_sigma(c9_grid, logl, i0)

    return c9_best, sigma, left, right, ll_best, at_boundary


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


def _eval_logl_worker(args):
    fl, c9 = args
    wc = make_wc_delta_c9(c9)
    par = fl.parameters_central
    return float(fl.log_likelihood(par, wc))


def scan_c9_fast(fl, c9_grid, label="", report_every=10, threads=1):
    t0 = time.time()
    n = len(c9_grid)
    print(f"[{label}] scan start ({n} points, threads={threads})", flush=True)

    if threads > 1:
        from multiprocessing import Pool
        with Pool(processes=threads) as pool:
            work = [(fl, c9) for c9 in c9_grid]
            out = []
            for i, val in enumerate(pool.imap(_eval_logl_worker, work), start=1):
                out.append(val)
                if report_every and (i % report_every == 0 or i == n):
                    dt = time.time() - t0
                    rate = i / dt if dt > 0 else float("inf")
                    eta = (n - i) / rate if rate > 0 else float("inf")
                    print(f"[{label}] {i}/{n} elapsed {dt:.1f}s eta {eta:.1f}s", flush=True)
    else:
        out = []
        for i, c9 in enumerate(c9_grid, start=1):
            out.append(_eval_logl_worker((fl, c9)))
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
    """Select observable keys whose q2 range is contained within q2bin.

    Returns (contained, excluded_overlap) where excluded_overlap are keys
    that overlap the scan bin but are not fully contained.
    """
    q2lo, q2hi = q2bin
    contained = []
    excluded_overlap = []
    for k in keys:
        if not is_binned_key(k):
            continue
        q2min = float(k[1])
        q2max = float(k[2])
        if q2min >= q2lo - 1e-9 and q2max <= q2hi + 1e-9:
            if veto_windows and overlaps_any(q2min, q2max, veto_windows):
                continue
            contained.append(k)
        elif q2min < q2hi and q2max > q2lo:
            # Overlaps but not contained
            if veto_windows and overlaps_any(q2min, q2max, veto_windows):
                continue
            excluded_overlap.append(k)
    return contained, excluded_overlap


##############################################################################
# Two-phase global bin selection
##############################################################################

def compute_active_intervals(q2bins, veto_windows):
    """Return sorted non-overlapping intervals = union(scan bins) minus veto."""
    # 1. Merge overlapping scan bins
    intervals = sorted(q2bins)
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1] + 1e-12:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    # 2. Subtract veto windows
    for vlo, vhi in veto_windows:
        new = []
        for lo, hi in merged:
            if vhi <= lo or vlo >= hi:
                new.append((lo, hi))
            else:
                if lo < vlo:
                    new.append((lo, vlo))
                if vhi < hi:
                    new.append((vhi, hi))
        merged = new
    return merged


def overlap_with_active(q2min, q2max, active_intervals):
    """Return total overlap length between [q2min, q2max] and active intervals."""
    total = 0.0
    for lo, hi in active_intervals:
        ov_lo = max(q2min, lo)
        ov_hi = min(q2max, hi)
        if ov_hi > ov_lo:
            total += ov_hi - ov_lo
    return total


def group_keys_by_paper_obs(all_keys, include_meas, meas_db):
    """Return dict[(paper_id, obs_name)] -> list[key] for binned observable keys.

    Groups keys by their publication and base observable name,
    considering only measurements in include_meas.
    """
    # Build key -> set of paper_ids
    key_papers = defaultdict(set)
    for mname, m in flavio.Measurement.instances.items():
        if include_meas is not None and mname not in include_meas:
            continue
        pid = _pub_key(meas_db, mname)
        try:
            params = set(m.all_parameters)
        except Exception:
            continue
        for k in all_keys:
            if k in params:
                key_papers[k].add(pid)

    groups = defaultdict(list)
    for k in all_keys:
        if not is_binned_key(k):
            continue
        obs_name = k[0]
        for pid in key_papers.get(k, set()):
            groups[(pid, obs_name)].append(k)
    return groups


def select_non_overlapping_max_coverage(bins, active_intervals):
    """Weighted interval scheduling DP to select non-overlapping bins
    maximizing total coverage of active range.

    bins: list of (q2min, q2max) tuples
    Returns: list of selected (q2min, q2max) tuples
    """
    if not bins:
        return []

    # Sort by right endpoint
    indexed = sorted(range(len(bins)), key=lambda i: bins[i][1])
    sorted_bins = [bins[i] for i in indexed]
    n = len(sorted_bins)

    # Compute weights
    eps = 1e-9  # tiebreaker: prefer more (finer) bins
    weights = []
    for q2min, q2max in sorted_bins:
        w = overlap_with_active(q2min, q2max, active_intervals)
        weights.append(w + eps)

    # p[i] = index of last interval that doesn't overlap with i, or -1
    import bisect
    right_ends = [b[1] for b in sorted_bins]
    p = []
    for i in range(n):
        lo = sorted_bins[i][0]
        # Find rightmost interval ending <= lo
        j = bisect.bisect_right(right_ends, lo + 1e-12) - 1
        if j < 0 or j >= i:
            # Ensure we don't point at i or beyond
            j2 = -1
            for jj in range(i - 1, -1, -1):
                if sorted_bins[jj][1] <= lo + 1e-12:
                    j2 = jj
                    break
            p.append(j2)
        else:
            p.append(j)

    # DP
    dp = [0.0] * n
    dp[0] = weights[0]
    for i in range(1, n):
        skip = dp[i - 1]
        take = weights[i] + (dp[p[i]] if p[i] >= 0 else 0.0)
        dp[i] = max(skip, take)

    # Backtrack
    selected = []
    i = n - 1
    while i >= 0:
        take = weights[i] + (dp[p[i]] if p[i] >= 0 else 0.0)
        skip = dp[i - 1] if i > 0 else 0.0
        if take >= skip:
            selected.append(sorted_bins[i])
            i = p[i]
        else:
            i -= 1

    selected.reverse()
    return selected


def select_bins_global(all_keys, include_meas, meas_db, q2bins, veto_windows):
    """Phase 1: globally select non-overlapping measurement bins per (paper, obs).

    Returns:
        selected_keys: set of keys that passed global selection
        report: list of (paper_id, obs_name, included_bins, excluded_bins) for printing
    """
    active = compute_active_intervals(q2bins, veto_windows)
    groups = group_keys_by_paper_obs(all_keys, include_meas, meas_db)

    selected_keys = set()
    report = []

    for (pid, obs_name), keys in sorted(groups.items()):
        # Collect measurement bins for this group
        bin_to_keys = defaultdict(list)
        for k in keys:
            q2min, q2max = float(k[1]), float(k[2])
            bin_to_keys[(q2min, q2max)].append(k)

        # Eligibility filter: >=50% of measurement bin width overlaps active range,
        # AND the bin can actually be assigned to at least one scan bin (>=50%
        # of its width falls within some scan bin).
        eligible_bins = []
        for (q2min, q2max), ks in bin_to_keys.items():
            width = q2max - q2min
            if width <= 0:
                continue
            ov = overlap_with_active(q2min, q2max, active)
            if ov / width < 0.5:
                continue
            # Check assignability: must fit in at least one scan bin
            assignable = False
            for sq2lo, sq2hi in q2bins:
                sov = min(sq2hi, q2max) - max(sq2lo, q2min)
                if sov > 0 and sov / width >= 0.5:
                    assignable = True
                    break
            if assignable:
                eligible_bins.append((q2min, q2max))

        # Run DP selection
        chosen = select_non_overlapping_max_coverage(eligible_bins, active)
        chosen_set = set(chosen)

        # Classify keys as selected or excluded
        included_bins = sorted(chosen_set)
        excluded_bins = sorted(set(bin_to_keys.keys()) - chosen_set)

        for b in included_bins:
            for k in bin_to_keys[b]:
                selected_keys.add(k)

        report.append((pid, obs_name, included_bins, excluded_bins))

    return selected_keys, report


def assign_keys_to_scan_bin(selected_keys, q2bin, veto_windows, min_containment=0.5):
    """Phase 2: assign globally-selected keys to a scan bin.

    A key is included if >=min_containment of its measurement bin width
    overlaps the scan bin, and it doesn't overlap a veto window.

    Returns list of assigned keys.
    """
    q2lo, q2hi = q2bin
    assigned = []
    for k in selected_keys:
        if not is_binned_key(k):
            continue
        q2min, q2max = float(k[1]), float(k[2])
        width = q2max - q2min
        if width <= 0:
            continue
        if veto_windows and overlaps_any(q2min, q2max, veto_windows):
            continue
        ov = min(q2hi, q2max) - max(q2lo, q2min)
        if ov <= 0:
            continue
        if ov / width >= min_containment:
            assigned.append(k)
    return sorted(assigned)


def _compact_obs_names(obs_names):
    """Format observable names compactly by factoring out the common decay channel.

    E.g. ['<FL>(B0->K*mumu)', '<P1>(B0->K*mumu)'] -> 'FL, P1 (B0->K*mumu)'
    """
    # Parse each name into (short, channel) where possible
    parsed = OrderedDict()  # channel -> [short_name, ...]
    for name in obs_names:
        # Match pattern like <obs>(decay)
        m = re.match(r'^<(.+?)>\((.+)\)$', name)
        if m:
            parsed.setdefault(m.group(2), []).append(m.group(1))
        else:
            parsed.setdefault('', []).append(name)
    parts = []
    for channel, shorts in parsed.items():
        if channel:
            parts.append(f"{', '.join(shorts)} ({channel})")
        else:
            parts.append(", ".join(shorts))
    return "; ".join(parts)


def print_selection_report(report):
    """Print the global bin selection summary, grouping observables with identical bins."""
    # Group by paper, then by (included, excluded) bin signature
    paper_groups = OrderedDict()  # pid -> {(inc_tuple, exc_tuple): [obs_name, ...]}
    for pid, obs_name, included, excluded in report:
        sig = (tuple(included), tuple(excluded))
        paper_groups.setdefault(pid, OrderedDict()).setdefault(sig, []).append(obs_name)

    n_papers = len(paper_groups)
    print(f"\nGlobal bin selection ({n_papers} papers):", flush=True)
    for pid, sig_map in paper_groups.items():
        for (inc, exc), obs_names in sig_map.items():
            obs_str = _compact_obs_names(obs_names)
            inc_str = " ".join(f"[{lo},{hi}]" for lo, hi in inc) if inc else "(none)"
            print(f"  {pid} / {obs_str}:", flush=True)
            print(f"    included: {inc_str}", flush=True)
            if exc:
                exc_str = " ".join(f"[{lo},{hi}]" for lo, hi in exc)
                print(f"    excluded: {exc_str}", flush=True)


def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.4g}"


def chi2_from_pulls(pull_list):
    """Compute chi2/ndf and p-value from a list of pull tuples.

    Accepts (label, pull) or (key, label, pull) tuples.
    chi2 = sum(pull_i^2).  This ignores off-diagonal correlations between
    observables but is always non-negative and easy to interpret.
    ndf = n_pulls - 1  (one fitted parameter: C9).
    """
    valid = [t[-1] for t in pull_list if not math.isnan(t[-1])]
    if not valid:
        return float("nan"), 0, float("nan")
    chi2_val = sum(p * p for p in valid)
    ndf = len(valid) - 1
    p_value = float(scipy_chi2.sf(chi2_val, ndf)) if ndf > 0 else float("nan")
    return chi2_val, ndf, p_value


def consistency_by_channel_type(keys_bin, include_meas, c9_grid_coarse, args):
    """Fit C9 separately for each (channel, obs_type) sub-group.

    Returns list of (label, c9_best, sigma, n_obs).
    """
    groups = defaultdict(list)
    for k in keys_bin:
        ch, ot = classify_obs_key(k)
        groups[(ch, ot)].append(k)

    results = []
    for (ch, ot), keys in sorted(groups.items()):
        label = f"{ch} {ot}"
        n_obs = len(keys)
        try:
            sub_meas = _filter_measurements_for_obs(keys, include_meas) if include_meas else None
            fl_sub = build_fastlikelihood(
                name=f"consist_{ch}_{ot}",
                observables=keys,
                include_measurements=sub_meas,
                threads=args.threads,
                fast_N=args.fast_N,
                fast_Nexp=args.fast_Nexp,
            )
            ll = scan_c9_fast(fl_sub, c9_grid_coarse, label=f"  {label}", report_every=0, threads=args.threads)
            c9_best, sigma, _, _, _, at_bnd = estimate_best_and_sigma(c9_grid_coarse, ll)
        except Exception as e:
            print(f"    [warn] sub-scan {label} failed: {e}", flush=True)
            c9_best, sigma, at_bnd = float("nan"), float("nan"), False
        results.append((label, c9_best, sigma, n_obs, at_bnd))
    return results


def consistency_by_publication(keys_bin, include_meas, c9_grid_coarse, meas_db, curated_pubs, args):
    """Fit C9 separately for each publication contributing to the bin.

    Returns list of (pub_id, c9_best, sigma, n_obs).
    """
    all_pubs = find_publications_for_keys(keys_bin, meas_db)
    if curated_pubs is not None:
        curated_set = set(curated_pubs)
        all_pubs = {p: obs for p, obs in all_pubs.items() if p in curated_set}

    results = []
    for pub_id, obs_keys in sorted(all_pubs.items()):
        n_obs = len(obs_keys)
        try:
            pub_meas = resolve_pub_to_measurement_names([pub_id], meas_db)
            if not pub_meas:
                raise ValueError(f"no measurements for {pub_id}")
            sub_meas = _filter_measurements_for_obs(obs_keys, pub_meas)
            if not sub_meas:
                raise ValueError(f"no constraining measurements for {pub_id}")
            fl_sub = build_fastlikelihood(
                name=f"consist_pub_{pub_id}",
                observables=obs_keys,
                include_measurements=sub_meas,
                threads=args.threads,
                fast_N=args.fast_N,
                fast_Nexp=args.fast_Nexp,
            )
            ll = scan_c9_fast(fl_sub, c9_grid_coarse, label=f"  {pub_id}", report_every=0, threads=args.threads)
            c9_best, sigma, _, _, _, at_bnd = estimate_best_and_sigma(c9_grid_coarse, ll)
        except Exception as e:
            print(f"    [warn] sub-scan {pub_id} failed: {e}", flush=True)
            c9_best, sigma, at_bnd = float("nan"), float("nan"), False
        results.append((pub_id, c9_best, sigma, n_obs, at_bnd))
    return results


def consistency_pulls(fl, c9_best, keys_bin):
    """Compute pull = (measured - predicted) / sigma for each observable at best-fit C9.

    Returns list of (key, obs_label, pull) sorted by |pull| descending.
    """
    wc = make_wc_delta_c9(c9_best)
    par = fl.parameters_central

    # Get measured central values
    centrals = fl.pseudo_measurement.get_central_all()

    # Get total uncertainties via random sampling
    try:
        errors_1d = fl.pseudo_measurement.get_1d_errors_random(N=500)
    except Exception:
        errors_1d = fl.pseudo_measurement.get_1d_errors_random()

    pulls = []
    for key in fl.observables:
        if isinstance(key, tuple):
            obs_name = key[0]
            obs_args = {"q2min": key[1], "q2max": key[2]} if len(key) >= 3 else {}
        else:
            obs_name = key
            obs_args = {}

        try:
            pred = flavio.Observable[obs_name].prediction_par(par, wc, **obs_args)
        except Exception:
            continue

        meas = centrals.get(key, float("nan"))
        sigma = errors_1d.get(key, float("nan"))

        if sigma > 0:
            pull = (meas - pred) / sigma
        else:
            pull = float("nan")

        if isinstance(key, tuple) and len(key) >= 3:
            label = f"{obs_name} [{key[1]},{key[2]}]"
        else:
            label = obs_name

        pulls.append((key, label, pull))

    pulls.sort(key=lambda x: abs(x[2]) if not math.isnan(x[2]) else 0, reverse=True)
    return pulls


def sigma_clip_observables(keys_bin, include_meas, c9_grid, args, clip_sigma=3.0):
    """Iteratively remove observables with |pull| > clip_sigma, refitting each round.

    Returns (clipped_keys, removed_list, c9_best, sigma, left, right, fl)
    where removed_list is [(label, pull, iteration), ...].
    """
    current_keys = list(keys_bin)
    removed = []
    iteration = 0
    fl = None
    c9_best = sigma = left = right = float("nan")

    for iteration in range(1, 11):  # max 10 iterations
        sub_meas = _filter_measurements_for_obs(current_keys, include_meas) if include_meas else None
        fl = build_fastlikelihood(
            name=f"clip_iter{iteration}",
            observables=current_keys,
            include_measurements=sub_meas,
            threads=args.threads,
            fast_N=args.fast_N,
            fast_Nexp=args.fast_Nexp,
        )
        ll = scan_c9_fast(fl, c9_grid, label=f"  clip iter {iteration}",
                          report_every=0, threads=args.threads)
        c9_best, sigma, left, right, _, _ = estimate_best_and_sigma(c9_grid, ll)

        pull_list = consistency_pulls(fl, c9_best, current_keys)
        outliers = [(key, label, pull) for key, label, pull in pull_list
                    if not math.isnan(pull) and abs(pull) > clip_sigma]

        if not outliers:
            break

        outlier_keys = set()
        for key, label, pull in outliers:
            removed.append((label, pull, iteration))
            outlier_keys.add(key)

        current_keys = [k for k in current_keys if k not in outlier_keys]
        if not current_keys:
            print(f"    [warn] sigma clipping removed all observables!", flush=True)
            break

    return current_keys, removed, c9_best, sigma, left, right, fl


def _fmt_c9_sig(c9, sig, at_boundary, grid_lo, grid_hi):
    """Format C9 ± sigma, marking boundary hits with < or >."""
    if math.isnan(c9):
        return "C9 = nan"
    if at_boundary:
        if c9 <= grid_lo:
            c9_str = f"C9 < {grid_lo:.2f}"
        else:
            c9_str = f"C9 > {grid_hi:.2f}"
        if not math.isnan(sig):
            return f"{c9_str}  (~{sig:.2f})"
        return c9_str
    return f"C9 = {c9:>6.2f} \u00b1 {sig:.2f}"


def print_consistency_report(fl, c9_best, sigma, keys_bin, include_meas,
                             c9_grid, c9_grid_coarse, meas_db, curated_pubs, args):
    """Print a full consistency report for the current bin.

    Returns (clipped_keys, cl_c9, cl_sig, cl_left, cl_right) when sigma-clipping
    removed outliers, or None if nothing was clipped.
    """
    grid_lo, grid_hi = c9_grid_coarse[0], c9_grid_coarse[-1]
    print(f"\n  Consistency report:", flush=True)

    # 1. Pulls (computed first so chi2 can be derived)
    pull_list = consistency_pulls(fl, c9_best, keys_bin)

    # 2. Combined chi2/ndf from pulls
    chi2_val, ndf, p_value = chi2_from_pulls(pull_list)
    print(f"    Combined: C9 = {c9_best:.2f} \u00b1 {sigma:.2f}, "
          f"chi2/ndf = {chi2_val:.1f}/{ndf}, p = {p_value:.2f}", flush=True)

    # 3. By channel x type
    print(f"\n    By channel x type:", flush=True)
    ch_results = consistency_by_channel_type(keys_bin, include_meas, c9_grid_coarse, args)
    for label, c9, sig, nobs, at_bnd in ch_results:
        c9s = _fmt_c9_sig(c9, sig, at_bnd, grid_lo, grid_hi)
        print(f"      {label:<20s} {c9s}  ({nobs} obs)", flush=True)

    # 4. By publication
    print(f"\n    By publication:", flush=True)
    pub_results = consistency_by_publication(keys_bin, include_meas, c9_grid_coarse, meas_db, curated_pubs, args)
    for pub_id, c9, sig, nobs, at_bnd in pub_results:
        c9s = _fmt_c9_sig(c9, sig, at_bnd, grid_lo, grid_hi)
        print(f"      {pub_id:<20s} {c9s}  ({nobs} obs)", flush=True)

    # 5. Pull table
    print(f"\n    Pulls at C9 = {c9_best:.2f}:", flush=True)
    for _, label, pull in pull_list:
        sign = "+" if pull >= 0 else ""
        print(f"      {label:<45s} {sign}{pull:.1f}\u03c3", flush=True)

    # 6. Sigma-clipping
    clip_sigma = args.clip_sigma
    print(f"\n    Sigma-clipped (|pull| > {clip_sigma:.1f}\u03c3):", flush=True)
    clipped_keys, removed, cl_c9, cl_sig, cl_left, cl_right, cl_fl = \
        sigma_clip_observables(keys_bin, include_meas, c9_grid, args, clip_sigma=clip_sigma)

    if removed:
        for label, pull, iteration in removed:
            sign = "+" if pull >= 0 else ""
            print(f"      iter {iteration}: removed {label} (pull = {sign}{pull:.1f}\u03c3)", flush=True)
        cl_pulls = consistency_pulls(cl_fl, cl_c9, clipped_keys)
        cl_chi2, cl_ndf, cl_pval = chi2_from_pulls(cl_pulls)
        n_total = len(keys_bin)
        n_removed = len(removed)
        print(f"      After clipping: C9 = {cl_c9:.2f} \u00b1 {cl_sig:.2f}, "
              f"chi2/ndf = {cl_chi2:.1f}/{cl_ndf}, p = {cl_pval:.2f} "
              f"(removed {n_removed} of {n_total} obs)", flush=True)
    else:
        print(f"      no outliers above {clip_sigma:.1f}\u03c3", flush=True)

    print(flush=True)

    if removed:
        return clipped_keys, cl_c9, cl_sig, cl_left, cl_right, cl_chi2, cl_ndf, cl_pval
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fast-N", type=int, default=100)
    ap.add_argument("--fast-Nexp", type=int, default=5000)
    ap.add_argument("--c9min", type=float, default=-3.0)
    ap.add_argument("--c9max", type=float, default=+1.0)
    ap.add_argument("--npts", type=int, default=161)
    ap.add_argument("--report-every", type=int, default=10)

    ap.add_argument("--mode", choices=["Kstar", "K", "phi", "all"], default="Kstar",
                    help="Which decay mode to use for the per-bin fit.")
    ap.add_argument("--obs-kind", choices=["angular", "rate", "both"], default="both",
                    help="Use angular observables, rate observables, or both.")

    ap.add_argument("--bins", default="0.1-0.98,1.1-2.5,2.5-4,4-6,15-17,17-19,19-22",
                    help="Comma-separated q2 bins, each as lo-hi in GeV^2.")
    ap.add_argument("--veto", default="8.68-14.18",
                    help="Comma-separated veto windows lo-hi. Use empty string to disable.")
    ap.add_argument("--all-publications", action="store_true",
                    help="Use all available measurements instead of the curated list.")
    ap.add_argument("--quiet-warnings", action="store_true")
    ap.add_argument("--consistency", action="store_true",
                    help="Print consistency report (chi2, sub-fits by channel/pub, pulls).")
    ap.add_argument("--clip-sigma", type=float, default=3.0,
                    help="Sigma threshold for iterative outlier clipping in consistency report.")
    ap.add_argument("--output-json", type=str, default=None,
                    help="Save results to a JSON file.")
    args = ap.parse_args()

    if args.quiet_warnings:
        warnings.filterwarnings("ignore", message=".*QCDF corrections should not be trusted.*")
        warnings.filterwarnings("ignore", message=".*predictions in the region of narrow charmonium resonances.*")

    if args.mode == "all":
        modes = list(CHANNEL_PATTERNS.keys())
    else:
        modes = [args.mode]

    base_names = []
    for m in modes:
        base_names.extend(discover_base_names(m, args.obs_kind))

    meas_db = _load_measurements_db()

    if args.all_publications:
        curated_pubs = None
        include_meas = None
    else:
        curated_pubs = []
        seen = set()
        for m in modes:
            for p in CURATED_PUBLICATIONS.get(m, []):
                if p not in seen:
                    curated_pubs.append(p)
                    seen.add(p)
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
    c9_grid_coarse = grid_linspace(args.c9min, args.c9max, 41) if args.consistency else None

    print(f"\nMode={args.mode} obs_kind={args.obs_kind}", flush=True)
    if curated_pubs is not None:
        print(f"Publications: {', '.join(curated_pubs)}", flush=True)
    else:
        print(f"Publications: all (no filter)", flush=True)
    print(f"Bins={q2bins}", flush=True)
    print(f"Veto={veto_windows}", flush=True)
    print(f"Grid: [{args.c9min},{args.c9max}] npts={args.npts}", flush=True)

    # Phase 1: global bin selection (used when curated publications are active)
    use_global_selection = not args.all_publications
    if use_global_selection:
        include_meas_set_for_sel = set(include_meas) if include_meas is not None else None
        selected_keys, selection_report = select_bins_global(
            all_keys, include_meas_set_for_sel, meas_db, q2bins, veto_windows)
        print_selection_report(selection_report)

    results = []
    clipped_results = []
    any_clipped = False
    for (q2lo, q2hi) in q2bins:
        if use_global_selection:
            # Phase 2: assign globally-selected keys to this scan bin
            keys_bin = assign_keys_to_scan_bin(selected_keys, (q2lo, q2hi), veto_windows)
        else:
            keys_bin, _ = filter_to_contained_bin(all_keys, (q2lo, q2hi), veto_windows=veto_windows)

        if not keys_bin:
            print(f"\nBin {q2lo}-{q2hi}: no constrained observables found, skipping.", flush=True)
            continue

        print(f"\nBin {q2lo}-{q2hi}: nobs={len(keys_bin)}", flush=True)
        print_bin_publications(q2lo, q2hi, keys_bin, meas_db, curated_pubs=curated_pubs, include_meas=include_meas)
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

        ll = scan_c9_fast(fl, c9_grid, label=f"{q2lo}-{q2hi}", report_every=args.report_every, threads=args.threads)
        c9_best, sig, left, right, _, _ = estimate_best_and_sigma(c9_grid, ll)

        pull_list = consistency_pulls(fl, c9_best, keys_bin)
        chi2_val, ndf, p_value = chi2_from_pulls(pull_list)

        results.append((q2lo, q2hi, len(keys_bin), c9_best, sig, left, right, chi2_val, ndf, p_value))

        if args.consistency:
            clip_result = print_consistency_report(
                fl, c9_best, sig, keys_bin, include_meas,
                c9_grid, c9_grid_coarse, meas_db, curated_pubs, args,
            )
            if clip_result is not None:
                cl_keys, cl_c9, cl_sig, cl_left, cl_right, cl_chi2, cl_ndf, cl_pval = clip_result
                clipped_results.append((q2lo, q2hi, len(cl_keys), cl_c9, cl_sig, cl_left, cl_right, cl_chi2, cl_ndf, cl_pval))
                any_clipped = True
            else:
                # No outliers — carry nominal result into clipped table
                clipped_results.append((q2lo, q2hi, len(keys_bin), c9_best, sig, left, right, chi2_val, ndf, p_value))

    hdr = "{:>10} {:>10} {:>6} {:>12} {:>10} {:>12} {:>12} {:>10} {:>6}".format(
        "q2min", "q2max", "Nobs", "C9_best", "sigma", "68%_lo", "68%_hi", "chi2/ndf", "p")

    def _fmt_row(q2lo, q2hi, nobs, c9_best, sig, left, right, chi2_val, ndf, p_value):
        left_s = f"{left:>12.2f}" if left is not None else f"{'n/a':>12}"
        right_s = f"{right:>12.2f}" if right is not None else f"{'n/a':>12}"
        chi2_s = f"{chi2_val:.1f}/{ndf}" if not math.isnan(chi2_val) else "n/a"
        p_s = f"{p_value:.2f}" if not math.isnan(p_value) else "n/a"
        return (f"{q2lo:>10.2f} {q2hi:>10.2f} {nobs:>6d} {c9_best:>12.2f} {sig:>10.2f}"
                f" {left_s} {right_s} {chi2_s:>10} {p_s:>6}")

    print("\n" + "=" * 110)
    print("Delta C9_bsmumu per q2 bin (piecewise effective fit)")
    print("=" * 110)
    print(hdr)
    for row in results:
        print(_fmt_row(*row))

    if any_clipped:
        print("\n" + "=" * 110)
        print(f"Delta C9_bsmumu per q2 bin — after sigma-clipping (|pull| > {args.clip_sigma:.1f}sigma)")
        print("=" * 110)
        print(hdr)
        for row in clipped_results:
            print(_fmt_row(*row))

    print("\nNotes:")
    print("- This is an effective Delta C9 per bin (absorbs nonlocal charm).")
    if use_global_selection:
        print("- Two-phase global bin selection: non-overlapping bins chosen per (paper, obs),")
        print("  then assigned to scan bins by >=50% width overlap.")
    else:
        print("- Using all observables whose q2 bin is contained within the requested bin.")
    print("- Veto removes any bin overlapping the specified windows.")
    print("- For speed while iterating: try --fast-N 50 --fast-Nexp 1000 and fewer --npts.")

    if args.output_json:
        import json

        def _row_dict(row):
            q2lo, q2hi, nobs, c9, sig, left, right, chi2v, ndf, pval = row
            return {
                "q2min": q2lo, "q2max": q2hi, "nobs": nobs,
                "c9_best": _json_safe(c9), "sigma": _json_safe(sig),
                "cl_lo": _json_safe(left), "cl_hi": _json_safe(right),
                "chi2": _json_safe(chi2v), "ndf": ndf, "p_value": _json_safe(pval),
            }

        payload = {
            "metadata": {
                "mode": args.mode,
                "obs_kind": args.obs_kind,
                "c9min": args.c9min, "c9max": args.c9max, "npts": args.npts,
                "fast_N": args.fast_N, "fast_Nexp": args.fast_Nexp,
                "clip_sigma": args.clip_sigma if args.consistency else None,
            },
            "results": [_row_dict(r) for r in results],
            "clipped_results": [_row_dict(r) for r in clipped_results] if any_clipped else None,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
