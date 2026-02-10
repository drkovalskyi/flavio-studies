#!/usr/bin/env python3
"""
impact_c9_blocks.py

- Loads grouped observable base names from out.json (produced by the listing script)
- Resolves them to constrained observable keys in the local flavio measurement DB, including binned tuples
- Builds a likelihood per block and does a 1D scan in Delta C9_bsmumu
- Computes leave-one-out impact numbers
- Shows progress with elapsed time and ETA
- Supports multi-core as a command line option

Multi-core behavior:
- If FastLikelihood is used: uses threads in make_measurement/make_measurements (if supported)
- If Likelihood (slow path) is used: optionally parallelizes the C9 scan across processes
  (each process builds its own likelihood object to avoid unsafe sharing)

Run:
  python -u impact_c9_blocks.py --out-json out.json

Examples:
  python -u impact_c9_blocks.py --out-json out.json --npts 61
  python -u impact_c9_blocks.py --out-json out.json --threads 8
  python -u impact_c9_blocks.py --out-json out.json --scan-procs 6
"""

import argparse
import json
import math
import os
import time
from collections import OrderedDict
from importlib import import_module


import flavio


# Default scan settings for Delta C9 in WET/flavio basis
C9_MIN_DEFAULT = -2.5
C9_MAX_DEFAULT = +1.0
C9_NPTS_DEFAULT = 141  # odd recommended


def load_groups(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_obs_tuple(x):
    return isinstance(x, tuple) and len(x) >= 1 and isinstance(x[0], str)


def normalize_obs_key(k):
    """
    Normalize a constraint key to a canonical observable tuple:
      - string "obs" -> ("obs",)
      - tuple ("obs", a, b, ...) -> itself
    """
    if isinstance(k, str):
        return (k,)
    if is_obs_tuple(k):
        return k
    return None


def iter_measurement_constraint_keys(m):
    """
    Yield constrained observable keys from a Measurement using public API.
    For flavio Measurement/Constraints objects, m.all_parameters exists and
    contains strings/tuples for all constrained observables.
    """
    try:
        for k in m.all_parameters:
            nk = normalize_obs_key(k)
            if nk is not None:
                yield nk
    except Exception:
        return


def collect_constrained_obs_for_base_names(base_names):
    """
    Given a list of base observable names (strings), find all constrained keys in
    the measurement DB matching those names, including binned tuples.
    """
    base_names = set(base_names)
    found = set()

    for m in flavio.Measurement.instances.values():
        for obs in iter_measurement_constraint_keys(m):
            if obs and obs[0] in base_names:
                found.add(obs)

    return sorted(found)


def make_wc_delta_c9(c9_value):
    """
    Build a WC object with Delta C9 (NP shift) in the WET/flavio basis.
    Compatible with flavio.WilsonCoefficients, otherwise uses wilson.Wilson.
    """
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
    """
    Estimate best-fit C9 and 68% interval from -2 Delta logL = 1 rule.
    logl is log-likelihood (higher is better).
    """
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

    if left is not None and right is not None:
        sigma = 0.5 * (right - left)
    else:
        sigma = float("nan")

    return c9_best, sigma, left, right, ll_best


def get_likelihood_classes():
    """
    Return (FastLikelihoodClass or None, LikelihoodClass).
    Works across flavio versions by trying import paths.
    """
    fast_cls = None
    like_cls = None

    try:
        mod = import_module("flavio.statistics.likelihood")
        fast_cls = getattr(mod, "FastLikelihood", None)
        like_cls = getattr(mod, "Likelihood", None)
    except Exception:
        pass

    if like_cls is None:
        like_cls = getattr(flavio.statistics, "Likelihood", None)
    if fast_cls is None:
        fast_cls = getattr(flavio.statistics, "FastLikelihood", None)

    if like_cls is None and fast_cls is None:
        try:
            import pkgutil
            subs = [m.name for m in pkgutil.iter_modules(flavio.statistics.__path__)]
        except Exception:
            subs = []
        raise RuntimeError(
            "Could not find Likelihood/FastLikelihood in this flavio version.\n"
            f"flavio version: {getattr(flavio, '__version__', 'unknown')}\n"
            f"flavio.statistics submodules: {subs}"
        )

    if like_cls is None and fast_cls is not None:
        like_cls = fast_cls

    return fast_cls, like_cls


def build_likelihood(name, observables, prefer_fast=True, threads=1):
    """
    Build either FastLikelihood (preferred if available) or Likelihood.
    Returns (likelihood_obj, is_fast).
    """
    fast_cls, like_cls = get_likelihood_classes()
    use_fast = prefer_fast and (fast_cls is not None)

    Like = fast_cls if use_fast else like_cls

    ctor_attempts = []
    if Like is fast_cls:
        ctor_attempts.append(dict(
            name=name,
            par_obj=flavio.default_parameters,
            fit_parameters=[],
            nuisance_parameters="all",
            observables=observables,
            exclude_measurements=None,
            include_measurements=None,
        ))
        ctor_attempts.append(dict(
            name=name,
            par_obj=flavio.default_parameters,
            fit_parameters=[],
            nuisance_parameters="all",
            observables=observables,
        ))
    else:
        ctor_attempts.append(dict(
            par_obj=flavio.default_parameters,
            observables=observables,
            include_measurements=None,
            exclude_measurements=None,
        ))
        ctor_attempts.append(dict(
            par_obj=flavio.default_parameters,
            observables=observables,
        ))
        ctor_attempts.append(dict(
            name=name,
            par_obj=flavio.default_parameters,
            observables=observables,
        ))

    last_err = None
    like_obj = None
    for kwargs in ctor_attempts:
        try:
            like_obj = Like(**kwargs)
            break
        except TypeError as e:
            last_err = e

    if like_obj is None:
        raise RuntimeError(f"Failed to construct likelihood object. Last error: {last_err}")

    if Like is fast_cls:
        # Precompute pseudo-measurement, can use multiple threads in many versions
        if hasattr(like_obj, "make_measurement"):
            like_obj.make_measurement(N=100, Nexp=5000, threads=int(threads))
        elif hasattr(like_obj, "make_measurements"):
            like_obj.make_measurements(N=100, Nexp=5000, threads=int(threads))

    return like_obj, (Like is fast_cls)


def eval_logl(like_obj, wc):
    """
    Evaluate log-likelihood for given WC across class variants.
    """
    if hasattr(like_obj, "parameters_central"):
        par = like_obj.parameters_central
        return float(like_obj.log_likelihood(par, wc))

    try:
        return float(like_obj.log_likelihood(wc))
    except TypeError:
        pass

    try:
        return float(like_obj.log_likelihood(flavio.default_parameters.get_central_all(), wc))
    except Exception as e:
        raise RuntimeError(f"Could not evaluate log_likelihood with this flavio version: {e}")


def scan_c9_serial(like_obj, c9_grid, label="", report_every=10):
    t0 = time.time()
    out = []
    n = len(c9_grid)
    for i, c9 in enumerate(c9_grid, start=1):
        wc = make_wc_delta_c9(c9)
        out.append(eval_logl(like_obj, wc))

        if report_every and (i % report_every == 0 or i == n):
            dt = time.time() - t0
            rate = i / dt if dt > 0 else float("inf")
            eta = (n - i) / rate if rate > 0 else float("inf")
            print(f"[{label}] {i}/{n} points, elapsed {dt:.1f}s, eta {eta:.1f}s", flush=True)

    return out


def _worker_scan_payload(payload):
    """
    Worker entry for multiprocessing scan on slow Likelihood.
    Each worker builds its own likelihood object (safe) and evaluates a chunk.
    """
    name = payload["name"]
    observables = payload["observables"]
    c9_chunk = payload["c9_chunk"]
    prefer_fast = payload["prefer_fast"]
    threads = payload["threads"]  # threads for FastLikelihood precompute if it happens
    # Build likelihood inside worker
    like_obj, _ = build_likelihood(name=name, observables=observables, prefer_fast=prefer_fast, threads=threads)
    out = []
    for c9 in c9_chunk:
        wc = make_wc_delta_c9(c9)
        out.append(eval_logl(like_obj, wc))
    return out


def scan_c9_parallel_build_per_process(observables, c9_grid, label, prefer_fast, threads, nprocs, report_every=10):
    """
    Parallel scan by splitting grid across processes. Each process builds its own likelihood object.
    This is mainly intended for the slow Likelihood path where scanning dominates.
    """
    if nprocs <= 1:
        like_obj, _ = build_likelihood(name=f"{label}_like", observables=observables, prefer_fast=prefer_fast, threads=threads)
        return scan_c9_serial(like_obj, c9_grid, label=label, report_every=report_every)

    from multiprocessing import get_context

    ctx = get_context("spawn")  # macOS safe
    chunks = [[] for _ in range(nprocs)]
    for i, c9 in enumerate(c9_grid):
        chunks[i % nprocs].append(c9)

    payloads = []
    for i in range(nprocs):
        payloads.append({
            "name": f"{label}_like_p{i}",
            "observables": observables,
            "c9_chunk": chunks[i],
            "prefer_fast": prefer_fast,
            "threads": threads,
        })

    t0 = time.time()
    results = []
    with ctx.Pool(processes=nprocs) as pool:
        # imap_unordered gives incremental progress
        for j, out_chunk in enumerate(pool.imap_unordered(_worker_scan_payload, payloads), start=1):
            results.append(out_chunk)
            if report_every:
                dt = time.time() - t0
                print(f"[{label}] finished {j}/{nprocs} worker chunks, elapsed {dt:.1f}s", flush=True)

    # Reconstruct in original grid order because chunks were round-robin
    out_full = [None] * len(c9_grid)
    # Map from c9 value to indices (grid may be unique, still do robust mapping)
    idx_map = {}
    for idx, c9 in enumerate(c9_grid):
        idx_map.setdefault(c9, []).append(idx)

    # Flatten chunks in the same round-robin assignment order
    # We do not know which worker returned which chunk in results, so rebuild deterministically:
    # Use the original chunks list and the returned chunk values by sorting payload order.
    # Instead, gather pool results with starmap order is easier. Use ordered map:
    # To keep this simple and deterministic, redo pool.map with payloads order if needed.
    # We already used unordered; rebuild by a second ordered pass on small data is wasteful.
    # Better: use ordered imap with payloads order.
    # We will switch to ordered imap here.
    # (Kept for safety: if here, we do one more pass ordered.)
    out_full = []
    # ordered second pass:
    with ctx.Pool(processes=nprocs) as pool2:
        ordered_chunks_out = list(pool2.imap(_worker_scan_payload, payloads))
    # Now fill round-robin
    out_full = [None] * len(c9_grid)
    ptrs = [0] * nprocs
    for gi in range(len(c9_grid)):
        p = gi % nprocs
        out_full[gi] = ordered_chunks_out[p][ptrs[p]]
        ptrs[p] += 1

    return out_full


def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.4g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="out.json", help="Path to out.json produced by the listing script")
    ap.add_argument("--c9min", type=float, default=C9_MIN_DEFAULT)
    ap.add_argument("--c9max", type=float, default=C9_MAX_DEFAULT)
    ap.add_argument("--npts", type=int, default=C9_NPTS_DEFAULT)
    ap.add_argument("--use-fast", type=int, default=1, help="Prefer FastLikelihood if available (1/0)")
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1),
                    help="Threads for FastLikelihood make_measurement (ignored for slow Likelihood)")
    ap.add_argument("--scan-procs", type=int, default=1,
                    help="Parallel processes for slow likelihood scan (each builds its own likelihood). "
                         "Set >1 only if you are using Likelihood or forcing --use-fast 0.")
    ap.add_argument("--report-every", type=int, default=10, help="Progress print frequency (grid points)")
    args = ap.parse_args()

    groups = load_groups(args.out_json)

    blocks = OrderedDict()
    blocks["K* angular (B0->K*0mumu)"] = groups["B0->K*0mumu"]["angular"]
    blocks["K* dBR/dq2 (B0->K*0mumu)"] = groups["B0->K*0mumu"]["rate"]
    blocks["K dBR/dq2 (B+->Kmumu)"] = groups["B+->Kmumu"]["rate"]
    blocks["phi angular (Bs->phimumu)"] = groups["Bs->phimumu"]["angular"]
    blocks["phi dBR/dq2 (Bs->phimumu)"] = groups["Bs->phimumu"]["rate"]

    block_obs = OrderedDict()
    for bname, base_list in blocks.items():
        obs_tuples = collect_constrained_obs_for_base_names(base_list)
        block_obs[bname] = obs_tuples

    print("\nResolved constrained observables per block (including bins if present):")
    for bname, obs in block_obs.items():
        print(f"\n[{bname}]  n={len(obs)}")
        for o in obs[:30]:
            print("  ", o)
        if len(obs) > 30:
            print(f"  ... ({len(obs)-30} more)")

    all_obs = []
    for obs in block_obs.values():
        all_obs.extend(obs)

    seen = set()
    all_obs_unique = []
    for o in all_obs:
        if o not in seen:
            all_obs_unique.append(o)
            seen.add(o)

    if not all_obs_unique:
        raise RuntimeError("No constrained observables found.")

    c9_grid = grid_linspace(args.c9min, args.c9max, args.npts)
    prefer_fast = bool(args.use_fast)

    print("\nBuilding likelihood for ALL blocks...")
    t_build = time.time()
    like_all, is_fast_all = build_likelihood(
        name="all_blocks",
        observables=all_obs_unique,
        prefer_fast=prefer_fast,
        threads=args.threads
    )
    print(f"Built in {time.time() - t_build:.1f}s (using {'FastLikelihood' if is_fast_all else 'Likelihood'})", flush=True)

    # Scan ALL
    if is_fast_all or args.scan_procs <= 1:
        ll_all = scan_c9_serial(like_all, c9_grid, label="ALL", report_every=args.report_every)
    else:
        ll_all = scan_c9_parallel_build_per_process(
            observables=all_obs_unique,
            c9_grid=c9_grid,
            label="ALL",
            prefer_fast=prefer_fast,
            threads=args.threads,
            nprocs=args.scan_procs,
            report_every=args.report_every
        )

    c9_best_all, sig_all, l_all, r_all, _ = estimate_best_and_sigma(c9_grid, ll_all)

    rows = []
    for drop_name, drop_obs in block_obs.items():
        keep = [o for o in all_obs_unique if o not in set(drop_obs)]
        print(f"\nBuilding likelihood without block: {drop_name}")
        t_build = time.time()
        like_drop, is_fast_drop = build_likelihood(
            name=f"minus_{drop_name}",
            observables=keep,
            prefer_fast=prefer_fast,
            threads=args.threads
        )
        print(f"Built in {time.time() - t_build:.1f}s (using {'FastLikelihood' if is_fast_drop else 'Likelihood'})", flush=True)

        label = f"minus:{drop_name}"

        if is_fast_drop or args.scan_procs <= 1:
            ll_drop = scan_c9_serial(like_drop, c9_grid, label=label, report_every=args.report_every)
        else:
            ll_drop = scan_c9_parallel_build_per_process(
                observables=keep,
                c9_grid=c9_grid,
                label=label,
                prefer_fast=prefer_fast,
                threads=args.threads,
                nprocs=args.scan_procs,
                report_every=args.report_every
            )

        c9_best_drop, sig_drop, _, _, _ = estimate_best_and_sigma(c9_grid, ll_drop)

        delta_invvar = 0.0
        if not (math.isnan(sig_all) or math.isnan(sig_drop) or sig_all == 0 or sig_drop == 0):
            delta_invvar = (1.0 / (sig_all * sig_all)) - (1.0 / (sig_drop * sig_drop))

        rows.append({
            "dropped_block": drop_name,
            "n_drop_obs": len(drop_obs),
            "C9_best_all": c9_best_all,
            "sigma_all": sig_all,
            "C9_best_minus": c9_best_drop,
            "sigma_minus": sig_drop,
            "delta_C9_best": (c9_best_all - c9_best_drop),
            "delta_invvar": delta_invvar,
        })

    print("\n" + "=" * 96)
    print("C9 impact summary (1D scan in Delta C9_bsmumu)")
    print("=" * 96)
    print(f"ALL blocks:  C9_best={fmt(c9_best_all)}  sigma~{fmt(sig_all)}  68%~[{fmt(l_all)}, {fmt(r_all)}]")

    print("\nImpact per dropped block (bigger |delta_invvar| means the dropped block was more constraining):")
    header = (
        "Dropped block",
        "Nobs",
        "C9_best(all)",
        "sig(all)",
        "C9_best(-block)",
        "sig(-block)",
        "delta_best",
        "delta_invvar",
    )
    print("{:<34} {:>5} {:>11} {:>10} {:>14} {:>11} {:>11} {:>12}".format(*header))
    for r in sorted(rows, key=lambda x: abs(x["delta_invvar"]), reverse=True):
        print("{:<34} {:>5} {:>11} {:>10} {:>14} {:>11} {:>11} {:>12}".format(
            r["dropped_block"][:34],
            r["n_drop_obs"],
            fmt(r["C9_best_all"]),
            fmt(r["sigma_all"]),
            fmt(r["C9_best_minus"]),
            fmt(r["sigma_minus"]),
            fmt(r["delta_C9_best"]),
            fmt(r["delta_invvar"]),
        ))

    print("\nNotes:")
    print("- Use --threads to speed up FastLikelihood precomputation (if FastLikelihood is available).")
    print("- Use --scan-procs > 1 to parallelize the slow Likelihood scan across processes.")
    print("- For progress output in real time, run with: python -u impact_c9_blocks.py ...")


if __name__ == "__main__":
    main()
