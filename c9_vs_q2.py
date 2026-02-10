#!/usr/bin/env python3
import argparse
import json
import math
import time
import warnings
from collections import OrderedDict, defaultdict
from importlib import import_module

import flavio


def load_groups(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def collect_constrained_obs_for_base_names(base_names):
    base_names = set(base_names)
    found = set()
    for m in flavio.Measurement.instances.values():
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


def build_fastlikelihood(name, observables, threads=1, fast_N=100, fast_Nexp=5000):
    FastLikelihood = get_fastlikelihood_class()
    fl = FastLikelihood(
        name=name,
        par_obj=flavio.default_parameters,
        fit_parameters=[],
        nuisance_parameters="all",
        observables=observables,
        exclude_measurements=None,
        include_measurements=None,
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


def filter_to_exact_bin(keys, q2bin, veto_windows):
    q2lo, q2hi = q2bin
    out = []
    for k in keys:
        if not is_binned_key(k):
            continue
        q2min = float(k[1])
        q2max = float(k[2])
        if abs(q2min - q2lo) < 1e-9 and abs(q2max - q2hi) < 1e-9:
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
    ap.add_argument("--out-json", default="out.json")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fast-N", type=int, default=100)
    ap.add_argument("--fast-Nexp", type=int, default=5000)
    ap.add_argument("--c9min", type=float, default=-3.0)
    ap.add_argument("--c9max", type=float, default=+1.0)
    ap.add_argument("--npts", type=int, default=161)
    ap.add_argument("--report-every", type=int, default=10)

    ap.add_argument("--mode", choices=["Kstar", "K", "phi"], default="Kstar",
                    help="Which decay mode to use for the per-bin fit.")
    ap.add_argument("--obs-kind", choices=["angular", "rate", "both"], default="angular",
                    help="Use angular observables, rate observables, or both.")

    ap.add_argument("--bins", default="0.1-0.98,1.1-2.5,2.5-4,4-6,15-17,17-19",
                    help="Comma-separated q2 bins, each as lo-hi in GeV^2.")
    ap.add_argument("--veto", default="8.68-14.18",
                    help="Comma-separated veto windows lo-hi. Use empty string to disable.")
    ap.add_argument("--quiet-warnings", action="store_true")
    args = ap.parse_args()

    if args.quiet_warnings:
        warnings.filterwarnings("ignore", message=".*QCDF corrections should not be trusted.*")
        warnings.filterwarnings("ignore", message=".*predictions in the region of narrow charmonium resonances.*")

    groups = load_groups(args.out_json)

    # Choose base-name lists from out.json groups
    if args.mode == "Kstar":
        base = groups["B0->K*0mumu"]
    elif args.mode == "K":
        base = groups["B+->Kmumu"]
    else:
        base = groups["Bs->phimumu"]

    base_names = []
    if args.obs_kind in ["angular", "both"]:
        base_names.extend(base["angular"])
    if args.obs_kind in ["rate", "both"]:
        base_names.extend(base["rate"])

    # Resolve to constrained keys across the DB
    all_keys = collect_constrained_obs_for_base_names(base_names)

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
    print(f"Bins={q2bins}", flush=True)
    print(f"Veto={veto_windows}", flush=True)
    print(f"Grid: [{args.c9min},{args.c9max}] npts={args.npts}", flush=True)

    results = []
    for (q2lo, q2hi) in q2bins:
        keys_bin = filter_to_exact_bin(all_keys, (q2lo, q2hi), veto_windows=veto_windows)
        if not keys_bin:
            print(f"\nBin {q2lo}-{q2hi}: no exact-edge constrained observables found, skipping.", flush=True)
            continue

        print(f"\nBin {q2lo}-{q2hi}: nobs={len(keys_bin)} build FastLikelihood...", flush=True)
        t0 = time.time()
        fl = build_fastlikelihood(
            name=f"{args.mode}_{args.obs_kind}_{q2lo}_{q2hi}",
            observables=keys_bin,
            threads=args.threads,
            fast_N=args.fast_N,
            fast_Nexp=args.fast_Nexp,
        )
        print(f"Bin {q2lo}-{q2hi}: built in {time.time()-t0:.1f}s", flush=True)

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
    print("- It uses only observables whose bin edges exactly match the requested bin list.")
    print("- Veto removes any bin overlapping the specified windows.")
    print("- For speed while iterating: try --fast-N 50 --fast-Nexp 1000 and fewer --npts.")


if __name__ == "__main__":
    main()
