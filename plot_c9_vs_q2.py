#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = ["C0", "C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "h"]


def load(path):
    with open(path) as f:
        return json.load(f)


def series_label(data, path):
    m = data.get("metadata", {})
    mode = m.get("mode", "")
    kind = m.get("obs_kind", "")
    if mode or kind:
        return f"{mode} {kind}".strip()
    return Path(path).stem


def plot_series(ax, rows, label, color, marker, offset=0.0):
    for r in rows:
        xc = 0.5 * (r["q2min"] + r["q2max"]) + offset
        y = r["c9_best"]
        if y is None:
            continue
        lo, hi = r["cl_lo"], r["cl_hi"]
        if lo is not None and hi is not None:
            yerr = [[y - lo], [hi - y]]
        elif r["sigma"] is not None:
            yerr = [[r["sigma"]], [r["sigma"]]]
        else:
            yerr = [[0], [0]]
        ax.errorbar(xc, y, yerr=yerr,
                    fmt=marker, color=color, capsize=3, label=label)
        label = None  # only first point gets legend entry


def make_figure(datasets, key, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(datasets)
    for i, (data, lbl) in enumerate(datasets):
        rows = data.get(key) or data["results"]
        off = (i - (n - 1) / 2) * 0.15  # spread points horizontally
        plot_series(ax, rows, lbl, COLORS[i % len(COLORS)],
                    MARKERS[i % len(MARKERS)], offset=off)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.axvspan(8.68, 14.18, color="lightgrey", alpha=0.4, label="charmonium veto")
    ax.set_xlabel(r"$q^2$ [GeV$^2$]")
    ax.set_ylabel(r"$\Delta C_9$")
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Legend labels, one per input file. "
                         "Falls back to metadata mode+obs_kind or filename stem.")
    ap.add_argument("--output", default=None, help="Output file prefix")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.input):
        ap.error(f"--labels count ({len(args.labels)}) must match "
                 f"--input count ({len(args.input)})")

    all_data = []
    for i, p in enumerate(args.input):
        d = load(p)
        lbl = args.labels[i] if args.labels else series_label(d, p)
        all_data.append((d, lbl))

    # Nominal figure
    fig_nom = make_figure(all_data, "results")
    if args.title:
        fig_nom.axes[0].set_title(args.title)

    # Clipped figure (only if any file has clipped data)
    has_clipped = any(d.get("clipped_results") for d, _ in all_data)
    fig_clip = None
    if has_clipped:
        fig_clip = make_figure(all_data, "clipped_results")
        suffix = ""
        for d, _ in all_data:
            cs = d.get("metadata", {}).get("clip_sigma")
            if cs is not None:
                suffix = f" (clipped >{cs}\u03c3)"
                break
        fig_clip.axes[0].set_title((args.title or "") + suffix)

    if args.output:
        fig_nom.savefig(f"{args.output}_nominal.png", dpi=150)
        print(f"Saved {args.output}_nominal.png")
        if fig_clip:
            fig_clip.savefig(f"{args.output}_clipped.png", dpi=150)
            print(f"Saved {args.output}_clipped.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
