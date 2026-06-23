"""Plot benchmark results.

Reads the detail CSV (which carries the system + status columns) and produces:

* ``build_time_vs_N.png``     -- log-log build time vs atom count, one line per
                                 backend (cubic system, primary cutoff, pinned
                                 threads by default).
* ``build_time_vs_cutoff.png``-- log-log build time vs cutoff at the fixed sweep
                                 size, one line per backend.

Usage::

    uv run python plots.py --results results/ [--threads 1] [--cutoff 5.0]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(path, time_col):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]
    for r in rows:
        r["n_atoms"] = int(r["n_atoms"])
        r["cutoff"] = float(r["cutoff"])
        r["y"] = float(r[time_col])
    return rows


def _series_by_backend(rows, xkey):
    series = defaultdict(list)
    for r in rows:
        series[r["backend"]].append((r[xkey], r["y"]))
    for b in series:
        series[b].sort()
    return series


def plot_vs_n(rows, threads, cutoff, system, out, *, ylabel, title):
    sel = [r for r in rows if r["threads"] == str(threads)
           and r["cutoff"] == cutoff and r["system"] == system]
    if not sel:
        print(f"  (no data for vs-N plot: threads={threads} cutoff={cutoff} system={system})")
        return
    series = _series_by_backend(sel, "n_atoms")
    fig, ax = plt.subplots(figsize=(7, 5))
    for backend, pts in sorted(series.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=backend)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of atoms N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_vs_cutoff(rows, threads, sweep_size, out):
    # cutoff sweep rows are tagged system=cubic at the fixed sweep size.
    sizes = sorted({r["n_atoms"] for r in rows
                    if r["system"] == "cubic" and r["threads"] == str(threads)})
    sel_size = min(sizes, key=lambda s: abs(s - sweep_size)) if sizes else None
    sel = [r for r in rows if r["threads"] == str(threads)
           and r["system"] == "cubic" and r["n_atoms"] == sel_size]
    cutoffs = sorted({r["cutoff"] for r in sel})
    if len(cutoffs) < 2:
        print("  (no cutoff-sweep data: run benchmark with --cutoff-sweep)")
        return
    series = _series_by_backend(sel, "cutoff")
    fig, ax = plt.subplots(figsize=(7, 5))
    for backend, pts in sorted(series.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="s", label=backend)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cutoff (Å)")
    ax.set_ylabel("build time (s, median)")
    ax.set_title(f"Neighbour-list build time vs cutoff\n(cubic, N={sel_size}, threads={threads})")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/")
    p.add_argument("--threads", default="1")
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--system", default="cubic")
    p.add_argument("--sweep-size", dest="sweep_size", type=int, default=32000)
    args = p.parse_args()

    d = Path(args.results)
    rows = _read(d / "results_detail.csv", "build_s_median")
    plot_vs_n(rows, args.threads, args.cutoff, args.system,
              d / "build_time_vs_N.png",
              ylabel="build time (s, median)",
              title=(f"Neighbour-list build time vs N\n({args.system}, "
                     f"cutoff {args.cutoff} Å, threads={args.threads})"))
    plot_vs_cutoff(rows, args.threads, args.sweep_size,
                   d / "build_time_vs_cutoff.png")

    # Update-check (needs_rebuild) plot -- separate output, device backends only.
    upath = d / "update_results_detail.csv"
    if upath.exists():
        urows = _read(upath, "update_s_median")
        plot_vs_n(urows, args.threads, args.cutoff, args.system,
                  d / "update_time_vs_N.png",
                  ylabel="update-check time (s, median, per call)",
                  title=(f"Verlet update check (needs_rebuild) vs N\n"
                         f"({args.system}, cutoff {args.cutoff} Å, "
                         f"threads={args.threads})"))
    else:
        print("  (no update_results_detail.csv; run benchmark to produce it)")


if __name__ == "__main__":
    main()
