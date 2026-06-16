"""Benchmark harness for ASE neighbour-list backends.

Times only the neighbour-list build (system construction is excluded), using a
best-of-N protocol after one warmup, and reports median + min. Each
backend x size measurement runs in its own subprocess so that:

* a per-run timeout cannot let an O(N^2) fallback hang the whole sweep;
* peak RSS (resource.ru_maxrss) is measured cleanly for that single build;
* thread pinning (OMP_NUM_THREADS etc.) is set in the child's environment
  *before* the compiled backends are imported.

The benchmark refuses to run until correctness.py passes.

CLI::

    uv run python benchmark.py --sizes 500,4000,32000 --cutoff 5.0 \
        --backends ase,ase-newprim,matscipy,vesin --threads 1 --out results/

Flags:
  --sizes        comma list of target atom counts (default 500,4000,32000,100000)
  --cutoff       cutoff in Angstrom for the size sweep (default 5.0)
  --cutoff-sweep comma list of cutoffs to sweep at --sweep-size (default off)
  --sweep-size   fixed size for the cutoff sweep (default 32000)
  --systems      comma list of system kinds (default cubic; also lowsym,slab)
  --backends     comma list (default all)
  --threads      comma list of thread settings: an int pins OMP_NUM_THREADS,
                 0 means unpinned (default '1')
  --repeats      timed repeats after warmup (default 5, min enforced 5)
  --warmup       warmup runs (default 1)
  --timeout      per-run timeout in seconds (default 300)
  --out          output directory (default results/)
  --skip-correctness   bypass the correctness gate (NOT recommended)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Exact CSV schema required by the spec.
CSV_COLUMNS = ["backend", "n_atoms", "cutoff", "threads",
               "build_s_median", "build_s_min", "peak_mb"]
# Extra detail written alongside (first-call vs steady-state, etc.).
DETAIL_COLUMNS = CSV_COLUMNS + ["system", "first_call_s", "tracemalloc_mb",
                                "n_repeats", "status"]


def _ru_maxrss_mb(ru_maxrss: int) -> float:
    """ru_maxrss is bytes on macOS, kilobytes on Linux."""
    if sys.platform == "darwin":
        return ru_maxrss / (1024 * 1024)
    return ru_maxrss / 1024


# --------------------------------------------------------------------------- #
# Worker: time ONE backend on ONE system, print a JSON line.
# --------------------------------------------------------------------------- #
def run_worker(args):
    import tracemalloc

    from backends import get_backend
    from systems import build

    backend = get_backend(args.name)
    ok, why = backend.available()
    if not ok:
        print(json.dumps({"status": "unavailable", "reason": why}))
        return 0

    atoms = build(args.kind, args.n_target)   # construction excluded from timing
    n_atoms = len(atoms)

    # Warmup (captures first-call setup cost: threadpool spin-up, JIT, etc.).
    first_call_s = None
    for _ in range(max(args.warmup, 1)):
        t0 = time.perf_counter()
        backend.compute(atoms, args.cutoff)
        t1 = time.perf_counter()
        if first_call_s is None:
            first_call_s = t1 - t0

    # Timed repeats -- NO tracemalloc here. tracemalloc traces every Python
    # allocation and inflates allocation-heavy backends (e.g. the cKDTree
    # per-atom assembly loop) by ~16x while barely touching array-based
    # backends, which would make the comparison meaningless.
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        backend.compute(atoms, args.cutoff)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # One separate, UNTIMED run under tracemalloc for the Python peak.
    tracemalloc.start()
    backend.compute(atoms, args.cutoff)
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "status": "ok",
        "n_atoms": n_atoms,
        "build_s_median": statistics.median(times),
        "build_s_min": min(times),
        "first_call_s": first_call_s,
        "tracemalloc_mb": tm_peak / (1024 * 1024),
        "peak_mb": _ru_maxrss_mb(ru),
        "n_repeats": len(times),
    }))
    return 0


# --------------------------------------------------------------------------- #
# Parent: spawn one worker per (backend, system, size, cutoff, threads).
# --------------------------------------------------------------------------- #
def _thread_env(threads):
    """Return (env, label) for a thread setting. threads<=0 means unpinned."""
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        env.pop(k, None)
    if threads and threads > 0:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
            env[k] = str(threads)
        return env, str(threads)
    return env, "unpinned"


def _run_one(name, kind, n_target, cutoff, threads, repeats, warmup, timeout):
    """Run a single measurement in a subprocess; return a result dict."""
    env, label = _thread_env(threads)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--name", name, "--kind", kind, "--n-target", str(n_target),
           "--cutoff", str(cutoff), "--repeats", str(repeats),
           "--warmup", str(warmup)]
    base = {"backend": name, "system": kind, "cutoff": cutoff, "threads": label}
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**base, "n_atoms": n_target, "status": "timeout"}
    if proc.returncode != 0:
        # Negative return code => killed by signal (likely OOM).
        reason = "killed/oom" if proc.returncode < 0 else "error"
        sys.stderr.write(proc.stderr[-500:])
        return {**base, "n_atoms": n_target, "status": reason}
    # The worker prints one JSON object on the last non-empty stdout line.
    line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    payload = json.loads(line)
    if payload.get("status") != "ok":
        return {**base, "n_atoms": n_target, "status": payload.get("status", "skip"),
                "reason": payload.get("reason", "")}
    return {**base, **payload}


def run_parent(args):
    from backends import resolve_backends
    import envinfo

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- correctness gate -------------------------------------------------- #
    if not args.skip_correctness:
        print("Running correctness gate before timing ...")
        import correctness
        errors = correctness.run_all(verbose=False)
        if errors:
            print("CORRECTNESS FAILED -- aborting benchmark:")
            for e in errors:
                print("  " + e.replace("\n", "\n  "))
            return 1
        print("Correctness gate passed.\n")

    # --- resolve backends -------------------------------------------------- #
    requested = args.backends.split(",") if args.backends else None
    resolved = resolve_backends(requested)
    runnable = []
    for b, ok, why in resolved:
        if ok:
            runnable.append(b.name)
        else:
            print(f"  skipping backend {b.name}: {why}")
    if not runnable:
        print("No runnable backends.")
        return 1

    # --- environment header ------------------------------------------------ #
    info = envinfo.collect()
    (out_dir / "environment.txt").write_text(envinfo.format_block(info) + "\n")
    print(envinfo.format_block(info))
    print()

    thread_settings = [int(t) for t in str(args.threads).split(",")]
    repeats = max(args.repeats, 5)

    # --- build the job matrix --------------------------------------------- #
    jobs = []  # (name, kind, n_target, cutoff)
    sizes = [int(s) for s in args.sizes.split(",")]
    systems = args.systems.split(",")
    if not args.no_size_sweep:
        for kind in systems:
            for n in sizes:
                for name in runnable:
                    jobs.append((name, kind, n, args.cutoff))
    if args.cutoff_sweep:
        for c in (float(x) for x in args.cutoff_sweep.split(",")):
            for name in runnable:
                jobs.append((name, "cubic", args.sweep_size, c))

    # --- run --------------------------------------------------------------- #
    rows = []
    for threads in thread_settings:
        _, label = _thread_env(threads)
        print(f"=== thread setting: {label} ===")
        for name, kind, n, cutoff in jobs:
            t0 = time.perf_counter()
            res = _run_one(name, kind, n, cutoff, threads, repeats,
                           args.warmup, args.timeout)
            wall = time.perf_counter() - t0
            rows.append(res)
            if res.get("status") == "ok":
                print(f"  {res['threads']:>8s} {name:12s} {kind:6s} "
                      f"N={res['n_atoms']:>8d} rc={cutoff:<4} "
                      f"median={res['build_s_median']*1e3:9.2f} ms  "
                      f"min={res['build_s_min']*1e3:9.2f} ms  "
                      f"peak={res['peak_mb']:7.1f} MB")
            else:
                print(f"  {label:>8s} {name:12s} {kind:6s} N~{n:>8d} rc={cutoff:<4} "
                      f"-> {res.get('status')} {res.get('reason','')} "
                      f"(after {wall:.1f}s)")

    _write_csv(out_dir, rows, append=args.append)
    print(f"\nWrote {out_dir/'results.csv'} and {out_dir/'results_detail.csv'}")
    return 0


def _write_csv(out_dir: Path, rows: list[dict], append: bool = False):
    mode = "a" if append else "w"
    primary = out_dir / "results.csv"
    detail = out_dir / "results_detail.csv"
    write_primary_header = not (append and primary.exists())
    write_detail_header = not (append and detail.exists())
    # Primary CSV: exact schema, only successful runs.
    with open(primary, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_primary_header:
            w.writeheader()
        for r in rows:
            if r.get("status") == "ok":
                w.writerow({
                    "backend": r["backend"], "n_atoms": r["n_atoms"],
                    "cutoff": r["cutoff"], "threads": r["threads"],
                    "build_s_median": f"{r['build_s_median']:.9g}",
                    "build_s_min": f"{r['build_s_min']:.9g}",
                    "peak_mb": f"{r['peak_mb']:.3f}",
                })
    # Detail CSV: all runs incl. drops, first-call, tracemalloc.
    with open(detail, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DETAIL_COLUMNS, extrasaction="ignore")
        if write_detail_header:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in DETAIL_COLUMNS})


# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--name")
    p.add_argument("--kind", default="cubic")
    p.add_argument("--n-target", type=int, dest="n_target")
    p.add_argument("--sizes", default="500,4000,32000,100000")
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--cutoff-sweep", dest="cutoff_sweep", default="")
    p.add_argument("--sweep-size", dest="sweep_size", type=int, default=32000)
    p.add_argument("--systems", default="cubic")
    p.add_argument("--backends", default="")
    p.add_argument("--threads", default="1")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", default="results/")
    p.add_argument("--append", action="store_true",
                   help="append to existing results CSVs instead of overwriting")
    p.add_argument("--no-size-sweep", dest="no_size_sweep", action="store_true",
                   help="skip the size x cutoff jobs (run only --cutoff-sweep)")
    p.add_argument("--skip-correctness", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
