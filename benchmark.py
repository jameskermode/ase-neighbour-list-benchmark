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

    measure = getattr(args, "measure", "build")
    if measure == "update":
        # Per-step Verlet update check (needs_rebuild). Only device backends with
        # the DeviceNeighborList protocol support it; the build is done once,
        # untimed, inside make_update_step.
        if not backend.supports_update():
            print(json.dumps({"status": "unsupported",
                              "reason": "no device update check"}))
            return 0
        try:
            target = backend.make_update_step(atoms, args.cutoff, args.skin)
        except Exception as exc:  # pragma: no cover - env dependent
            print(json.dumps({"status": "error", "reason": str(exc)}))
            return 0
        # The check is ~tens of microseconds; batch many per timed repeat so the
        # per-call median is robust against launch jitter / timer resolution.
        inner = max(args.inner, 1)
    else:
        target = lambda: backend.compute(atoms, args.cutoff)  # noqa: E731
        inner = 1

    # Warmup (captures first-call setup cost: threadpool spin-up, JIT, etc.).
    first_call_s = None
    for _ in range(max(args.warmup, 1)):
        t0 = time.perf_counter()
        target()
        t1 = time.perf_counter()
        if first_call_s is None:
            first_call_s = t1 - t0

    # Timed repeats -- NO tracemalloc here. tracemalloc traces every Python
    # allocation and inflates allocation-heavy backends (e.g. the cKDTree
    # per-atom assembly loop) by ~16x while barely touching array-based
    # backends, which would make the comparison meaningless. For the update
    # measurement each repeat times `inner` calls and reports the per-call time.
    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            target()
        t1 = time.perf_counter()
        times.append((t1 - t0) / inner)

    # One separate, UNTIMED run under tracemalloc for the Python peak.
    tracemalloc.start()
    target()
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


def _run_one(name, kind, n_target, cutoff, threads, repeats, warmup, timeout,
             measure="build", inner=1, skin=0.4):
    """Run a single measurement in a subprocess; return a result dict."""
    env, label = _thread_env(threads)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--name", name, "--kind", kind, "--n-target", str(n_target),
           "--cutoff", str(cutoff), "--repeats", str(repeats),
           "--warmup", str(warmup), "--measure", measure,
           "--inner", str(inner), "--skin", str(skin)]
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

    # --- build phase: full neighbour-list rebuild timings ------------------ #
    if not args.only_update:
        rows = _run_phase(jobs, thread_settings, repeats, args,
                          measure="build", inner=1, label="build")
        _write_csv(out_dir, rows, "results.csv", "results_detail.csv",
                   label="build", append=args.append)
        print(f"\nWrote {out_dir/'results.csv'} and "
              f"{out_dir/'results_detail.csv'}")

    # --- update phase: per-step Verlet update check (needs_rebuild) --------- #
    # Only device backends implementing the DeviceNeighborList protocol have a
    # device update check; host backends rebuild from scratch.
    if not args.no_update:
        from backends import get_backend
        update_runnable = [n for n in runnable if get_backend(n).supports_update()]
        # The update check (needs_rebuild) is displacement-based and cutoff-
        # independent, so only the size sweep is run -- no cutoff sweep.
        ujobs = []
        if update_runnable and not args.no_size_sweep:
            for kind in systems:
                for n in sizes:
                    for name in update_runnable:
                        ujobs.append((name, kind, n, args.cutoff))
        if ujobs:
            print(f"\n=== update check (needs_rebuild), {args.update_inner} "
                  f"calls/repeat ===")
            urows = _run_phase(ujobs, thread_settings, repeats, args,
                               measure="update", inner=args.update_inner,
                               label="update")
            _write_csv(out_dir, urows, "update_results.csv",
                       "update_results_detail.csv", label="update",
                       append=args.append)
            print(f"\nWrote {out_dir/'update_results.csv'} and "
                  f"{out_dir/'update_results_detail.csv'}")
        else:
            print("\n(no update-capable backends among the selected; "
                  "skipping update phase)")
    return 0


def _run_phase(jobs, thread_settings, repeats, args, *, measure, inner, label):
    """Run a set of measurements (one phase) and return the result rows."""
    rows = []
    for threads in thread_settings:
        _, tlabel = _thread_env(threads)
        print(f"=== thread setting: {tlabel} ({label}) ===")
        for name, kind, n, cutoff in jobs:
            t0 = time.perf_counter()
            res = _run_one(name, kind, n, cutoff, threads, repeats,
                           args.warmup, args.timeout, measure=measure,
                           inner=inner, skin=args.skin)
            wall = time.perf_counter() - t0
            rows.append(res)
            if res.get("status") == "ok":
                # The worker reports the timed quantity under build_s_*; here it
                # is a build or an update-check time depending on `measure`.
                unit = "ms" if measure == "build" else "us"
                scale = 1e3 if measure == "build" else 1e6
                print(f"  {res['threads']:>8s} {name:24s} {kind:6s} "
                      f"N={res['n_atoms']:>8d} rc={cutoff:<4} "
                      f"median={res['build_s_median']*scale:9.2f} {unit}  "
                      f"min={res['build_s_min']*scale:9.2f} {unit}")
            else:
                print(f"  {tlabel:>8s} {name:24s} {kind:6s} N~{n:>8d} "
                      f"rc={cutoff:<4} -> {res.get('status')} "
                      f"{res.get('reason','')} (after {wall:.1f}s)")
    return rows


def _write_csv(out_dir: Path, rows: list[dict], primary_name: str,
               detail_name: str, *, label: str, append: bool = False):
    """Write a phase's results. ``label`` names the timed quantity, so the build
    phase emits build_s_* columns and the update phase update_s_* columns."""
    mode = "a" if append else "w"
    primary = out_dir / primary_name
    detail = out_dir / detail_name
    med_col, min_col = f"{label}_s_median", f"{label}_s_min"
    primary_cols = ["backend", "n_atoms", "cutoff", "threads",
                    med_col, min_col, "peak_mb"]
    detail_cols = primary_cols + ["system", "first_call_s", "tracemalloc_mb",
                                  "n_repeats", "status"]
    write_primary_header = not (append and primary.exists())
    write_detail_header = not (append and detail.exists())
    # Primary CSV: only successful runs.
    with open(primary, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=primary_cols, extrasaction="ignore")
        if write_primary_header:
            w.writeheader()
        for r in rows:
            if r.get("status") == "ok":
                w.writerow({
                    "backend": r["backend"], "n_atoms": r["n_atoms"],
                    "cutoff": r["cutoff"], "threads": r["threads"],
                    med_col: f"{r['build_s_median']:.9g}",
                    min_col: f"{r['build_s_min']:.9g}",
                    "peak_mb": f"{r['peak_mb']:.3f}",
                })
    # Detail CSV: all runs incl. drops, first-call, tracemalloc.
    with open(detail, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=detail_cols, extrasaction="ignore")
        if write_detail_header:
            w.writeheader()
        for r in rows:
            row = {c: r.get(c, "") for c in detail_cols}
            row[med_col] = r.get("build_s_median", "")
            row[min_col] = r.get("build_s_min", "")
            w.writerow(row)


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
    # Verlet update-check phase (device backends only).
    p.add_argument("--no-update", dest="no_update", action="store_true",
                   help="skip the per-step update-check (needs_rebuild) phase")
    p.add_argument("--only-update", dest="only_update", action="store_true",
                   help="run only the update-check phase (leave results.csv intact)")
    p.add_argument("--skin", type=float, default=0.4,
                   help="Verlet skin (A) for the update-check phase")
    p.add_argument("--update-inner", dest="update_inner", type=int, default=64,
                   help="needs_rebuild calls timed per repeat in the update phase")
    # Worker-internal (set by the parent when spawning subprocess workers).
    p.add_argument("--measure", default="build", choices=["build", "update"],
                   help=argparse.SUPPRESS)
    p.add_argument("--inner", type=int, default=1, help=argparse.SUPPRESS)
    return p


def main():
    args = build_argparser().parse_args()
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
