# Findings: ASE neighbour-list backends vs matscipy & vesin

**For the ASE v4 Atoms/Calculator interface discussion.** All backends were first
proven to return *identical* neighbour lists (see Correctness) before any timing.

## Environment
- Platform: macOS-26.5.1-arm64 (Apple M3 Pro, 12 logical cores)
- Python 3.12.5; **ase 3.28.0, matscipy 1.2.0, vesin 0.5.8**, scipy 1.17.1, numpy 2.4.6
- Timing: `perf_counter`, median of 5 after 1 warmup; build only (Atoms construction excluded).
- `peak_mb` is whole-process peak RSS (`ru_maxrss`), so it includes a ~80 MB
  interpreter+import baseline — read it for *scaling/relative* comparison, not absolute.

## ⚠️ Correction to the brief's premise (verified in installed ASE 3.28.0)
The spec's class→algorithm mapping is **inverted** relative to released ASE:
- **`PrimitiveNeighborList` IS the cKDTree path** (scipy `cKDTree` + per-shift
  `query_ball_point`), and it is the **default** used by `ase.neighborlist.NeighborList`.
- **`NewPrimitiveNeighborList.build` just calls the cell-binning `primitive_neighbor_list`** —
  it is *not* a tree path; it is a thin wrapper over the binning function.

So there are **three classes but two algorithms**: cell-binning
(`primitive_neighbor_list` ≡ `NewPrimitiveNeighborList`) and cKDTree
(`PrimitiveNeighborList`, the default). Backends below are labelled by what they
actually run: `ase` = binning function, `ase-newprim` = binning via the class,
`ase-ckdtree` = the real tree path (= the default).

## Headline: build time, cubic fcc Ni, cutoff 5.0 Å, 1 thread (median)

| N | ase (binning) | ase-newprim | ase-ckdtree (default) | matscipy | vesin |
|--:|--:|--:|--:|--:|--:|
| 500 | 30 ms | 31 ms | 19 ms | 2.9 ms | **1.4 ms** |
| 4 000 | 126 ms | 136 ms | 151 ms | 14.6 ms | **10.8 ms** |
| 32 000 | 1.23 s | 1.31 s | 1.23 s | 0.12 s | **0.10 s** |
| 108 000 | 5.76 s | 5.76 s | 4.11 s | 0.40 s | **0.34 s** |
| 256 000 | (capped, >15 GB proj.) | (capped) | 9.95 s | 0.96 s | **0.84 s** |
| 1 000 188 | (capped) | (capped) | (capped, ~24 GB proj.) | **4.05 s** | 4.73 s |

Memory note: the ASE cell-binning paths were not run above ~108 k atoms on this
18 GB machine (binning peaks at 4.5 GB @108 k and grows ~linearly; 256 k/1 M
projected to exceed RAM). The cKDTree path uses roughly **half** the memory of
binning (2.3 GB vs 4.5 GB @108 k) so it reached 256 k; 1 M (~24 GB) was skipped.
matscipy/vesin reached 1 M comfortably.

## Who wins where
- **vesin is fastest from 500 up to ~256 k**; **matscipy overtakes vesin at ~1 M**
  (4.05 s vs 4.73 s) and uses **far less memory there** (3.6 GB vs 8.0 GB). The
  vesin↔matscipy crossover sits in the few-hundred-k–1 M range.
- **Compiled vs pure-Python is the dominant effect: matscipy/vesin are ~10–17×
  faster than any ASE path at every size**, and the gap is flat in N (all are
  ~O(N) at fixed density) — see `build_time_vs_N.png`, which shows two parallel
  bands an order of magnitude apart.
- **There is no crossover where an ASE path beats a compiled backend.** The only
  ASE-vs-ASE crossovers are minor: `ase-ckdtree` (the default) is fastest of the
  three ASE paths at 500 and again from ~50 k up, while binning is marginally
  faster at ~4 k; around 32 k they tie.

## Is the default (`PrimitiveNeighborList`, cKDTree) actually slower than binning?
**No — that premise is backwards for released ASE.** The default *is* the tree
path, and it is **competitive with or faster than cell-binning, and roughly 2×
leaner on memory**:
- @108 k: 4.11 s (tree) vs 5.76 s (binning) — tree **29 % faster**, half the RAM.
- Cutoff sensitivity @32 k (see `build_time_vs_cutoff.png`): the tree scales far
  better with cutoff. At rc=8 Å binning needs **9.5 GB and 7.5 s** while the tree
  needs **2.4 GB and 3.3 s** (2.2× faster). At rc=3 Å binning wins (0.30 s vs
  0.59 s). Binning's cost and memory blow up with neighbour count; the tree degrades gently.

So the v4 takeaway is **not** "switch the default to binning". If anything,
binning is the one to avoid at large N / large cutoff. The real takeaway is that
**both pure-Python paths are ~10× off a compiled cell list**, so v4 should make
it trivial to plug in a compiled backend (vesin/matscipy) via the
`(i, j, d, D, S)` contract, without making either a hard dependency.

## Where does the pure-Python time go — algorithm or array assembly?
**Array assembly, overwhelmingly** (profiled in `profile_ckdtree.py`). For the
cKDTree path @32 k (total build ≈ 1.2 s):
- scipy `cKDTree.query_ball_point` (the actual algorithm): **~182 ms (~15 %)**.
- The `for a in range(natoms)` per-atom assembly + the `bothways` Python
  doubling loop: **~1.0 s of pure-Python bytecode plus ~0.2 s of `numpy.array`
  calls and 1.8 M `list.append`s (~85 %)**.

Implications for a cheap pure-Python win:
- **Vectorising the assembly** (build `i` via `np.repeat`, concatenate `j`/`S`
  once, drop the per-atom Python loop and the `bothways` re-loop) targets the 85 %
  — the highest-leverage change that adds no dependency.
- **`query_ball_point(..., workers=-1)`** (no new dep) speeds up *only the query*:
  measured **2.3× @32 k (182→80 ms)**, **1.3× @4 k**. Because the query is only
  ~15 % of the build, this yields **<10 % end-to-end** — worth having, not a fix.

## Why isn't scipy's compiled cKDTree more competitive?
Intuition says "it's C, it should be fast." It isn't, for four compounding
reasons — measured in `probe_query.py` (cubic Ni, 32k, rc=5 Å):

| component | time | note |
|---|--:|---|
| tree construction | 3.7 ms | negligible |
| **traversal only (counts, workers=1)** | **128 ms** | the "pure compiled tree" — already ≈ matscipy's *entire* 119 ms build |
| traversal only (workers=-1) | 46 ms | traversal parallelises 2.8× |
| + Python list-of-lists materialise (workers=1) | 175 ms | +47 ms = **27 %** is building Python lists |
| full ASE build (query + assembly) | 1178 ms | the per-atom Python assembly is the other ~85 % |
| matscipy / vesin full build | 119 / 104 ms | for comparison |

1. **Wrong algorithm for the workload.** For uniform density at a fixed cutoff, a
   cell/linked list is O(N) with a tiny, cache-friendly constant (bin atoms, scan
   27 neighbouring cells with linear memory access). A KD-tree `query_ball_point`
   does an O(log N) descent **with backtracking, per query point**, chasing
   pointers through a tree — far worse cache behaviour. The *pure compiled
   traversal* (128 ms) already loses to matscipy's whole build (119 ms).
2. **ASE handles periodicity the expensive way — 14× redundant passes.** It issues
   one full N-point query per periodic-image shift. Per-shift breakdown:
   the central `(0,0,0)` shift finds 1.62 M of the 1.69 M neighbours (96 %) in
   64 ms; the **other 13 shifts each still query all 32 000 points** but together
   find only ~4 % of neighbours for ~61 ms of **almost entirely wasted tree
   descents**. A cell list ghosts the boundary once and enumerates in a single
   pass — boundary atoms cost essentially nothing.
3. **scipy's API forces Python-object materialisation.** `query_ball_point`
   returns one Python `list` per query point (an object array of lists), costing
   ~27–30 % on top of traversal. matscipy/vesin write directly into preallocated
   C arrays.
4. **ASE then wraps it in a Python per-atom assembly loop** (`for a in
   range(natoms)` + the `bothways` doubling loop) that is ~85 % of the full build.

Net: "compiled" doesn't save you when it's the *wrong algorithm* (general tree
vs cell list), *used inefficiently* (14 full passes for periodicity), behind a
*Python-object-producing API*, wrapped in a *Python post-processing loop*.
matscipy/vesin are the right algorithm, single-pass, C-array output, zero Python.

## Prototype: vectorised assembly (the cheap, no-dependency win)
The 85 % is addressable without any new dependency or algorithm change. The
prototype backend `ase-ckdtree-vec` (`backends.py`; a standalone re-implementation,
**not** an ASE source edit) keeps the *identical* cKDTree per-shift query but
replaces the per-atom Python loop and the `bothways` doubling loop with array ops
(`np.repeat` for `i`, `np.concatenate` for `j`/`S`, boolean-mask distance filter,
and `concatenate([i,j],[j,i])` / `[S,-S]` for symmetrisation). It is verified
edge-for-edge identical to every other backend by the correctness gate.

Result (cubic Ni, rc=5 Å, 1 thread; `results/prototype_comparison.csv`):

| N | ase-ckdtree (current) | ase-ckdtree-vec | speedup | matscipy | vesin |
|--:|--:|--:|--:|--:|--:|
| 4 000 | 150 ms | 41 ms | **3.6×** | 14 ms | 9.7 ms |
| 32 000 | 1221 ms | 346 ms | **3.5×** | 121 ms | 99 ms |
| 108 000 | 4202 ms | 1195 ms | **3.5×** | 408 ms | 348 ms |

A consistent **~3.5× speedup** and somewhat lower peak memory, for a low-risk,
dependency-free change to the *default* path that every ASE user gets. It cuts
the assembly portion ~6× (so the scipy query + list-materialisation, ~175 ms at
32k, now dominates the remaining time). As expected it **does not catch the
compiled backends** (still ~3× off matscipy/vesin, because reasons 1–3 above
remain) — but it materially improves the common mid-size case for free. The
v4 recommendation stands: ship this assembly fix for the default path *and* make
a compiled backend trivially pluggable via the `(i,j,d,D,S)` contract.

## Thread scaling (pinned `OMP_NUM_THREADS=1` vs unpinned)
Negligible on this machine. matscipy @1 M: 4.05 s pinned vs 3.83 s unpinned (~5 %);
vesin showed no speedup unpinned (4.73 vs 4.89 s). vesin uses its own (rayon)
threading not governed by `OMP_NUM_THREADS`; neither compiled backend exhibited
meaningful build-time parallelism at these sizes, so the pinned and unpinned
rankings are identical. Both pinned and unpinned numbers are in `results.csv`.

## Correctness
All five backends produce **identical** edge sets (sorted on `(i,j,Sx,Sy,Sz)`) on
cubic fcc Ni, a low-symmetry hcp cell, and a mixed-pbc `[T,T,F]` slab, at
cutoffs 3 and 5 Å. `D` vectors agree to ≤ 4e-15 (tol 1e-10) and each satisfies
`D = positions[j]-positions[i]+S@cell`. No reconciliation of half/full or
self-interaction conventions was needed — all four upstream APIs already emit the
full list excluding pure self-pairs.

## Artifacts
- `results/results.csv` — `backend,n_atoms,cutoff,threads,build_s_median,build_s_min,peak_mb`
- `results/results_detail.csv` — adds system, first-call time, tracemalloc peak, status
- `results/build_time_vs_N.png`, `results/build_time_vs_cutoff.png`
- `results/environment.txt` — full platform + version capture

## Harness caveat worth recording
An early version timed the build with `tracemalloc` active; that traced every
Python allocation and inflated the allocation-heavy cKDTree assembly ~16× (made
it look 18× slower than binning) while barely touching the array-based backends.
Memory tracking is now a separate untimed run. Lesson for the v4 benchmarking:
never profile memory and time in the same measurement.
