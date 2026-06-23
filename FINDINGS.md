# Findings: ASE neighbour-list backends vs matscipy & vesin

**For the ASE v4 Atoms/Calculator interface discussion.** All backends were first
proven to return *identical* neighbour lists (see Correctness) before any timing.

## Update — `matscipy-neighbours` added (refreshed run)

Added the standalone **`matscipy-neighbours`** package
(https://github.com/libAtoms/matscipy-neighbours) — a slim, separately-packaged
extraction of `matscipy.neighbours` with the same `(i, j, d, D, S)` API and an
optional CUDA/HIP GPU backend. This section reports a fresh run on the machine
below; the **original ase/matscipy/vesin analysis sections further down are
retained for context** (their larger-N 256k/1M, thread-scaling, and prototype
figures are from the earlier extended run and were *not* re-measured this round).

**Build time, cubic fcc Ni, cutoff 5.0 Å, 1 thread (median ms).** `ase-newprim`
omitted (≈ `ase`).

| N | ase (binning) | ase-ckdtree (default) | ase-ckdtree-vec | matscipy | **matscipy-neighbours** | vesin |
|--:|--:|--:|--:|--:|--:|--:|
| 500 | 28.5 | 18.9 | 4.8 | 2.64 | **0.86** | 1.30 |
| 4 000 | 122.9 | 146.9 | 39.8 | 13.3 | **4.61** | 9.53 |
| 32 000 | 1217 | 1161 | 344 | 113 | **33.8** | 95.0 |
| 108 000 | 6316 | 4100 | 1216 | 405 | **113** | 348 |

Peak RSS @108 k (MB, incl. ~80 MB baseline): ase 3856 · ase-ckdtree 2134 ·
matscipy 1167 · **matscipy-neighbours 669** · vesin 1199 — `matscipy-neighbours`
is both the fastest and the leanest.

**Cutoff sweep @ 32 k (median ms; peak GB for the rc=8 binning blow-up).**

| rc (Å) | ase | ase-ckdtree | matscipy | **matscipy-neighbours** | vesin |
|--:|--:|--:|--:|--:|--:|
| 3 | 309 | 590 | 31.0 | **12.3** | 23.7 |
| 5 | 1205 | 1196 | 121 | **35.1** | 99.6 |
| 8 | 10427 (6.0 GB) | 3313 (2.4 GB) | 621 | **156** | 384 |

**Takeaways:**
- **`matscipy-neighbours` is the fastest backend at every size and cutoff** —
  ~**3× faster than full `matscipy`**, ~2.5–3× faster than `vesin`, and **~55×
  faster than ASE's default at 108 k** (113 ms vs 4.1 s), with the lowest peak
  memory. (It beating full matscipy is consistent across runs — the standalone
  kernel appears newer/faster.) It matches the `ase` reference edge sets exactly
  (`D` diff ~1e-15).
- This *strengthens* the v4 conclusion below: keep the pluggable
  `(i, j, d, D, S)` contract so a fast compiled backend drops in without becoming
  a hard dependency.
- **GPU:** `matscipy-neighbours` has a **CUDA/HIP** GPU backend (no Metal/MPS),
  reached via device (CuPy) positions, not the ASE `Atoms` path. The benchmark
  carries a `matscipy-neighbours-gpu` backend that auto-skips without a CUDA/HIP
  build + GPU. It was skipped on the Apple/Metal machine above; it has now been
  run for real on an NVIDIA box — see **[GPU run](#gpu-run-nvidia-rtx-a4500-cuda-126)** below.

## GPU run (NVIDIA RTX A4500, CUDA 12.6)

First real run of the `matscipy-neighbours` **CUDA** backend (the Apple/Metal dev
machine above can't build it — CUDA/HIP only, no Metal/MPS). The
`matscipy-neighbours-gpu` benchmark backend passes *device* (CuPy) positions to
the compiled GPU kernel and copies the result back to host; reported timing is
therefore **end-to-end** (host→device positions + device build + device→host copy
of the `(i,j,d,D,S)` arrays).

**Two more device backends now run in the same comparison via ASE's experimental
*device neighbour-list protocol*** (see
[`DESIGN-device-neighbourlist.md`](DESIGN-device-neighbourlist.md)) — reached through
`build_device(...)` rather than a direct library call:
- **`ALCHEMI-gpu`** (NVIDIA ALCHEMI / `nvalchemiops`) — JAX/Warp O(N) cell list,
  device-resident JAX arrays;
- **`vesin-gpu`** (Vesin / metatensor-metatomic) — Vesin's CUDA cell list via its
  CuPy interface, device-resident CuPy arrays.

So the device rows below are **three independent implementations — author
(matscipy-neighbours), hardware vendor (ALCHEMI), and ecosystem (Vesin) — behind one
ASE protocol**, all timed end-to-end. The correctness gate confirmed all three agree
edge-for-edge with the `ase` reference (and each other) before timing — the spec's
headline author/vendor/ecosystem validation.

**GPU environment**
- GPU: **NVIDIA RTX A4500**, compute capability **8.6** (`CMAKE_CUDA_ARCHITECTURES=86`), driver 610.43.02
- `matscipy-neighbours-gpu`: CUDA toolkit **12.6.0**; CuPy **cupy-cuda12x 14.1.1**; host compiler gcc 11.5
- `ALCHEMI-gpu`: **nvalchemi-toolkit-ops 0.3.1** (JAX **0.10.2** + **warp-lang 1.14.0**), float64.
- `vesin-gpu`: **vesin 0.5.8** (the installed wheel ships a CUDA backend reachable via its CuPy interface; `dlopen`s `libcudart`).
- All three GPU backends + CuPy + JAX share one process **without a system `module load CUDA`** by putting the venv `nvidia-cuda-runtime` wheel (libcudart 12.9, same as JAX's) on `LD_LIBRARY_PATH`.
- CPU figures below from the **same run** for context: Xeon Silver 4216, 1 thread (`OMP_NUM_THREADS=1`)

**Correctness — all green.** The benchmark's correctness gate copies the device
results to host and asserts edge sets **identical to the `ase` reference** before
timing: **passed at every size, no GPU-vs-CPU discrepancies.** The package's own
suite also passed: **`ctest` 31/31** (incl. all `NeighbourListGpu.*` match-CPU
tests, `MemorySpace.Device*`, device scan/radix-sort) and **`test_dlpack.py` 14
passed** (device DLPack round-trip; the JAX-namespace case now runs too).

**Build time:** GPU build ≈ **18–24 s** (`-DENABLE_CUDA=ON`, nvcc) vs **≈ 5 s** for
the CPU wheel.

**Build time, cubic fcc Ni, cutoff 5.0 Å, 1 thread (median ms), one self-consistent run.**
`mn-gpu` calls `neighbour_list` directly; `mn-device` routes the same kernel through
ASE's `DeviceNeighborList.build_device` protocol — they match to within noise, so the
protocol path adds no measurable overhead. `vesin-gpu` is Vesin's CUDA cell list via
its CuPy interface (same protocol).

| N | ase | matscipy | mn (CPU) | vesin | **mn-gpu** | **mn-device** | **vesin-gpu** | **ALCHEMI-gpu** |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 4 000 | 396 | 45.8 | 27.0 | 41.6 | **8.7** | **9.0** | **12.7** | 100.2 |
| 32 000 | 4157 | 367.2 | 220.5 | 387.6 | **53.8** | **54.2** | **80.5** | 204.8 |
| 108 000 | 14910 | 1199.9 | 731.8 | 1352.1 | **186.3** | **185.7** | **261.4** | 618.2 |

**Takeaways:**
- `matscipy-neighbours-gpu` is **~3–4× faster than the fastest CPU backend** (CPU
  `matscipy-neighbours`) and the lead **grows with N**, *even including* H2D/D2H
  transfer. Peak host RSS stays low (~0.6 GB @108 k vs 5.5 GB for ASE).
- **`ALCHEMI-gpu` is slower here than `mn-gpu` only because this benchmark times
  each build as an isolated, eager call — the wrong regime for ALCHEMI.** Profiling
  (see [mechanism analysis](#why-is-alchemi-gpu-slower-than-mn-gpu-here) below)
  shows a fixed **~86 ms/call** JAX+Warp dispatch + host-side cell-grid-sizing
  floor that is *N-independent* (same at 108 atoms). Under `jax.jit` — the
  compiled, device-resident regime the protocol exists for — that floor vanishes
  and **ALCHEMI's kernel is 0.5–3.6 ms, ~2.5× *faster* than matscipy's at 108 k.**
  So do **not** read this table as "matscipy beats NVIDIA": NVIDIA's kernel is
  excellent; the eager harness just measured launch overhead a real MLIP never
  pays per step. The load-bearing result is that **three** independent device
  backends agree edge-for-edge through one ASE protocol; the eager *ranking* is a
  harness artefact.
- **`vesin-gpu`** (Vesin's CUDA cell list via its CuPy interface) is a solid GPU
  performer — ~1.4× `mn-gpu` and well ahead of the CPU backends (12.7 / 80.5 /
  261 ms), eager and end-to-end. It is the **ecosystem** member of the
  author/vendor/ecosystem trio, reached with no new install (the installed vesin
  0.5.8 wheel already ships a CUDA backend). `differentiable=False` only because
  the CuPy path is not an autograd framework (not a vesin limitation; `vesin-torch`
  is GPU-capable **and** autograd-differentiable, so it is a candidate
  `differentiable=True` device backend); COO output only (no dense `max_capacity`
  path).
- This is an off-ASE-path option (device positions, not `Atoms`); it reinforces
  the v4 message that the `(i,j,d,D,S)` contract should admit a device/compiled
  backend without it becoming a hard dependency.
- **Workflow gotcha:** `uv run`/`uv sync` rebuild `matscipy-neighbours` from
  source *without* the CUDA flags, silently clobbering the GPU wheel back to
  CPU-only (the GPU backend then self-skips). Build the GPU wheel with
  `uv pip install --reinstall -C cmake.define.ENABLE_CUDA=ON
  -C cmake.define.CMAKE_CUDA_ARCHITECTURES=<arch> ../matscipy-neighbours`, then
  run the benchmark with **`uv run --no-sync`**.

### Why is ALCHEMI-gpu slower than mn-gpu here?

A solo-developer library appearing to beat NVIDIA's by 3–10× is a red flag that
the *harness*, not the kernel, is being measured. It is. A decomposition
(`profile_device.py`, same A4500, cubic, cutoff 5.0, 1 thread, median ms)
separates host→device copy, the device kernel, and eager-vs-`jax.jit` dispatch:

| measurement | N=4 000 | N=32 000 | N=108 000 |
|--|--:|--:|--:|
| matscipy end-to-end (build + d,D,S + D2H) | 8.8 | 55.9 | 197.2 |
| matscipy kernel only (i,j; no D2H) | 1.4 | 3.6 | 9.2 |
| ALCHEMI **eager** end-to-end (COO) | 86.7 | 199.0 | 518.4 |
| ALCHEMI **eager** kernel (COO; no D2H) | 88.0 | 172.9 | 450.9 |
| ALCHEMI eager floor **@108 atoms** | 86.8 | 86.8 | 86.8 |
| ALCHEMI eager kernel, **float32** | 85.7 | 156.5 | 442.8 |
| **ALCHEMI `jax.jit` kernel (dense; no D2H)** | **0.5** | **1.7** | **3.6** |
| host→device copy (either backend) | <0.2 | <0.3 | <0.6 |

Three facts pin the mechanism:

1. **A fixed ~86 ms per-call floor, independent of N** — identical at 108 atoms and
   4 000 atoms. This is JAX trace/dispatch + the Warp FFI launch + ALCHEMI's
   *host-side, data-dependent cell-grid sizing on every call* (the same dynamic
   sizing that makes a naïve `jax.jit` of `cell_list` fail until the grid size is
   pinned). It is pure per-call overhead, not kernel work.
2. **float32 ≈ float64** (e.g. 443 vs 451 ms @108 k) — so it is *not* an FP64
   penalty on the consumer A4500; the neighbour search is integer/index-bound.
   H2D is negligible (<0.6 ms) for both.
3. **Under `jax.jit` with the grid size pinned (the compiled, fixed-shape regime
   the device protocol exists for), the floor vanishes and ALCHEMI's kernel is
   0.5–3.6 ms — *faster* than matscipy's kernel (1.4–9.2 ms), ~2.5× at 108 k.**

So **NVIDIA's kernel is excellent**; the eager per-call benchmark simply measured
JAX/Warp launch overhead that a real device-resident MLIP never pays per step. The
overhead is amortised to zero inside a `jax.jit` / CUDA-graph MD loop — which is
*precisely* the residency the device capability targets (build rarely via
`needs_rebuild`, keep everything on-device). matscipy-neighbours wins the *eager,
host-handoff* contest because its CuPy path has almost no Python per-call cost and
its kernel is genuinely fast; ALCHEMI wins the *compiled, device-resident* contest
it was designed for. The headline table above is the former regime; treat its
ALCHEMI column as a launch-overhead measurement, not a kernel-throughput verdict.

(The `jax.jit` row above uses ALCHEMI's dense output vs matscipy's variable-length
COO kernel, so it is not strictly like-for-like. The fully apples-to-apples
compiled build comparison — both dense, same capacity, same output — is in the
[compiled head-to-head](#compiled-head-to-head-alchemi-vs-matscipy-the-fair-comparison)
below.)

### Verlet update check: build vs reuse, and the CuPy → C++/CUDA migration

A device-resident MD loop rebuilds the neighbour list only rarely; **every step it
runs the cheap Verlet update check** (`needs_rebuild`: has any atom moved more than
the skin?). The benchmark times this separately from the build —
`update_results.csv` / `update_time_vs_N.png` — for the device-protocol backends
(`matscipy-neighbours-device`, `alchemi-gpu`); host backends have no device update
check. The whole point of skin reuse is that the per-step check is orders of
magnitude cheaper than a rebuild.

This check originally ran in CuPy; it is now a **native C++/CUDA kernel** (a CUB/
hipCUB transform-then-`DeviceReduce::Max` over per-atom squared displacement, then
an on-device threshold against `skin²`, returning a 1-element `uint8` device
scalar — no host sync, no CuPy). Per `needs_rebuild` call, cubic fcc Ni, A4500
(median ms, `profile_device.py update`):

| N | **C++ (device)** | C++ + `bool()` | CuPy (device) | CuPy + `bool()` |
|--:|--:|--:|--:|--:|
| 4 000 | **0.043** | 0.047 | 0.124 | 0.142 |
| 32 000 | **0.040** | 0.047 | 0.437 | 0.450 |
| 108 000 | **0.042** | 0.049 | 1.341 | 1.354 |

- **The C++ check is ~3× faster at 4 k and ~32× at 108 k, and is N-independent
  (~0.04 ms).** It is one fused transform+reduce kernel: no O(N) temporaries, the
  per-atom work is trivial, so the time is a flat launch/alloc floor. CuPy instead
  runs 3–4 separate kernels (`(cur-ref)`, `**2`, `.sum(1)`, `.max()`), each sweeping
  an O(N) temporary, so it scales with N.
- The eager `bool()` sync adds only ~5 µs (a 1-byte device→host read), so the full
  eager per-step check is ~0.047 ms; a compiled consumer pays even that in-graph.
- Beyond speed, the migration removes CuPy from the update path entirely, so a
  torch-/JAX-only consumer can drive the skin loop without it.
- **Fair, compiled-vs-compiled methodology.** matscipy's C++ check is
  ahead-of-time compiled, so the benchmark times ALCHEMI's `needs_rebuild`
  **`jax.jit`'d at steady state** (how a device-resident loop runs it), *not*
  eagerly — an eager call would measure JAX/Warp tracing + dispatch the C++ path
  never pays. The difference is stark: ~**13.5 ms eager → ~0.26 ms jit'd** for the
  same op (~50×, N-independent; `needs_rebuild` has static shapes so it jits with
  no grid pinning, unlike `cell_list`). The residual gap to matscipy (~0.045 ms) is
  the Warp-FFI-from-XLA launch vs a direct C-extension CUDA call — same order.

**In-harness update vs rebuild** (`benchmark.py` update phase, `update_results.csv`;
median per call, with the `bool()` decision; ALCHEMI **jit-compiled** for parity with
the AOT-compiled C++ kernel). The point of the separate phase is the
**build-vs-reuse ratio**: a Verlet step that reuses the list is hundreds-to-thousands
of times cheaper than rebuilding it. The three device backends use three different
update implementations:

| N | rebuild (`mn-device`) | update `mn-device` (C++) | update `vesin-gpu` (CuPy) | update `alchemi-gpu` (jit) | reuse speed-up |
|--:|--:|--:|--:|--:|--:|
| 4 000 | 9.0 ms | 48 µs | 157 µs | 269 µs | ~190× |
| 32 000 | 54.2 ms | 48 µs | 421 µs | 273 µs | ~1130× |
| 108 000 | 185.7 ms | 149 µs | 1223 µs | 270 µs | ~1250× |

So a no-rebuild step costs tens-to-hundreds of microseconds — skin reuse turns the
per-step neighbour cost from a full rebuild into essentially free. The three update
implementations also illustrate the design space cleanly: matscipy's **native fused
CUDA kernel** is fastest and near-flat in N; ALCHEMI's **`jax.jit`'d** op is flat
(~0.27 ms, the Warp-FFI-from-XLA launch); Vesin's **CuPy reduction** (the multi-kernel
`((cur-ref)**2).sum(1).max()` approach matscipy was migrated *away* from) is cheapest at
small N but **scales with N** (multiple O(N) temporaries), so it's slowest at 108 k.
See `update_time_vs_N.png`.

### Compiled head-to-head: ALCHEMI vs matscipy (the fair comparison)

Both backends in their **compiled** form — matscipy AOT C++, ALCHEMI `jax.jit` at
steady state — same **dense** output and capacity (`K=64`), cubic fcc Ni, cutoff
5.0 Å, A4500 (`profile_device.py matscipy-dense` / `alchemi-dense`).

**Dense build (median ms):**

| N | matscipy kernel | ALCHEMI kernel | ALCHEMI faster | matscipy e2e (+D2H) | ALCHEMI e2e (+D2H) |
|--:|--:|--:|--:|--:|--:|
| 4 000 | 1.91 | **0.62** | 3.1× | 2.86 | 1.40 |
| 32 000 | 6.77 | **1.76** | 3.8× | 35.5 | 8.75 |
| 108 000 | 18.62 | **3.52** | 5.3× | 114.8 | 60.9 |

**Update check (median µs/call):** matscipy 49 / 45 / 154 vs ALCHEMI 264 / 265 / 275
→ **matscipy 2–6× faster**.

So once both are compiled the picture inverts from the eager benchmark and **the two
split**:

- **Build (the occasional rebuild): ALCHEMI's Warp cell-list kernel is 3–5× faster,
  and pulls *further* ahead with N** (3.1× → 5.3×) — NVIDIA's kernel scales better.
- **Update check (every step): matscipy is 2–6× faster**, because a direct
  C-extension CUDA call carries no XLA/Warp-FFI launch layer.

Net for a device-resident MD loop: rebuilds (rare) favour ALCHEMI; the per-step skin
check (constant) favours matscipy — and both are far cheaper than an eager,
un-jitted call, which is the regime the plain `benchmark.py` table measures.

## Environment
- Platform: macOS-26.5.1-arm64 (Apple M3 Pro, 12 logical cores)
- Python 3.12.5; **ase 3.28.0, matscipy 1.2.0, matscipy-neighbours 0.1.0, vesin 0.5.8**, scipy 1.17.1, numpy 2.4.6
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
All available backends (the ASE variants, `matscipy`, `matscipy-neighbours`,
`vesin`) produce **identical** edge sets (sorted on `(i,j,Sx,Sy,Sz)`) on
cubic fcc Ni, a low-symmetry hcp cell, and a mixed-pbc `[T,T,F]` slab, at
cutoffs 3 and 5 Å. `D` vectors agree to ≤ 4e-15 (tol 1e-10) and each satisfies
`D = positions[j]-positions[i]+S@cell`. No reconciliation of half/full or
self-interaction conventions was needed — all four upstream APIs already emit the
full list excluding pure self-pairs.

## Artifacts
- `results/results.csv` — `backend,n_atoms,cutoff,threads,build_s_median,build_s_min,peak_mb`
- `results/results_detail.csv` — adds system, first-call time, tracemalloc peak, status
- `results/update_results.csv` / `results/update_results_detail.csv` — the per-step
  Verlet update-check (`needs_rebuild`) timings, `update_s_median`/`update_s_min`
  (device-protocol backends only)
- `results/build_time_vs_N.png`, `results/build_time_vs_cutoff.png`,
  `results/update_time_vs_N.png`
- `results/environment.txt` — full platform + version capture

## Harness caveat worth recording
An early version timed the build with `tracemalloc` active; that traced every
Python allocation and inflated the allocation-heavy cKDTree assembly ~16× (made
it look 18× slower than binning) while barely touching the array-based backends.
Memory tracking is now a separate untimed run. Lesson for the v4 benchmarking:
never profile memory and time in the same measurement.
