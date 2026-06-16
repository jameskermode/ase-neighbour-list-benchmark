# ASE neighbour-list backend benchmark

Benchmarks ASE's in-tree neighbour-list paths against compiled backends
(matscipy, vesin) on periodic systems, and proves all backends return identical
lists first. See **[FINDINGS.md](FINDINGS.md)** for results and conclusions.

## Backends (canonical `(i, j, d, D, S)`, `D = pos[j]-pos[i]+S@cell`)
| name | what it runs |
|------|--------------|
| `ase` | `primitive_neighbor_list` (cell-binning) |
| `ase-newprim` | `NewPrimitiveNeighborList` (thin wrapper over cell-binning) |
| `ase-ckdtree` | `PrimitiveNeighborList` (scipy cKDTree path; the `NeighborList` **default**) |
| `matscipy` | `matscipy.neighbours.neighbour_list` (optional) |
| `vesin` | `vesin.ase_neighbor_list` (optional) |

> Note: in released ASE 3.28.0 the cKDTree path is `PrimitiveNeighborList`, not
> `NewPrimitiveNeighborList` — the opposite of what the original brief assumed.
> See FINDINGS.md.

## Setup
```sh
uv sync          # installs ase, matscipy, vesin, scipy, matplotlib, pytest (pinned in uv.lock)
```

## Use
```sh
# Correctness gate (also a pytest test)
uv run python correctness.py
uv run pytest correctness.py

# Benchmark (gates on correctness first)
uv run python benchmark.py --sizes 500,4000,32000 --cutoff 5.0 \
    --backends ase,ase-newprim,ase-ckdtree,matscipy,vesin --threads 1 --out results/

# Cutoff sweep at a fixed size, append to existing results
uv run python benchmark.py --append --no-size-sweep --sweep-size 32000 \
    --cutoff-sweep 3.0,8.0 --threads 1 --out results/

# Plots
uv run python plots.py --results results/

# Profile the cKDTree path (query vs assembly; workers=-1 probe)
uv run python profile_ckdtree.py --n 32000 --cutoff 5.0
```

### Key flags
`--sizes` target atom counts · `--cutoff` Å · `--cutoff-sweep` cutoffs at
`--sweep-size` · `--systems` `cubic,lowsym,slab` · `--backends` subset ·
`--threads` `1` pins `OMP_NUM_THREADS`, `0` = unpinned (comma list for both) ·
`--timeout` per-run seconds · `--append` / `--no-size-sweep` for staged runs.

## Files
`backends.py` adapter registry · `systems.py` test systems · `correctness.py`
equivalence gate · `benchmark.py` harness (subprocess-per-run: timeout + clean
peak-RSS + thread pinning) · `profile_ckdtree.py` · `plots.py` · `envinfo.py`.

Optional backends skip with a clear message if not installed; the benchmark
never runs timing unless the correctness gate passes.
