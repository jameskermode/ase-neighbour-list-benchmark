"""Decompose device neighbour-list timing: ALCHEMI (JAX/Warp) vs matscipy (CuPy).

Why: the eager per-call benchmark shows ALCHEMI slower than matscipy-neighbours-gpu,
which is surprising. This separates the pieces -- host->device copy, the device
kernel itself (no host copy), and (for ALCHEMI) eager dispatch vs a jax.jit'd
steady state -- to find the mechanism.

Run one mode per process (avoids CuPy/JAX coexistence and GPU contention):
    uv run --no-sync python profile_device.py matscipy
    uv run --no-sync python profile_device.py alchemi        # float64 (x64, matches benchmark)
    uv run --no-sync python profile_device.py alchemi-f32    # float32
    uv run --no-sync python profile_device.py alchemi-jit    # jax.jit'd dense path, steady state
"""
import sys
import time

import numpy as np

from systems import build

CUTOFF = 5.0
SIZES = [4000, 32000, 108000]


def med(fn, reps=9, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3  # ms


def bench_matscipy():
    import cupy as cp
    from matscipy_neighbours import neighbour_list
    print("=== matscipy-neighbours-gpu (CuPy) ===")
    for n in SIZES:
        atoms = build("cubic", n)
        posn = atoms.get_positions(); cell = np.asarray(atoms.cell); pbc = atoms.pbc
        posd = cp.asarray(posn)

        def e2e():
            pd = cp.asarray(posn)
            i, j, d, D, S = neighbour_list("ijdDS", positions=pd, cell=cell,
                                           pbc=pbc, cutoff=CUTOFF, array_namespace=cp)
            [cp.asnumpy(x) for x in (i, j, d, D, S)]

        def dev():
            neighbour_list("ij", positions=posd, cell=cell, pbc=pbc,
                           cutoff=CUTOFF, array_namespace=cp)
            cp.cuda.runtime.deviceSynchronize()

        def h2d():
            cp.asarray(posn); cp.cuda.runtime.deviceSynchronize()

        na = len(atoms)
        print(f"N={na:>7d}: e2e={med(e2e):8.1f}  kernel(ij,no D2H)={med(dev):8.1f}  "
              f"H2D={med(h2d):6.2f} ms")


def _alchemi_common(x64):
    import jax
    if x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    jax.block_until_ready(jnp.zeros(1) + 1)        # force CUDA live before warp
    import nvalchemiops.jax.neighbors as nl
    return jax, jnp, nl


def bench_alchemi(x64=True):
    jax, jnp, nl = _alchemi_common(x64)
    tag = "float64" if x64 else "float32"
    print(f"=== ALCHEMI-gpu (JAX/Warp, {tag}) ===")
    dt = jnp.float64 if x64 else jnp.float32
    for n in SIZES:
        atoms = build("cubic", n)
        posn = atoms.get_positions()
        cj = jnp.asarray(np.asarray(atoms.cell).reshape(1, 3, 3), dtype=dt)
        pj = jnp.asarray(np.array(atoms.pbc).reshape(1, 3))
        posd = jnp.asarray(posn, dtype=dt)

        def e2e():
            out = nl.cell_list(jnp.asarray(posn, dtype=dt), CUTOFF, cell=cj,
                               pbc=pj, return_neighbor_list=True)
            nbr, ptr, sh = out
            [np.asarray(a) for a in (nbr, sh)]      # D2H of i/j + shifts

        def dev():
            out = nl.cell_list(posd, CUTOFF, cell=cj, pbc=pj,
                               return_neighbor_list=True)
            jax.block_until_ready(out)

        def h2d():
            jax.block_until_ready(jnp.asarray(posn, dtype=dt))

        na = len(atoms)
        print(f"N={na:>7d}: e2e={med(e2e):8.1f}  kernel(COO,no D2H)={med(dev):8.1f}  "
              f"H2D={med(h2d):6.2f} ms")

    # Fixed-overhead floor: same call at tiny N.
    atoms = build("cubic", 108)
    cj = jnp.asarray(np.asarray(atoms.cell).reshape(1, 3, 3), dtype=dt)
    pj = jnp.asarray(np.array(atoms.pbc).reshape(1, 3))
    posd = jnp.asarray(atoms.get_positions(), dtype=dt)

    def floor():
        out = nl.cell_list(posd, CUTOFF, cell=cj, pbc=pj, return_neighbor_list=True)
        jax.block_until_ready(out)
    print(f"N={len(atoms):>7d}: kernel(COO,no D2H)={med(floor):8.1f} ms  "
          f"(fixed per-call floor)")


def bench_alchemi_jit():
    """jax.jit'd dense (fixed-shape) path -- the compiled-loop regime the protocol
    targets. Compilation happens in warmup; timed repeats are steady state."""
    jax, jnp, nl = _alchemi_common(x64=True)
    print("=== ALCHEMI-gpu (JAX/Warp, float64, jax.jit dense steady-state) ===")
    for n in SIZES:
        atoms = build("cubic", n)
        cj = jnp.asarray(np.asarray(atoms.cell).reshape(1, 3, 3))
        pj = jnp.asarray(np.array(atoms.pbc).reshape(1, 3))
        posd = jnp.asarray(atoms.get_positions())
        K = 64

        def call(p):
            return nl.cell_list(p, CUTOFF, cell=cj, pbc=pj, max_neighbors=K,
                                return_neighbor_list=False)
        try:
            jcall = jax.jit(call)
            jax.block_until_ready(jcall(posd))      # compile

            def dev():
                jax.block_until_ready(jcall(posd))
            print(f"N={len(atoms):>7d}: jit kernel(dense,no D2H)={med(dev):8.1f} ms")
        except Exception as e:
            print(f"N={len(atoms):>7d}: jit FAILED: {type(e).__name__}: {str(e)[:80]}")


def bench_update():
    """Verlet-skin *update check* (needs_rebuild) per call: the native C++/CUDA
    kernel vs the CuPy reduction it replaced. This is the hot path of a
    device-resident MD loop (called every step; the build runs only on a
    rebuild). Both are timed device-only (no host sync, the residency regime) and
    with the eager bool() sync (the documented sync point)."""
    import cupy as cp
    from matscipy_neighbours._ase_plugin import device_neighbor_list

    be = device_neighbor_list()
    skin = 0.4
    print("=== Verlet update check: needs_rebuild per call (median ms) ===")
    print(f"{'N':>8}  {'C++ dev':>9}  {'C+++bool':>9}  {'CuPy dev':>9}  "
          f"{'CuPy+bool':>10}")
    for n in SIZES:
        atoms = build("cubic", n)
        base = atoms.get_positions()
        cell = np.asarray(atoms.cell)
        pbc = tuple(bool(b) for b in atoms.pbc)
        be.build_device(cp.asarray(base), cell, pbc, CUTOFF, "ijS")  # C++ ref
        cur = cp.asarray(base + 0.01)        # a typical sub-skin step
        ref_cp = cp.asarray(base)            # CuPy reference snapshot

        def cpp_dev():                       # native kernel, no host sync
            be.needs_rebuild(cur, skin=skin)
            cp.cuda.runtime.deviceSynchronize()

        def cpp_bool():                      # native kernel + eager sync
            bool(be.needs_rebuild(cur, skin=skin))

        def cupy_dev():                      # old CuPy reduction, no host sync
            _ = cp.max(((cur - ref_cp) ** 2).sum(axis=1)) > skin ** 2
            cp.cuda.runtime.deviceSynchronize()

        def cupy_bool():                     # old CuPy reduction + eager sync
            bool(cp.max(((cur - ref_cp) ** 2).sum(axis=1)) > skin ** 2)

        print(f"{len(atoms):>8d}  {med(cpp_dev):>9.3f}  {med(cpp_bool):>9.3f}  "
              f"{med(cupy_dev):>9.3f}  {med(cupy_bool):>10.3f}")


DENSE_K = 64  # capacity > max neighbours (54) at cutoff 5.0 fcc -> no overflow


def bench_dense_matscipy():
    """Dense fixed-capacity build via matscipy `neighbour_matrix` (AOT C++).
    Same (idx, vec, count) output and capacity as ALCHEMI's dense path -- the
    apples-to-apples counterpart for the compiled build comparison."""
    import cupy as cp
    from matscipy_neighbours import neighbour_matrix
    print(f"=== matscipy-neighbours dense build (neighbour_matrix, K={DENSE_K}) ===")
    for n in SIZES:
        atoms = build("cubic", n)
        base = atoms.get_positions(); cell = np.asarray(atoms.cell); pbc = atoms.pbc
        posd = cp.asarray(base)

        def dev():
            neighbour_matrix(positions=posd, cell=cell, pbc=pbc, cutoff=CUTOFF,
                             max_neighbours=DENSE_K, array_namespace="dlpack")
            cp.cuda.runtime.deviceSynchronize()

        def e2e():
            pd = cp.asarray(base)
            idx, dist, cnt = neighbour_matrix(
                positions=pd, cell=cell, pbc=pbc, cutoff=CUTOFF,
                max_neighbours=DENSE_K, array_namespace=cp)
            cp.asnumpy(idx); cp.asnumpy(dist); cp.asnumpy(cnt)

        print(f"N={len(atoms):>7d}: kernel(no D2H)={med(dev):8.2f}  "
              f"e2e(+D2H)={med(e2e):8.2f} ms")


def bench_dense_alchemi():
    """Dense fixed-capacity build via ALCHEMI `cell_list` (JAX, jit'd steady
    state) -- compiled, same capacity/output as matscipy's dense path."""
    jax, jnp, nl = _alchemi_common(x64=True)
    print(f"=== ALCHEMI dense build (cell_list, jit, K={DENSE_K}) ===")
    for n in SIZES:
        atoms = build("cubic", n)
        cj = jnp.asarray(np.asarray(atoms.cell).reshape(1, 3, 3))
        pj = jnp.asarray(np.array(atoms.pbc).reshape(1, 3))
        posd = jnp.asarray(atoms.get_positions())
        mtc, _, _ = nl.estimate_cell_list_sizes(posd, cj, CUTOFF, pj)

        @jax.jit
        def call(p):
            return nl.cell_list(p, CUTOFF, cell=cj, pbc=pj, max_neighbors=DENSE_K,
                                max_total_cells=int(mtc),
                                return_neighbor_list=False)
        jax.block_until_ready(call(posd))  # compile (untimed)

        def dev():
            jax.block_until_ready(call(posd))

        def e2e():
            out = call(posd)
            [np.asarray(a) for a in out]

        print(f"N={len(atoms):>7d}: kernel(no D2H)={med(dev):8.2f}  "
              f"e2e(+D2H)={med(e2e):8.2f} ms")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "matscipy"
    if mode == "matscipy":
        bench_matscipy()
    elif mode == "alchemi":
        bench_alchemi(x64=True)
    elif mode == "alchemi-f32":
        bench_alchemi(x64=False)
    elif mode == "alchemi-jit":
        bench_alchemi_jit()
    elif mode == "update":
        bench_update()
    elif mode == "matscipy-dense":
        bench_dense_matscipy()
    elif mode == "alchemi-dense":
        bench_dense_alchemi()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
