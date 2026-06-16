"""Profile ASE's cKDTree neighbour-list path (PrimitiveNeighborList).

Answers two questions for the v4 discussion:

1. Within the cKDTree path, does time go into the per-shift cKDTree
   ``query_ball_point`` calls, or into the ``for a in range(natoms)`` Python
   list-assembly loop that follows? (Determines whether a cheap pure-Python win
   exists in vectorising the assembly.)

2. Does ``cKDTree.query_ball_point(..., workers=-1)`` -- a parallel option that
   needs no new dependency -- meaningfully speed up the query portion?

No ASE source is modified: (1) uses cProfile on the real ``build``; (2)
reconstructs the exact per-shift query loop from ASE's own helpers and times it
with workers=1 vs workers=-1.

Usage::  uv run python profile_ckdtree.py [--n 4000] [--cutoff 5.0]
"""

from __future__ import annotations

import argparse
import cProfile
import itertools
import pstats
import time

import numpy as np
from scipy.spatial import cKDTree

from systems import make_cubic


def cprofile_build(atoms, cutoff):
    """cProfile PrimitiveNeighborList.build; return (total_s, query_s, stats)."""
    from ase.neighborlist import PrimitiveNeighborList

    nat = len(atoms)
    nl = PrimitiveNeighborList([cutoff / 2.0] * nat, skin=0.0,
                               self_interaction=False, bothways=True)
    pbc, cell, pos = atoms.pbc, np.asarray(atoms.cell), atoms.get_positions()

    prof = cProfile.Profile()
    prof.enable()
    nl.build(pbc, cell, pos)
    prof.disable()

    stats = pstats.Stats(prof)
    total = stats.total_tt
    query = 0.0
    for (fname), st in stats.stats.items():  # st: (cc, nc, tt, ct, callers)
        if "query_ball_point" in fname[2]:
            query += st[2]  # tottime
    return total, query, stats


def time_query_workers(atoms, cutoff):
    """Reconstruct the per-shift query loop; time it with workers=1 vs -1.

    Mirrors PrimitiveNeighborList.build (lines ~961-991): minkowski-reduce the
    cell, wrap positions, build one tree, then loop over image shifts issuing
    query_ball_point. Returns (query1_s, queryN_s, n_shifts) or None if ASE's
    helpers are unavailable.
    """
    try:
        from ase.cell import Cell
        from ase.geometry import minkowski_reduce, wrap_positions
        from ase.neighborlist import _calc_expansion
    except Exception as exc:  # pragma: no cover
        print(f"  (skipping workers probe: {exc})")
        return None

    nat = len(atoms)
    cutoffs = np.array([cutoff / 2.0] * nat)
    rcmax = cutoffs.max()
    pbc = np.array(atoms.pbc)
    cell = Cell(np.asarray(atoms.cell))
    positions0 = atoms.get_positions()

    rcell, op = minkowski_reduce(cell, pbc)
    positions = wrap_positions(positions0, rcell, pbc=pbc, eps=0)
    N = _calc_expansion(rcell, pbc, rcmax)

    shifts = [(n1, n2, n3)
              for n1, n2, n3 in itertools.product(
                  range(N[0] + 1), range(-N[1], N[1] + 1), range(-N[2], N[2] + 1))
              if not (n1 == 0 and (n2 < 0 or (n2 == 0 and n3 < 0)))]

    def run(workers):
        tree = cKDTree(positions, copy_data=True)
        t0 = time.perf_counter()
        for (n1, n2, n3) in shifts:
            displacement = (n1, n2, n3) @ rcell
            tree.query_ball_point(positions - displacement,
                                  r=cutoffs + rcmax, workers=workers)
        return time.perf_counter() - t0

    run(1)  # warmup
    q1 = min(run(1) for _ in range(3))
    qN = min(run(-1) for _ in range(3))
    return q1, qN, len(shifts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--cutoff", type=float, default=5.0)
    args = p.parse_args()

    atoms = make_cubic(args.n)
    nat = len(atoms)
    print(f"Profiling PrimitiveNeighborList (cKDTree path) on cubic Ni "
          f"N={nat}, cutoff={args.cutoff} Å\n")

    total, query, stats = cprofile_build(atoms, args.cutoff)
    assembly = total - query
    print("=== cProfile breakdown of build() ===")
    print(f"  total build (cProfile tottime sum) : {total*1e3:9.1f} ms")
    print(f"  query_ball_point (C, tottime)      : {query*1e3:9.1f} ms "
          f"({100*query/total:4.1f}%)")
    print(f"  everything else (Python assembly)  : {assembly*1e3:9.1f} ms "
          f"({100*assembly/total:4.1f}%)")
    print("\n  top 8 functions by cumulative time:")
    stats.sort_stats("tottime")
    stats.print_stats(8)

    print("=== query_ball_point workers=1 vs workers=-1 (query portion only) ===")
    res = time_query_workers(atoms, args.cutoff)
    if res:
        q1, qN, nshifts = res
        print(f"  shifts queried        : {nshifts}")
        print(f"  workers=1  query time : {q1*1e3:9.1f} ms")
        print(f"  workers=-1 query time : {qN*1e3:9.1f} ms")
        print(f"  speedup               : {q1/qN:4.2f}x")


if __name__ == "__main__":
    main()
