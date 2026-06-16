"""Decompose the scipy cKDTree neighbour-list cost.

Goal: explain why scipy's compiled cKDTree is *not* competitive with the
compiled cell-list backends (matscipy, vesin), despite all being "C code".

We split the cKDTree path into:
  1. tree construction
  2. the per-image-shift query traversal, two ways:
       a. return_length=True  -> counts only (pure compiled traversal, no Python
          objects materialised)
       b. default             -> list-of-lists (forces one Python list per query
          point, i.e. the materialisation cost)
  3. (for reference) the full ASE PrimitiveNeighborList build and the matscipy /
     vesin total build.

The gap between (2a) and (2b) is the Python-object materialisation overhead that
scipy's API forces on this workload; (2a) is the fair "compiled tree traversal"
number to compare against a cell list.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
from scipy.spatial import cKDTree

from ase.cell import Cell
from ase.geometry import minkowski_reduce, wrap_positions
from ase.neighborlist import _calc_expansion
from systems import make_cubic


def setup(n, cutoff):
    at = make_cubic(n)
    nat = len(at)
    cutoffs = np.full(nat, cutoff / 2.0)
    rcmax = float(cutoffs.max())
    pbc = np.array(at.pbc)
    cell = Cell(np.asarray(at.cell))
    pos0 = at.get_positions()
    rcell, op = minkowski_reduce(cell, pbc)
    positions = wrap_positions(pos0, rcell, pbc=pbc, eps=0)
    N = _calc_expansion(rcell, pbc, rcmax)
    shifts = [(n1, n2, n3) for n1, n2, n3 in itertools.product(
        range(N[0] + 1), range(-N[1], N[1] + 1), range(-N[2], N[2] + 1))
        if not (n1 == 0 and (n2 < 0 or (n2 == 0 and n3 < 0)))]
    return at, positions, rcell, cutoffs, rcmax, shifts


def best(fn, k=3):
    fn()  # warmup
    return min(_timed(fn) for _ in range(k))


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32000)
    p.add_argument("--cutoff", type=float, default=5.0)
    args = p.parse_args()

    at, positions, rcell, cutoffs, rcmax, shifts = setup(args.n, args.cutoff)
    nat = len(positions)
    r = cutoffs + rcmax
    print(f"cubic Ni N={nat}, cutoff={args.cutoff} Å, "
          f"{len(shifts)} image shifts queried, query radius={r[0]:.2f} Å\n")

    # 1. tree build
    t_build = best(lambda: cKDTree(positions, copy_data=True))

    tree = cKDTree(positions, copy_data=True)

    # how many neighbour entries are produced across all shifts (the work done)
    total_found = 0
    for s in shifts:
        disp = np.asarray(s) @ rcell
        lengths = tree.query_ball_point(positions - disp, r=r, return_length=True)
        total_found += int(lengths.sum())

    # 2a. traversal only (counts), workers=1 and workers=-1
    def q_len(workers):
        for s in shifts:
            disp = np.asarray(s) @ rcell
            tree.query_ball_point(positions - disp, r=r,
                                  return_length=True, workers=workers)
    t_len1 = best(lambda: q_len(1))
    t_lenN = best(lambda: q_len(-1))

    # 2b. list-of-lists materialisation, workers=1 and workers=-1
    def q_list(workers):
        for s in shifts:
            disp = np.asarray(s) @ rcell
            tree.query_ball_point(positions - disp, r=r, workers=workers)
    t_list1 = best(lambda: q_list(1))
    t_listN = best(lambda: q_list(-1))

    # 3. references
    from backends import get_backend
    def full_build(name):
        b = get_backend(name)
        ok, _ = b.available()
        if not ok:
            return None
        b.compute(at, args.cutoff)  # warmup
        return min(_timed(lambda: b.compute(at, args.cutoff)) for _ in range(3))

    t_ckd_full = full_build("ase-ckdtree")
    t_mat = full_build("matscipy")
    t_ves = full_build("vesin")

    ms = lambda x: "   n/a" if x is None else f"{x*1e3:7.1f}"
    print("=== cKDTree cost decomposition (ms, best of 3) ===")
    print(f"  tree construction                         : {ms(t_build)}")
    print(f"  traversal only, counts  (workers=1)       : {ms(t_len1)}   <- 'pure compiled tree'")
    print(f"  traversal only, counts  (workers=-1)      : {ms(t_lenN)}")
    print(f"  + list-of-lists materialise (workers=1)   : {ms(t_list1)}")
    print(f"  + list-of-lists materialise (workers=-1)  : {ms(t_listN)}")
    print(f"  python-list materialisation overhead      : {ms(t_list1 - t_len1)} "
          f"({100*(t_list1-t_len1)/t_list1:.0f}% of the list query)")
    print(f"  neighbour entries produced (all shifts)   : {total_found:,}")
    print()
    print("=== full neighbour-list build, same system (ms, best of 3) ===")
    print(f"  ase-ckdtree (tree build + assembly)       : {ms(t_ckd_full)}")
    print(f"  matscipy (compiled cell list)             : {ms(t_mat)}")
    print(f"  vesin    (compiled cell list)             : {ms(t_ves)}")
    print()
    if t_mat:
        print(f"  scipy traversal-only / matscipy total     : {t_len1/t_mat:4.1f}x")
        print(f"  scipy traversal-only / vesin total        : {t_len1/t_ves:4.1f}x")


if __name__ == "__main__":
    main()
