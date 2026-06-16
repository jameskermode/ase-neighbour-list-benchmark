"""Test systems for the neighbour-list benchmark.

All builders return an ``ase.Atoms`` only -- construction/replication is never
included in any timing.

Three families exercise different periodic-image logic:

* ``cubic``     -- fcc Ni in its cubic (4-atom) cell, replicated to ~N atoms.
                   This is the system used for the main size sweep.
* ``lowsym``    -- a low-symmetry (non-cubic, non-orthogonal) cell: hcp Ti,
                   replicated. Stresses the minkowski-reduce / image logic.
* ``slab``      -- an fcc(111) slab with mixed pbc=[True, True, False] and a
                   vacuum gap, to exercise the non-periodic-direction handling.
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk, fcc111


def _replication_for(n_target: int, atoms_per_cell: int) -> int:
    """Smallest cubic replication ``r`` with ``r**3 * atoms_per_cell >= n_target``."""
    r = int(round((n_target / atoms_per_cell) ** (1.0 / 3.0)))
    r = max(r, 1)
    while r ** 3 * atoms_per_cell < n_target:
        r += 1
    return r


def make_cubic(n_target: int):
    """fcc Ni cubic cell (a=3.52 A, 4 atoms) replicated to ~n_target atoms."""
    unit = bulk("Ni", "fcc", a=3.52, cubic=True)  # 4 atoms
    r = _replication_for(n_target, len(unit))
    atoms = unit * (r, r, r)
    return atoms


def make_lowsym(n_target: int):
    """Low-symmetry hcp Ti cell replicated to ~n_target atoms (non-orthogonal)."""
    unit = bulk("Ti", "hcp", a=2.95, c=4.68)  # 2 atoms, non-orthogonal cell
    r = _replication_for(n_target, len(unit))
    atoms = unit * (r, r, r)
    return atoms


def make_slab(n_target: int):
    """fcc(111) Ni slab, pbc=[True, True, False], with a vacuum gap."""
    # Pick lateral size and layer count so the total lands near n_target.
    # fcc111 with size=(L, L, layers) has L*L*layers atoms.
    layers = 6
    L = max(1, int(round((n_target / layers) ** 0.5)))
    while L * L * layers < n_target:
        L += 1
    atoms = fcc111("Ni", size=(L, L, layers), a=3.52, vacuum=10.0)
    atoms.pbc = [True, True, False]
    return atoms


BUILDERS = {
    "cubic": make_cubic,
    "lowsym": make_lowsym,
    "slab": make_slab,
}


def build(kind: str, n_target: int):
    if kind not in BUILDERS:
        raise KeyError(f"unknown system {kind!r}; known: {', '.join(BUILDERS)}")
    return BUILDERS[kind](n_target)


if __name__ == "__main__":  # quick sanity check
    for kind in BUILDERS:
        for n in (500, 4000, 32000):
            at = build(kind, n)
            print(f"{kind:7s} target={n:>7d} -> {len(at):>7d} atoms, "
                  f"pbc={list(at.pbc)}, cubic_cell={np.allclose(at.cell, np.diag(np.diag(at.cell)))}")
