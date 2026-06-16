"""Neighbour-list backend adapters.

Every backend is wrapped in an adapter exposing a uniform interface so that the
benchmark and the correctness check consume one registry and can never drift on
what "a backend" means.

Canonical output contract (matches ASE):

    compute(atoms, cutoff) -> (i, j, d, D, S)

where, for every edge,

    D = positions[j] - positions[i] + S @ cell
    d = |D|

and the returned list is the *full* (symmetric) list: every neighbour pair
appears as both i->j and j->i, with pure self-pairs (i == j and S == 0)
excluded. This is what matscipy and vesin emit natively; the ASE backends are
configured to match (verified empirically in correctness.py).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Adapter base
# --------------------------------------------------------------------------- #
class Backend:
    """Uniform neighbour-list backend.

    Subclasses set ``name`` and implement ``_compute``. ``available`` reports
    whether the backend can run (optional backends report a reason when not).
    """

    name = "base"
    #: Human-readable note about this backend's raw half/full + self convention.
    convention = ""

    def available(self) -> tuple[bool, str]:
        """Return (is_available, reason_if_not)."""
        return True, ""

    def compute(self, atoms, cutoff: float):
        """Return canonical (i, j, d, D, S) for ``atoms`` at ``cutoff`` (Angstrom)."""
        return self._compute(atoms, cutoff)

    def _compute(self, atoms, cutoff: float):  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. ase  -- primitive_neighbor_list (cell-binning; the path most callers hit)
# --------------------------------------------------------------------------- #
class AseBackend(Backend):
    name = "ase"
    convention = "full list, self_interaction=False (native canonical)"

    def _compute(self, atoms, cutoff):
        from ase.neighborlist import primitive_neighbor_list

        i, j, d, D, S = primitive_neighbor_list(
            "ijdDS",
            atoms.pbc,
            np.asarray(atoms.cell),
            atoms.get_positions(),
            cutoff,
            self_interaction=False,
        )
        return i, j, d, D, S


# --------------------------------------------------------------------------- #
# 2. ase-newprim -- NewPrimitiveNeighborList
#
# NOTE: despite the spec's description, in released ASE 3.28.0 this class's
# .build() simply calls the cell-binning primitive_neighbor_list -- it is NOT
# the cKDTree path. It is kept as a separate backend to measure the (tiny)
# class-wrapper overhead over the bare function. The real cKDTree path is
# AseCKDTreeBackend below (PrimitiveNeighborList).
# --------------------------------------------------------------------------- #
class AseNewPrimBackend(Backend):
    name = "ase-newprim"
    convention = (
        "NewPrimitiveNeighborList -> primitive_neighbor_list (cell-binning); "
        "per-atom radius cutoff/2, skin=0, self_interaction=False, bothways=True; "
        "stores only (i, j, S) -> d, D derived from contract"
    )

    def _compute(self, atoms, cutoff):
        from ase.neighborlist import NewPrimitiveNeighborList

        nat = len(atoms)
        nl = NewPrimitiveNeighborList(
            [cutoff / 2.0] * nat,
            skin=0.0,
            sorted=False,
            self_interaction=False,
            bothways=True,
        )
        cell = np.asarray(atoms.cell)
        pos = atoms.get_positions()
        nl.build(atoms.pbc, cell, pos)
        i = np.asarray(nl.pair_first)
        j = np.asarray(nl.pair_second)
        S = np.asarray(nl.offset_vec)
        # NewPrimitiveNeighborList only stores (i, j, S); derive d, D the way an
        # ASE caller would. This is a genuine part of producing a usable list, so
        # timing it is fair across backends.
        D = pos[j] - pos[i] + S @ cell
        d = np.linalg.norm(D, axis=1)
        return i, j, d, D, S


# --------------------------------------------------------------------------- #
# 2b. ase-ckdtree -- PrimitiveNeighborList (the actual cKDTree path)
#
# This is the OLDER class and, crucially, the DEFAULT used by ase.neighborlist
# .NeighborList. Its build() uses scipy cKDTree with a per-shift
# query_ball_point loop followed by a per-atom Python list-assembly loop. It
# stores neighbours per atom in .neighbors / .displacements (S offsets); we
# assemble (i, j, d, D, S) the way a caller would.
# --------------------------------------------------------------------------- #
class AseCKDTreeBackend(Backend):
    name = "ase-ckdtree"
    convention = (
        "PrimitiveNeighborList (scipy cKDTree + per-shift query_ball_point + "
        "per-atom assembly); per-atom radius cutoff/2, skin=0, "
        "self_interaction=False, bothways=True"
    )

    def _compute(self, atoms, cutoff):
        from ase.neighborlist import PrimitiveNeighborList

        nat = len(atoms)
        nl = PrimitiveNeighborList(
            [cutoff / 2.0] * nat,
            skin=0.0,
            sorted=False,
            self_interaction=False,
            bothways=True,
        )
        cell = np.asarray(atoms.cell)
        pos = atoms.get_positions()
        nl.build(atoms.pbc, cell, pos)
        counts = [len(x) for x in nl.neighbors]
        if sum(counts) == 0:
            empty_i = np.empty(0, dtype=int)
            return (empty_i, empty_i.copy(), np.empty(0),
                    np.empty((0, 3)), np.empty((0, 3), dtype=int))
        i = np.repeat(np.arange(nat), counts)
        j = np.concatenate(nl.neighbors).astype(int)
        S = np.concatenate(nl.displacements).astype(int).reshape(-1, 3)
        D = pos[j] - pos[i] + S @ cell
        d = np.linalg.norm(D, axis=1)
        return i, j, d, D, S


# --------------------------------------------------------------------------- #
# 3. matscipy -- matscipy.neighbours.neighbour_list (optional)
# --------------------------------------------------------------------------- #
class MatscipyBackend(Backend):
    name = "matscipy"
    convention = "full list, no self-pairs (native canonical)"

    def available(self):
        try:
            import matscipy.neighbours  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            return False, f"matscipy not importable: {exc}"
        return True, ""

    def _compute(self, atoms, cutoff):
        from matscipy.neighbours import neighbour_list

        i, j, d, D, S = neighbour_list("ijdDS", atoms, cutoff)
        return i, j, d, D, S


# --------------------------------------------------------------------------- #
# 4. vesin -- vesin's ASE-compatible API (optional)
# --------------------------------------------------------------------------- #
class VesinBackend(Backend):
    name = "vesin"
    convention = "full list, no self-pairs (native canonical)"

    def available(self):
        try:
            import vesin  # noqa: F401
            from vesin import ase_neighbor_list  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            return False, f"vesin not importable: {exc}"
        return True, ""

    def _compute(self, atoms, cutoff):
        from vesin import ase_neighbor_list

        i, j, d, D, S = ase_neighbor_list("ijdDS", atoms, cutoff=cutoff)
        return i, j, d, D, S


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY = {
    b.name: b
    for b in (
        AseBackend(),
        AseNewPrimBackend(),
        AseCKDTreeBackend(),
        MatscipyBackend(),
        VesinBackend(),
    )
}

#: Canonical ordering used wherever "all backends" is iterated.
ALL_NAMES = list(_REGISTRY)


def get_backend(name: str) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(ALL_NAMES)}")
    return _REGISTRY[name]


def resolve_backends(names=None, *, require_available=False):
    """Return ``[(backend, available, reason), ...]`` for the requested names.

    ``names=None`` selects all backends in canonical order. When
    ``require_available`` is True, unavailable backends are dropped silently;
    otherwise they are returned with ``available=False`` so the caller can
    report a clean skip message.
    """
    if names is None:
        names = ALL_NAMES
    out = []
    for name in names:
        b = get_backend(name)
        ok, reason = b.available()
        if require_available and not ok:
            continue
        out.append((b, ok, reason))
    return out
