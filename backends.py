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
# 2c. ase-ckdtree-vec -- PROTOTYPE: cKDTree query + vectorised assembly
#
# Demonstrates the cheap, dependency-free win identified in FINDINGS: keep the
# exact same scipy cKDTree per-shift query as PrimitiveNeighborList, but replace
# the `for a in range(natoms)` per-atom Python assembly loop AND the bothways
# Python doubling loop with array operations (np.repeat / np.concatenate /
# boolean masks). This is NOT an ASE source edit; it is a standalone re-
# implementation using ASE's own helpers, to measure what such a patch would buy.
# Correctness.py validates it against every other backend.
# --------------------------------------------------------------------------- #
class AseCKDTreeVecBackend(Backend):
    name = "ase-ckdtree-vec"
    convention = ("prototype: cKDTree query (as PrimitiveNeighborList) + "
                  "vectorised array assembly instead of per-atom Python loops")

    def available(self):
        try:
            from ase.cell import Cell  # noqa: F401
            from ase.geometry import minkowski_reduce, wrap_positions  # noqa: F401
            from ase.neighborlist import _calc_expansion  # noqa: F401
        except Exception as exc:  # pragma: no cover
            return False, f"ASE internals unavailable for prototype: {exc}"
        return True, ""

    def _compute(self, atoms, cutoff):
        import itertools

        from scipy.spatial import cKDTree

        from ase.cell import Cell
        from ase.geometry import minkowski_reduce, wrap_positions
        from ase.neighborlist import _calc_expansion

        nat = len(atoms)
        cutoffs = np.full(nat, cutoff / 2.0)
        rcmax = float(cutoffs.max())
        pbc = np.array(atoms.pbc)
        cell = Cell(np.asarray(atoms.cell))
        positions0 = atoms.get_positions()

        # --- identical setup to PrimitiveNeighborList.build ----------------- #
        rcell, op = minkowski_reduce(cell, pbc)
        positions = wrap_positions(positions0, rcell, pbc=pbc, eps=0)
        offsets = cell.scaled_positions(positions - positions0).round().astype(int)
        tree = cKDTree(positions, copy_data=True)
        N = _calc_expansion(rcell, pbc, rcmax)
        arange = np.arange(nat)

        i_parts, j_parts, S_parts = [], [], []
        for n1, n2, n3 in itertools.product(
                range(N[0] + 1), range(-N[1], N[1] + 1), range(-N[2], N[2] + 1)):
            if n1 == 0 and (n2 < 0 or (n2 == 0 and n3 < 0)):
                continue
            displacement = np.array((n1, n2, n3)) @ rcell
            shift0 = np.array((n1, n2, n3)) @ op

            idx_lists = tree.query_ball_point(positions - displacement,
                                              r=cutoffs + rcmax)
            counts = np.fromiter((len(x) for x in idx_lists), dtype=np.intp,
                                 count=nat)
            total = int(counts.sum())
            if total == 0:
                continue

            # --- VECTORISED assembly (replaces the per-atom Python loop) ---- #
            first = np.repeat(arange, counts)              # atom a
            second = np.concatenate([np.asarray(x, dtype=np.intp)
                                     for x in idx_lists if len(x)])  # candidate b
            delta = positions[second] + displacement - positions[first]
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            mask = dist < (cutoffs[second] + cutoffs[first])
            if n1 == 0 and n2 == 0 and n3 == 0:
                mask &= second > first   # self_interaction=False, half of central cell
            first, second = first[mask], second[mask]
            i_parts.append(first)
            j_parts.append(second)
            S_parts.append(shift0 + offsets[second] - offsets[first])

        if not i_parts:
            empty_i = np.empty(0, dtype=int)
            return (empty_i, empty_i.copy(), np.empty(0),
                    np.empty((0, 3)), np.empty((0, 3), dtype=int))

        i_half = np.concatenate(i_parts)
        j_half = np.concatenate(j_parts)
        S_half = np.concatenate(S_parts)

        # --- VECTORISED bothways (replaces the Python doubling loop) -------- #
        i = np.concatenate([i_half, j_half])
        j = np.concatenate([j_half, i_half])
        S = np.concatenate([S_half, -S_half])

        cell_arr = np.asarray(atoms.cell)
        D = positions0[j] - positions0[i] + S @ cell_arr
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
# 5. matscipy-neighbours -- standalone compiled package (CPU), optional
#
# Slim extraction of matscipy.neighbours with the same public API
# (https://github.com/libAtoms/matscipy-neighbours). CPU path: pass an ASE
# Atoms object, exactly like the full-matscipy backend.
# --------------------------------------------------------------------------- #
class MatscipyNeighboursBackend(Backend):
    name = "matscipy-neighbours"
    convention = "full list, no self-pairs (native canonical)"

    def available(self):
        try:
            import matscipy_neighbours  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            return False, f"matscipy_neighbours not importable: {exc}"
        return True, ""

    def _compute(self, atoms, cutoff):
        from matscipy_neighbours import neighbour_list

        i, j, d, D, S = neighbour_list("ijdDS", atoms, cutoff)
        return i, j, d, D, S


# --------------------------------------------------------------------------- #
# 5b. matscipy-neighbours GPU -- CUDA/HIP backend via device (CuPy) positions
#
# The compiled GPU backend runs only when matscipy-neighbours was built with
# -DENABLE_CUDA=ON / -DENABLE_HIP=ON AND a GPU + CuPy are present. It is reached
# by passing *device* positions (not an ASE Atoms); results come back on-device
# and are copied to host numpy for the equivalence gate. On machines without a
# CUDA/HIP build (e.g. Apple/Metal here) available() reports False and the
# benchmark skips it -- run it on the Nvidia box per the README GPU recipe.
#
# NB: timing here is end-to-end (H2D of positions + device build + D2H of the
# result arrays), which is the honest cost of producing a host-side list.
# --------------------------------------------------------------------------- #
class MatscipyNeighboursGPUBackend(Backend):
    name = "matscipy-neighbours-gpu"
    convention = "GPU (CUDA/HIP) via CuPy device positions; results copied to host"

    def available(self):
        try:
            import cupy as cp  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            return False, f"cupy not importable: {exc}"
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                return False, "no CUDA device found"
        except Exception as exc:  # pragma: no cover - env dependent
            return False, f"no CUDA runtime: {exc}"
        # Probe that matscipy-neighbours was actually built with a GPU backend
        # by running a tiny device computation.
        try:
            from matscipy_neighbours import neighbour_list

            pos = cp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            neighbour_list(
                "i", positions=pos, cell=np.eye(3) * 10.0,
                pbc=[False, False, False], cutoff=2.0,
            )
        except Exception as exc:  # pragma: no cover - env dependent
            return False, f"matscipy-neighbours GPU backend unavailable: {exc}"
        return True, ""

    def _compute(self, atoms, cutoff):
        import cupy as cp

        from matscipy_neighbours import neighbour_list

        pos = cp.asarray(atoms.get_positions())
        i, j, d, D, S = neighbour_list(
            "ijdDS",
            positions=pos,
            cell=np.asarray(atoms.cell),
            pbc=atoms.pbc,
            numbers=atoms.get_atomic_numbers(),
            cutoff=cutoff,
        )
        # Results are device (CuPy) arrays; copy to host for the equivalence
        # gate and uniform downstream handling.
        return tuple(cp.asnumpy(x) for x in (i, j, d, D, S))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY = {
    b.name: b
    for b in (
        AseBackend(),
        AseNewPrimBackend(),
        AseCKDTreeBackend(),
        AseCKDTreeVecBackend(),
        MatscipyBackend(),
        MatscipyNeighboursBackend(),
        MatscipyNeighboursGPUBackend(),
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
