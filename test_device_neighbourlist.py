"""Tests for the experimental device-resident neighbour-list capability.

Exercises the SPEC-device-neighbourlist.md contract against the matscipy-neighbours
device backend (the load-bearing proof): cross-backend equivalence vs the host
oracle, quantity-order independence, Verlet-skin correctness with on-device reuse,
device residency (no host transfer on reused steps), device tagging, and
fixed-capacity padding + overflow.

Auto-skips when CuPy / a CUDA device / a GPU build of matscipy-neighbours / ASE's
experimental device protocol module is unavailable.  Run with::

    uv run --no-sync pytest test_device_neighbourlist.py
"""
import numpy as np
import pytest

from correctness import AGREE_TOL, canonicalise
from systems import build


def _skip_reason():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - env dependent
        return f"cupy not importable: {exc}"
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return "no CUDA device"
    except Exception as exc:  # pragma: no cover - env dependent
        return f"no CUDA runtime: {exc}"
    try:
        from ase._4.plugins.neighborlist_device import DeviceNeighborList
        from matscipy_neighbours._ase_plugin import device_neighbor_list
        be = device_neighbor_list()
        if not isinstance(be, DeviceNeighborList):
            return "matscipy device backend is not a DeviceNeighborList"
    except Exception as exc:  # pragma: no cover - env dependent
        return f"device capability unavailable: {exc}"
    return None


_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=str(_REASON))

CUTOFF = 2.5
SYSTEMS = [("cubic", 256), ("lowsym", 256), ("slab", 200)]


def _backend():
    from matscipy_neighbours._ase_plugin import device_neighbor_list
    return device_neighbor_list()


def _host_oracle(atoms, cutoff, quantities="ijdDS"):
    from matscipy_neighbours import neighbour_list
    return neighbour_list(quantities, atoms, cutoff)


def _device_to_host(res, quantities):
    import cupy as cp
    return {q: cp.asnumpy(res.get(q)) for q in quantities}


# --------------------------------------------------------------------------- #
# 1. Cross-backend equivalence (the heart of the validation)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,n", SYSTEMS)
def test_equivalence_vs_host_oracle(kind, n):
    import cupy as cp

    atoms = build(kind, n)
    be = _backend()
    pos = cp.asarray(atoms.get_positions())
    res = be.build_device(pos, np.asarray(atoms.cell),
                          tuple(bool(b) for b in atoms.pbc), CUTOFF, "ijSD")
    h = _device_to_host(res, "ijSD")
    d_dev = np.linalg.norm(h["D"], axis=1)
    cdev = canonicalise(h["i"], h["j"], d_dev, h["D"], h["S"])

    hi, hj, hd, hD, hS = _host_oracle(atoms, CUTOFF)
    chost = canonicalise(hi, hj, hd, hD, hS)

    # integer-exact endpoint+shift set, float-tolerant vectors.
    assert np.array_equal(cdev["keys"], chost["keys"]), \
        f"{kind}: device edge set differs from host oracle"
    if len(cdev["i"]):
        assert np.abs(cdev["D"] - chost["D"]).max() <= AGREE_TOL


# --------------------------------------------------------------------------- #
# 2. Quantity-order independence (the §2.3 silent-corruption guard)
# --------------------------------------------------------------------------- #
def test_quantity_order_independence():
    import cupy as cp

    atoms = build("lowsym", 256)
    be = _backend()
    pos = cp.asarray(atoms.get_positions())
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)

    a = be.build_device(pos, cell, pbc, CUTOFF, "ijSD")
    b = be.build_device(pos, cell, pbc, CUTOFF, "SDji")  # different request order

    for q in "ijSD":
        assert np.array_equal(cp.asnumpy(a.get(q)), cp.asnumpy(b.get(q))), \
            f"get({q!r}) depends on request order"


# --------------------------------------------------------------------------- #
# 3. Skin correctness (no missed edges on reuse; small skin still rebuilds)
# --------------------------------------------------------------------------- #
def _true_edge_keys(i, j, S, D, cutoff):
    """Canonical (i,j,S) keys of edges within the true cutoff."""
    d = np.linalg.norm(D, axis=1)
    keep = d <= cutoff
    return canonicalise(i[keep], j[keep], d[keep], D[keep], S[keep])["keys"]


def test_skin_reuse_misses_no_edges():
    import cupy as cp
    from ase.neighborlist import NeighborListBuilder

    rng = np.random.RandomState(7)
    atoms = build("cubic", 256)
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    skin = 0.4
    be = _backend()
    builder = NeighborListBuilder(CUTOFF, skin=skin, neighbor_list=be,
                                  quantities="ijSD")

    base = atoms.get_positions()
    pos = cp.asarray(base)
    assert builder.update_device(pos, cell, pbc) is True

    # Take steps each below the skin so the list is reused; on every step the
    # cached (padded) pair set, filtered to the true cutoff at the *current*
    # positions, must equal a fresh build -- i.e. no true edge is missed.
    cur = base.copy()
    reused_at_least_once = False
    for _ in range(6):
        step = rng.uniform(-0.15, 0.15, size=cur.shape)  # max |move| < skin
        cur = cur + step
        pos = cp.asarray(cur)
        rebuilt = builder.update_device(pos, cell, pbc)
        reused_at_least_once |= (not rebuilt)

        r = builder.device_result()
        ci = cp.asnumpy(r.get("i")); cj = cp.asnumpy(r.get("j"))
        cS = cp.asnumpy(r.get("S"))
        D_now = cur[cj] - cur[ci] + cS @ cell             # exact D at current pos
        cached_keys = _true_edge_keys(ci, cj, cS, D_now, CUTOFF)

        fi, fj, fd, fD, fS = _host_oracle(atoms.__class__(
            numbers=atoms.numbers, positions=cur, cell=cell, pbc=pbc), CUTOFF)
        fresh_keys = canonicalise(fi, fj, fd, fD, fS)["keys"]

        assert np.array_equal(cached_keys, fresh_keys), \
            "skin reuse missed (or invented) an edge within the true cutoff"

    assert reused_at_least_once, "expected at least one reused (no-rebuild) step"


def test_too_small_skin_still_rebuilds():
    import cupy as cp
    from ase.neighborlist import NeighborListBuilder

    atoms = build("cubic", 256)
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    be = _backend()
    builder = NeighborListBuilder(CUTOFF, skin=0.05, neighbor_list=be,
                                  quantities="ijSD")
    base = atoms.get_positions()
    assert builder.update_device(cp.asarray(base), cell, pbc) is True
    moved = base.copy(); moved[0] += 0.5                  # >> skin
    assert builder.update_device(cp.asarray(moved), cell, pbc) is True, \
        "a displacement beyond the skin must force a rebuild"


# --------------------------------------------------------------------------- #
# 4. Residency: no host transfer of positions on a reused step
# --------------------------------------------------------------------------- #
def test_residency_no_host_transfer_on_reuse(monkeypatch):
    import cupy as cp
    from ase.neighborlist import NeighborListBuilder

    atoms = build("cubic", 256)
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    be = _backend()
    builder = NeighborListBuilder(CUTOFF, skin=0.5, neighbor_list=be,
                                  quantities="ijSD")
    base = atoms.get_positions()
    builder.update_device(cp.asarray(base), cell, pbc)   # initial build

    calls = {"n": 0}
    real = cp.asnumpy
    monkeypatch.setattr(cp, "asnumpy", lambda a, *x, **k: (
        calls.__setitem__("n", calls["n"] + 1), real(a, *x, **k))[1])

    moved = cp.asarray(base + 0.02)                      # < skin -> reuse
    rebuilt = builder.update_device(moved, cell, pbc)
    assert rebuilt is False
    assert calls["n"] == 0, "reused step pulled data to host via cp.asnumpy"


def test_needs_rebuild_native_kernel_threshold():
    """The migrated (CuPy -> C++/CUDA) update check: needs_rebuild returns an
    on-device uint8 scalar, correct across the skin**2 threshold, matching the
    reference max-squared-displacement computation, with no host sync inside the
    call itself."""
    import cupy as cp

    rng = np.random.RandomState(0)
    atoms = build("cubic", 256)
    base = atoms.get_positions()
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    skin = 0.4
    be = _backend()
    be.build_device(cp.asarray(base), cell, pbc, CUTOFF, "ijS")

    def reference(cur):
        return float(cp.max(((cp.asarray(cur) - cp.asarray(base)) ** 2)
                            .sum(axis=1))) > skin ** 2

    for mag in (0.0, 0.39, 0.41, 1.0):                   # straddle the skin
        cur = base.copy(); cur[0, 0] += mag
        nr = be.needs_rebuild(cp.asarray(cur), skin=skin)
        assert nr.__dlpack_device__()[0] == 2            # device-resident (CUDA)
        assert bool(nr) is reference(cur)                # __bool__ via C++ readback

    # The reduction is native, so needs_rebuild itself must not call into CuPy.
    import matscipy_neighbours._device as _dev
    orig = _dev._cupy
    called = {"n": 0}
    _dev._cupy = lambda: (called.__setitem__("n", called["n"] + 1), orig())[1]
    try:
        _ = be.needs_rebuild(cp.asarray(base), skin=skin)
    finally:
        _dev._cupy = orig
    assert called["n"] == 0, "needs_rebuild still imports/uses CuPy"


# --------------------------------------------------------------------------- #
# 5. Device tagging (the device-match contract)
# --------------------------------------------------------------------------- #
def test_device_tag_consistent():
    import cupy as cp

    atoms = build("cubic", 108)
    be = _backend()
    pos = cp.asarray(atoms.get_positions())
    res = be.build_device(pos, np.asarray(atoms.cell),
                          tuple(bool(b) for b in atoms.pbc), CUTOFF, "ijS")
    dtype = cp.cuda.runtime.getDevice()
    assert be.device == (2, dtype) or be.device[0] == 2  # CUDA device type
    for q in "ijS":
        assert res.get(q).__dlpack_device__() == be.device, \
            f"{q}: result device tag disagrees with backend.device"


# --------------------------------------------------------------------------- #
# 6. Padding & overflow (fixed-capacity, JAX-MD-style overflow flag)
# --------------------------------------------------------------------------- #
def test_padding_shape_stable_and_overflow_flag():
    import cupy as cp

    atoms = build("cubic", 256)
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    be = _backend()
    cap = 80

    r1 = be.build_device(cp.asarray(atoms.get_positions()), cell, pbc,
                         CUTOFF, max_capacity=cap)
    assert r1.padded and not r1.did_overflow
    idx = r1.get("idx")
    assert idx.shape == (len(atoms), cap)
    m = r1.mask()
    assert int(cp.asnumpy(m).sum()) == r1.n_edges

    # Shape stays constant across a rebuild with a different configuration.
    rng = np.random.RandomState(1)
    r2 = be.build_device(cp.asarray(atoms.get_positions() + rng.uniform(
        -0.3, 0.3, size=(len(atoms), 3))), cell, pbc, CUTOFF, max_capacity=cap)
    assert r2.get("idx").shape == idx.shape

    # Deliberately undersized capacity -> overflow flag, no silent drop.
    r3 = be.build_device(cp.asarray(atoms.get_positions()), cell, pbc,
                         CUTOFF, max_capacity=2)
    assert r3.did_overflow is True
    with pytest.raises(KeyError):
        r3.get("S")  # dense path has no cell shifts; COO unavailable here
