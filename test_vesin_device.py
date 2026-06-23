"""Tests for the third device backend: Vesin (ecosystem) via the CuPy path.

Proves the SPEC-device-neighbourlist.md contract holds for a third independent
implementation behind the same ASE device protocols — cross-backend equivalence
vs the host matscipy oracle (so, transitively, vs the matscipy and ALCHEMI device
backends), quantity-order independence, on-device needs_rebuild, residency, and
that the unsupported padded path raises.

Vesin's CuPy GPU path ``dlopen``s libcudart, so this needs ``libcudart.so`` on the
loader path. Run with the venv CUDA-runtime wheel on LD_LIBRARY_PATH (or a CUDA
module loaded), e.g.::

    LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH" \
        uv run --no-sync pytest test_vesin_device.py

Auto-skips when CuPy / a CUDA device / Vesin's CUDA backend is unavailable.
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
        from vesin_device import VesinDeviceNeighborList
        be = VesinDeviceNeighborList()
        if not isinstance(be, DeviceNeighborList):
            return "vesin backend is not a DeviceNeighborList"
        # Probe the GPU compute -- catches a missing libcudart cleanly.
        be.build_device(cp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                        np.eye(3) * 10.0, (False, False, False), 2.0, "i")
    except Exception as exc:  # pragma: no cover - env dependent
        return f"vesin device backend unavailable: {exc}"
    return None


_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=str(_REASON))

CUTOFF = 2.5
SYSTEMS = [("cubic", 256), ("lowsym", 256), ("slab", 200)]


def _backend():
    from vesin_device import VesinDeviceNeighborList
    return VesinDeviceNeighborList()


def _host_oracle(atoms, cutoff):
    from matscipy_neighbours import neighbour_list
    return neighbour_list("ijdDS", atoms, cutoff)


def test_satisfies_protocol():
    from ase._4.plugins.neighborlist_device import DeviceNeighborList
    be = _backend()
    assert isinstance(be, DeviceNeighborList)
    assert be.differentiable is False
    assert be.device[0] == 2  # CUDA


@pytest.mark.parametrize("kind,n", SYSTEMS)
def test_equivalence_vs_host_oracle(kind, n):
    import cupy as cp

    atoms = build(kind, n)
    posn = atoms.get_positions()
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)

    res = _backend().build_device(cp.asarray(posn), cell, pbc, CUTOFF, "ijSD")
    i = cp.asnumpy(res.get("i")).astype(np.int64)
    j = cp.asnumpy(res.get("j")).astype(np.int64)
    S = cp.asnumpy(res.get("S"))
    D = cp.asnumpy(res.get("D"))
    cdev = canonicalise(i, j, np.linalg.norm(D, axis=1), D, S)

    hi, hj, hd, hD, hS = _host_oracle(atoms, CUTOFF)
    chost = canonicalise(hi, hj, hd, hD, hS)

    assert np.array_equal(cdev["keys"], chost["keys"]), \
        f"{kind}: Vesin edge set differs from host oracle"
    if len(cdev["i"]):
        assert np.abs(cdev["D"] - chost["D"]).max() <= AGREE_TOL


def test_quantity_order_independence():
    import cupy as cp

    atoms = build("lowsym", 256)
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    pos = cp.asarray(atoms.get_positions())
    be = _backend()
    a = be.build_device(pos, cell, pbc, CUTOFF, "ijSD")
    b = be.build_device(pos, cell, pbc, CUTOFF, "SDji")
    for q in "ijSD":
        assert np.array_equal(cp.asnumpy(a.get(q)), cp.asnumpy(b.get(q))), \
            f"get({q!r}) depends on request order"


def test_device_residency_of_outputs():
    import cupy as cp

    atoms = build("cubic", 256)
    be = _backend()
    res = be.build_device(cp.asarray(atoms.get_positions()),
                          np.asarray(atoms.cell),
                          tuple(bool(b) for b in atoms.pbc), CUTOFF, "ijS")
    for q in "ijS":
        assert res.get(q).__dlpack_device__()[0] == 2   # CUDA-resident
    assert be.device[0] == 2


def test_needs_rebuild_device_scalar_threshold():
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

    for mag in (0.0, 0.39, 0.41, 1.0):
        cur = base.copy(); cur[0, 0] += mag
        nr = be.needs_rebuild(cp.asarray(cur), skin=skin)
        assert nr.__dlpack_device__()[0] == 2            # device-resident
        assert bool(nr) is reference(cur)


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


def test_padded_path_unsupported():
    import cupy as cp

    atoms = build("cubic", 108)
    be = _backend()
    with pytest.raises(NotImplementedError):
        be.build_device(cp.asarray(atoms.get_positions()), np.asarray(atoms.cell),
                        tuple(bool(b) for b in atoms.pbc), CUTOFF,
                        max_capacity=64)
