"""Tests for the second device backend: ALCHEMI (NVIDIA) via the JAX path.

Proves the SPEC-device-neighbourlist.md contract holds for a *second, independent*
implementation behind the same ASE device protocols -- cross-backend equivalence
vs the host matscipy oracle (so, transitively, vs the matscipy device backend),
quantity-order independence, on-device needs_rebuild, device residency, and
fixed-capacity padding + overflow (with the adapter's own overflow guard, since
ALCHEMI silently truncates).

Environment note: ALCHEMI uses JAX with its own *bundled* CUDA wheels.  Do NOT
``module load CUDA/...`` for this file -- a system CUDA module makes JAX fail to
init CUDA and fall back to CPU.  (Conversely, the matscipy device tests need the
system module for CuPy's JIT.)  Run with::

    uv run --no-sync pytest test_alchemi_device.py

Auto-skips when JAX / a CUDA device / nvalchemiops is unavailable.
"""
# Import the adapter FIRST: its module-level code forces JAX's CUDA backend live
# before anything (here or in _skip_reason) imports nvalchemiops/Warp, so Warp
# registers its JAX FFI kernels on the GPU platform (see alchemi_device.py).
import alchemi_device  # noqa: E402,F401  (import-order is load-bearing)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from correctness import AGREE_TOL, canonicalise  # noqa: E402
from systems import build  # noqa: E402


def _skip_reason():
    try:
        import jax
        if jax.default_backend() != "gpu":
            return f"jax backend is {jax.default_backend()!r}, not gpu"
    except Exception as exc:  # pragma: no cover - env dependent
        return f"jax unavailable: {exc}"
    try:
        from alchemi_device import AlchemiDeviceNeighborList
        AlchemiDeviceNeighborList()   # imports nvalchemiops (after jax is live)
    except Exception as exc:  # pragma: no cover - env dependent
        return f"ALCHEMI device backend unavailable: {exc}"
    return None


_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=str(_REASON))

CUTOFF = 2.5
SYSTEMS = [("cubic", 256), ("lowsym", 256), ("slab", 200)]


def _backend():
    from alchemi_device import AlchemiDeviceNeighborList
    return AlchemiDeviceNeighborList()


def _host_oracle(atoms, cutoff):
    from matscipy_neighbours import neighbour_list
    return neighbour_list("ijdDS", atoms, cutoff)


def _protocols():
    from ase._4.plugins.neighborlist_device import (
        BatchedDeviceNeighborList, DeviceNeighborList, DeviceNeighborResult)
    return DeviceNeighborList, DeviceNeighborResult, BatchedDeviceNeighborList


def test_satisfies_protocols():
    DeviceNeighborList, _, BatchedDeviceNeighborList = _protocols()
    be = _backend()
    assert isinstance(be, DeviceNeighborList)
    assert isinstance(be, BatchedDeviceNeighborList)  # ALCHEMI is natively batched
    assert be.differentiable is False


@pytest.mark.parametrize("kind,n", SYSTEMS)
def test_equivalence_vs_host_oracle(kind, n):
    import jax.numpy as jnp

    atoms = build(kind, n)
    posn = atoms.get_positions()
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)

    res = _backend().build_device(jnp.asarray(posn), cell, pbc, CUTOFF, "ijSD")
    i = np.asarray(res.get("i")); j = np.asarray(res.get("j"))
    S = np.asarray(res.get("S")); D = np.asarray(res.get("D"))
    cdev = canonicalise(i, j, np.linalg.norm(D, axis=1), D, S)

    hi, hj, hd, hD, hS = _host_oracle(atoms, CUTOFF)
    chost = canonicalise(hi, hj, hd, hD, hS)

    assert np.array_equal(cdev["keys"], chost["keys"]), \
        f"{kind}: ALCHEMI edge set differs from host oracle"
    if len(cdev["i"]):
        assert np.abs(cdev["D"] - chost["D"]).max() <= AGREE_TOL


def test_quantity_order_independence():
    import jax.numpy as jnp

    atoms = build("lowsym", 256)
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    pos = jnp.asarray(atoms.get_positions())
    be = _backend()
    a = be.build_device(pos, cell, pbc, CUTOFF, "ijSD")
    b = be.build_device(pos, cell, pbc, CUTOFF, "SDji")
    for q in "ijSD":
        assert np.array_equal(np.asarray(a.get(q)), np.asarray(b.get(q))), \
            f"get({q!r}) depends on request order"


def test_device_residency_of_outputs():
    import jax.numpy as jnp

    atoms = build("cubic", 256)
    be = _backend()
    res = be.build_device(jnp.asarray(atoms.get_positions()),
                          np.asarray(atoms.cell),
                          tuple(bool(b) for b in atoms.pbc), CUTOFF, "ijS")
    for q in "ijS":
        # device_type 2 == CUDA; arrays never came to host.
        assert res.get(q).__dlpack_device__()[0] == 2
    assert be.device[0] == 2


def test_needs_rebuild_is_device_scalar():
    import jax.numpy as jnp

    atoms = build("cubic", 256)
    be = _backend()
    pos = jnp.asarray(atoms.get_positions())
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    be.build_device(pos, cell, pbc, CUTOFF, "ijS")

    nr = be.needs_rebuild(pos, skin=0.3)
    assert nr.__dlpack_device__()[0] == 2          # on device, no host sync
    assert bool(nr) is False
    moved = pos.at[0, 0].add(1.0)
    assert bool(be.needs_rebuild(moved, skin=0.3)) is True


def test_padding_and_overflow_guard():
    import jax.numpy as jnp

    atoms = build("cubic", 256)
    cell = np.asarray(atoms.cell)
    pbc = tuple(bool(b) for b in atoms.pbc)
    pos = jnp.asarray(atoms.get_positions())
    be = _backend()

    r = be.build_device(pos, cell, pbc, CUTOFF, max_capacity=80)
    assert r.padded and not r.did_overflow
    assert r.get("idx").shape == (len(atoms), 80)
    assert int(np.asarray(r.mask()).sum()) == r.n_edges

    # ALCHEMI silently truncates; the adapter must still flag overflow.
    r2 = be.build_device(pos, cell, pbc, CUTOFF, max_capacity=2)
    assert r2.did_overflow is True
