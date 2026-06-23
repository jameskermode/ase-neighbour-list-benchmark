"""Experimental: ALCHEMI (NVIDIA) device neighbour-list adapter (JAX path).

Second device backend for the SPEC-device-neighbourlist.md contract, behind the
*same* ASE experimental protocols as the matscipy-neighbours adapter
(:mod:`ase._4.plugins.neighborlist_device`).  It wraps NVIDIA's
``nvalchemiops.jax.neighbors`` O(N) cell list, which produces device-resident
JAX arrays (``jax.jit`` / batched), so this exercises the protocol from a second,
independent implementation (hardware vendor) on the JAX side -- proving the
abstraction is not matscipy-specific.

In a real deployment this adapter would live in an ``nvalchemi``↔ASE bridge
package; it sits in the benchmark repo here because that is where the
cross-backend equivalence test runs.

Implements both the single-system core :class:`DeviceNeighborList` and the
optional :class:`BatchedDeviceNeighborList` marker (ALCHEMI is natively batched).
``differentiable`` is ``False`` (index/shift output; ``D`` recomputed).

Notes
-----
* **Overflow guard.** ALCHEMI's dense ``cell_list`` does *not* raise when
  ``max_neighbors`` is too small -- it silently truncates.  Its ``num_neighbors``
  reports the *true* per-atom count regardless, so this adapter flags
  ``did_overflow = (count > max_capacity).any()`` on device.  Surfacing the flag
  (rather than relying on the kernel) is exactly the "never silently truncate"
  requirement of the spec.
* ``needs_rebuild`` delegates to ALCHEMI's own device-resident
  ``neighbor_list_needs_rebuild`` (returns a JAX scalar) -- no host sync.
* Scalar cutoffs only; a per-element/per-pair dict cutoff raises (the backend
  cannot honour it, so it does not silently differ).
"""
from __future__ import annotations

import os

import numpy as np

# JAX preallocates ~75% of GPU memory on first use by default, which starves the
# CuPy / Vesin backends sharing the device (in the benchmark's correctness gate
# and across timing subprocesses -> CUDA OOM). Allocate on demand instead. Set
# before JAX is imported below.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Warp (which nvalchemiops builds on) registers its JAX FFI kernel handlers for a
# specific platform -- GPU vs Host -- *when nvalchemiops is first imported*,
# based on whether JAX's CUDA backend is already live at that moment.  If warp
# imports before JAX has initialised CUDA, the cell-list kernels register on the
# Host platform and every device call then fails with "No FFI handler registered
# ... on a platform Host".  Force JAX's CUDA backend live here, before any
# nvalchemiops import, so the kernels register on the GPU.
try:
    import jax as _jax
    # Match the float64 host oracle: with x64 off, near-cutoff edges can flip vs
    # a float64 reference and break the integer-exact edge-set equivalence.
    _jax.config.update('jax_enable_x64', True)
    import jax.numpy as _jnp
    (_jnp.zeros(1) + 1).block_until_ready()
except Exception:  # pragma: no cover - jax absent or no GPU; adapter unusable
    pass


def _to_jax(x):
    """Normalise input to a JAX array on the default (GPU) device.

    Host (NumPy) arrays go through ``jnp.asarray`` so they land on the default
    device.  NB: NumPy arrays *do* implement ``__dlpack__``, but importing them
    via ``jnp.from_dlpack`` would keep them CPU-resident -- which then makes the
    Warp cell-list kernels dispatch on the Host platform.  DLPack import is
    therefore reserved for genuine *device* arrays (CuPy/torch), kept zero-copy.
    """
    import jax
    import jax.numpy as jnp
    if isinstance(x, jax.Array):
        return x
    if isinstance(x, np.ndarray):
        return jnp.asarray(x)
    dev = getattr(x, '__dlpack_device__', None)
    if dev is not None and dev()[0] != 1:        # device (non-CPU) DLPack array
        return jnp.from_dlpack(x)
    return jnp.asarray(np.asarray(x))


_PLATFORM_TO_DLPACK = {'gpu': 2, 'cuda': 2, 'rocm': 10, 'cpu': 1}


class AlchemiDeviceResult:
    """On-device neighbour data (JAX); satisfies ``DeviceNeighborResult``."""

    def __init__(self, arrays, *, padded, did_overflow, n_edges, mask=None):
        self._arrays = arrays
        self._padded = bool(padded)
        self._did_overflow = did_overflow
        self._n_edges = n_edges
        self._mask = mask

    @property
    def n_edges(self):
        if not isinstance(self._n_edges, int):
            self._n_edges = int(self._n_edges)
        return self._n_edges

    @property
    def did_overflow(self):
        return self._did_overflow

    @property
    def padded(self):
        return self._padded

    def get(self, quantity):
        try:
            return self._arrays[quantity]
        except KeyError:
            available = ', '.join(sorted(self._arrays)) or '(none)'
            raise KeyError(
                f'quantity {quantity!r} not available; have {available}. '
                + ('The padded path exposes the dense idx/count/shift_matrix '
                   'layout; request the tight path (max_capacity=None) for COO '
                   'i/j/S/D.' if self._padded else
                   'It was not among the requested quantities.')) from None

    def mask(self):
        return self._mask


class AlchemiDeviceNeighborList:
    """ALCHEMI device backend; satisfies ``DeviceNeighborList`` (+ batched)."""

    differentiable = False

    def __init__(self, device_id=0):
        # Import here so merely importing the module does not require JAX/ALCHEMI.
        import nvalchemiops.jax.neighbors as _nl  # noqa: F401
        self._device = (2, int(device_id))
        self._ref = None
        self._cell = None
        self._cell_inv = None
        self._pbc = None

    @property
    def device(self):
        return self._device

    @staticmethod
    def _check_scalar_cutoff(cutoff):
        if isinstance(cutoff, dict) or np.ndim(cutoff) != 0:
            raise NotImplementedError(
                'the ALCHEMI device backend supports only a scalar cutoff; '
                'per-element/per-pair cutoffs are not honoured.')

    def _prep(self, positions, cell, pbc):
        import jax.numpy as jnp
        pos = _to_jax(positions)
        cell3 = _to_jax(cell).reshape(1, 3, 3)
        pbc_arr = jnp.asarray(
            np.array([bool(b) for b in pbc], dtype=bool).reshape(1, 3))
        dev = next(iter(pos.devices()))
        self._device = (_PLATFORM_TO_DLPACK.get(dev.platform, 2), int(dev.id))
        return pos, cell3, pbc_arr

    def build_device(self, positions, cell, pbc, cutoff, quantities='ijS', *,
                     self_interaction=False, max_capacity=None, stream=None):
        if self_interaction:
            raise NotImplementedError(
                'the ALCHEMI device backend does not support '
                'self_interaction=True')
        self._check_scalar_cutoff(cutoff)
        import jax.numpy as jnp
        import nvalchemiops.jax.neighbors as nl

        pos, cell3, pbc_arr = self._prep(positions, cell, pbc)
        # Cache reference state on device for needs_rebuild.
        self._ref = pos
        self._cell = cell3
        self._cell_inv = jnp.linalg.inv(cell3)
        self._pbc = pbc_arr

        if max_capacity is None:
            invalid = set(quantities) - set('ijdDS')
            if invalid or not quantities:
                raise ValueError(
                    f'quantities must be a non-empty subset of "ijdDS"; '
                    f'got {quantities!r}.')
            nbr, _ptr, shifts = nl.cell_list(
                pos, float(cutoff), cell=cell3, pbc=pbc_arr,
                return_neighbor_list=True)
            i, j = nbr[0], nbr[1]
            avail = {'i': i, 'j': j, 'S': shifts}
            if 'D' in quantities or 'd' in quantities:
                D = pos[j] - pos[i] + shifts @ cell3[0]
                avail['D'] = D
                avail['d'] = jnp.linalg.norm(D, axis=1)
            arrays = {q: avail[q] for q in quantities}  # NAME-keyed
            return AlchemiDeviceResult(arrays, padded=False, did_overflow=False,
                                       n_edges=int(i.shape[0]))

        mat, count, shift_mat = nl.cell_list(
            pos, float(cutoff), cell=cell3, pbc=pbc_arr,
            max_neighbors=int(max_capacity), return_neighbor_list=False)
        # ALCHEMI silently truncates on overflow; count is the TRUE per-atom
        # count, so flag overflow ourselves (never silently drop).
        did_overflow = bool((count > int(max_capacity)).any())
        col = jnp.arange(int(max_capacity))
        mask = col[None, :] < count[:, None]
        arrays = {'idx': mat, 'count': count, 'shift_matrix': shift_mat}
        return AlchemiDeviceResult(arrays, padded=True,
                                   did_overflow=did_overflow,
                                   n_edges=count.sum(), mask=mask)

    def needs_rebuild(self, positions, *, skin, stream=None):
        if self._ref is None:
            raise RuntimeError('call build_device(...) before needs_rebuild(...)')
        import nvalchemiops.jax.neighbors as nl
        cur = _to_jax(positions)
        # needs_rebuild wants a batched cell (1,3,3) but an unbatched pbc (3,).
        return nl.neighbor_list_needs_rebuild(
            self._ref, cur, float(skin),
            cell=self._cell, cell_inv=self._cell_inv,
            pbc=self._pbc.reshape(3))

    # -- optional BatchedDeviceNeighborList marker -----------------------------
    def build_device_batched(self, positions, cell, pbc, cutoff,
                             quantities='ijS', *, self_interaction=False,
                             max_capacity=None, stream=None):
        """Build neighbour lists for a batch of systems via ALCHEMI's native
        batched cell list.

        ``positions`` is the concatenation of all systems' coordinates;
        ``batch_idx`` (length n_atoms, the per-atom system index) is passed
        through ``**kwargs`` of the underlying op via the ``batch_idx`` keyword
        below.  Experimental: batch layout for the *output* edges is left to the
        caller (see the spec's batch fork).
        """
        if self_interaction:
            raise NotImplementedError('self_interaction=True not supported')
        self._check_scalar_cutoff(cutoff)
        import nvalchemiops.jax.neighbors as nl
        pos = _to_jax(positions)
        cell3 = _to_jax(cell)            # (n_systems, 3, 3)
        pbc_arr = _to_jax(pbc)           # (n_systems, 3)
        nbr, ptr, shifts = nl.batch_cell_list(
            pos, float(cutoff), cell=cell3, pbc=pbc_arr,
            return_neighbor_list=True)
        i, j = nbr[0], nbr[1]
        avail = {'i': i, 'j': j, 'S': shifts[0] if isinstance(shifts, tuple)
                 else shifts}
        arrays = {q: avail[q] for q in quantities if q in avail}
        return AlchemiDeviceResult(arrays, padded=False, did_overflow=False,
                                   n_edges=int(i.shape[1] if i.ndim > 1
                                               else i.shape[0]))
