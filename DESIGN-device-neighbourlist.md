# Design note: a device-resident neighbour-list capability for ASE

**Status:** experimental / provisional — for steering-committee discussion and the
MACE-rewrite / TorchSim / JAX-MD device-MLIP authors.
**Companion spec:** [`SPEC-device-neighbourlist.md`](SPEC-device-neighbourlist.md).
**What this note reports:** a working prototype of the spec, validated on real GPU
hardware (NVIDIA RTX A4500, arch 86) against **three independent device backends**
behind one ASE protocol, plus the contestable decisions to settle.

## 1. Motivation

Every group doing device-resident MLIP MD currently reinvents the device-resident
neighbour-graph handoff ad hoc (MACE #1296 bolts a torch GPU neighbour list into
`AtomicData`; the water-ice / dense-hydrogen papers reimplement matscipy's logic
as tensor ops, several with O(N²) broadcast distance that blows up memory). A
neutral, **DLPack-based** capability owned by ASE consolidates this the way extxyz
consolidated trajectory I/O. ASE ships **zero GPU code**: the contract is duck-typed
over `__dlpack__` / `__dlpack_device__`, so ASE never depends on
torch/jax/cupy/cuda. The capability is purely additive — it does not touch the host
neighbour-list protocol from MR !4163.

## 2. What was built and proven

Two protocols, additive, in a new `typing`-only module
`ase/_4/plugins/neighborlist_device.py` (beside !4163's host `neighborlist.py`):

- **`DeviceNeighborList`** — `device`, `differentiable`, `build_device(...)`,
  `needs_rebuild(...)`.
- **`DeviceNeighborResult`** — `n_edges`, `did_overflow`, `padded`, `get(name)`,
  `mask()`.
- **`BatchedDeviceNeighborList`** — optional marker for natively-batched backends.

One skin-wrapper extension: `NeighborListBuilder.update_device(...)` in
`ase/neighborlist.py` (does not fork the host `update()`), delegating the
displacement test to the backend's on-device `needs_rebuild` so positions never
come to host on a reused step.

Two device backends behind the **same** protocol:

| backend | provenance | framework | path |
|---|---|---|---|
| **matscipy-neighbours** | author | CuPy / CUDA cell list (native DLPack) | `matscipy_neighbours._device` |
| **ALCHEMI (nvalchemiops)** | hardware vendor (NVIDIA) | JAX / Warp cell list | `alchemi_device` (benchmark repo) |
| **Vesin** | ecosystem (metatensor/metatomic) | CuPy / CUDA cell list | `vesin_device` (benchmark repo) |

**Validation on the A4500 (all green):**

- **Cross-backend equivalence** — both backends' edge sets are **integer-exact in
  `(i, j, S)`** and float-tolerant in `D` (≤1e-10) versus the host matscipy oracle,
  on cubic (fcc Ni, full PBC), low-symmetry (hcp Ti, non-orthogonal), and slab
  (mixed PBC) systems. Author and vendor agreeing through one protocol is the
  load-bearing evidence the abstraction is real, not matscipy-specific.
- **Quantity-order independence** — `get("S")` returns the shifts whether
  `build_device` was asked for `"ijSD"` or `"SDji"` (the §2.3 silent-corruption guard).
- **Residency** — on a reused (no-rebuild) step the matscipy path makes **zero**
  `cupy.asnumpy` calls; `needs_rebuild` returns an on-device 0-d scalar
  (`__dlpack_device__()[0] != CPU`) — only that 1-element scalar is ever synced.
- **Skin correctness** — over a rattled trajectory the skin-reused list, filtered
  to the true cutoff at current positions, equals a fresh per-step build (no missed
  or invented edges); a displacement beyond the skin always forces a rebuild.
- **Padding & overflow** — with `max_capacity` the output shape is constant across
  rebuilds; an undersized capacity sets `did_overflow` with no silent edge drop.

Test suites: `test_device_neighbourlist.py` (matscipy, 9 tests) and
`test_alchemi_device.py` (ALCHEMI, 8 tests).

## 3. Implementation findings worth keeping

- **The native matscipy-neighbours API already does most of this.**
  `neighbour_list(..., positions=<device>, array_namespace="dlpack")` returns
  device-resident DLPack arrays in requested order; `neighbour_matrix(...,
  max_neighbours=K)` plus the raw `_ext.neighbour_matrix_dlpack` gives fixed-capacity
  padded output with an overflow bool. The adapter is a thin protocol shim over them.
- **`needs_rebuild` is a native C++/CUDA kernel (CuPy removed from the update
  check).** `_ext.needs_rebuild` runs a CUB/hipCUB transform-then-
  `DeviceReduce::Max` over per-atom squared displacement and thresholds against
  `skin**2` on-device, returning a 1-element uint8 device DLPack scalar; the
  reference positions are snapshotted on-device by `_ext.clone_device` into a
  reusable handle. Nothing is read back inside the call (residency preserved);
  the eager `bool()` sync is a tiny `DLPackTensor.__bool__` (one-element
  device→host read), so a torch-/JAX-only consumer drives the skin loop with no
  CuPy. (CuPy remains only in `build_device` for returning the edge arrays and
  the padded mask — separate from the update check.)
- **ALCHEMI silently truncates on overflow.** Its dense `cell_list` does *not* raise
  when `max_neighbors` is too small, but `num_neighbors` reports the *true* count, so
  the adapter flags `did_overflow = (count > max_capacity).any()` itself. This is a
  concrete vindication of "never trust the kernel; surface the flag at the contract."
- **DLPack-import direction matters.** NumPy arrays implement `__dlpack__` but
  importing them via `jnp.from_dlpack` keeps them **host-resident**, which silently
  pushed ALCHEMI's Warp kernels onto the Host platform. Host arrays must go through
  `jnp.asarray` (default device); DLPack import is for genuine device arrays only.
- **CuPy and JAX need mutually exclusive shells here.** CuPy's JIT (matscipy path)
  needs `module load CUDA/12.6.0` for its headers; JAX (ALCHEMI path) uses its own
  bundled CUDA wheels and *fails to init CUDA* when that system module is loaded,
  falling back to CPU. The two suites therefore run in separate invocations, and the
  cross-backend equivalence is established transitively via the common host oracle
  rather than in one process. (A real bridge package would pin one framework.)
- **Float width.** ALCHEMI defaults to float32; near-cutoff edges then flip versus a
  float64 oracle and break integer-exact equivalence. The adapter enables
  `jax_enable_x64` to match. Backends should document their precision.
- **Eager-call timing is the wrong benchmark for these backends — and that is the
  whole point of the protocol.** ALCHEMI timed as isolated eager `build_device`
  calls shows a fixed ~86 ms/call floor (JAX dispatch + Warp launch + host-side
  cell-grid sizing), *N-independent*. Under `jax.jit` with the grid pinned, that
  floor vanishes and its kernel is 0.5–3.6 ms (4 k–108 k), ~2.5× faster than
  matscipy's at 108 k. The device capability exists so the build runs *inside* a
  compiled, device-resident loop (rebuild rarely via `needs_rebuild`, never leave
  the device) — exactly where that overhead is amortised to zero. A naïve
  eager-per-step host harness inverts the ranking and hides it. (Full
  decomposition in `FINDINGS.md` → "Why is ALCHEMI-gpu slower than mn-gpu here?".)

## 4. Contestable decisions (to settle with the committee)

1. **`needs_rebuild` returns a device scalar** (we return an on-device 0-d bool).
   Confirmed necessary: a Python `bool` would force a per-step host sync and defeat
   residency. ASE deliberately owns **no** `bool_from_device_scalar` helper — the
   eager sync is a replaceable seam in `update_device._to_bool`.
2. **Optional `max_capacity` padding + `did_overflow`.** Prior art agrees fixed-
   capacity device lists are universal and need an overflow signal, not truncation:
   JAX-MD `did_buffer_overflow`, TorchMD-Net `-1` padding, ALCHEMI fixed capacity.
   Keep it optional (tight by default).
3. **Batch fork.** Single-system core + optional `BatchedDeviceNeighborList`
   (implemented: matscipy single-system; ALCHEMI advertises the marker). Caveat:
   TorchSim is batched by construction and is the target consumer, so batch support
   is effectively required for adoption — revisit whether batch belongs in the core.
4. **Output format: COO-only core, dense as backend-specific.** The tight path is the
   canonical COO `i/j/S/D` (both backends); the padded path exposes the dense
   `idx/count/(shift)` layout and has **no cell shifts**, so COO `get("S")` is a tight-
   path guarantee only. Both adapters raise on padded `get("S")` rather than fabricate.
5. **Skin threshold.** !4163 uses **full `skin**2`** (`((pos-ref)**2).sum(1).max() >
   skin**2`), padding scalar/dict cutoffs by `2*skin`; `update_device` matches it.
   (Resolves the spec's "skin vs skin/2" question: it is full skin.)
6. **`needs_rebuild` and cell deltas (NPT).** v1 keys on positions only; cell/pbc
   changes are caught by the wrapper's host-side comparison (cell is 3×3, not a
   residency concern). Folding cell deltas into the device reduction is open.
7. **Expose `D` from a non-differentiable backend.** Yes, with the documented
   autograd caveat (`differentiable=False` ⇒ recompute `D` from `i,j,S` in-graph for
   gradients). Both backends set `differentiable=False`.

## 5. Next steps

- ~~**Vesin (ecosystem, the third backend).**~~ **Done** — via the plain `vesin`
  wheel, which ships a **CUDA backend reachable through its CuPy interface**:
  passing CuPy device positions to `vesin.NeighborList.compute` returns
  device-resident CuPy arrays. `vesin_device.VesinDeviceNeighborList` wraps that
  (`differentiable=False` *only because the CuPy path is not an autograd
  framework*; COO only, so **no padded path**; `needs_rebuild` is a CuPy
  reduction). It is edge-exact with the other two backends vs the host oracle,
  completing the **author / vendor / ecosystem** spread. CuPy was chosen for the
  adapter as it needs no extra install; `vesin-torch` is a separate option (next).
  Caveat: Vesin `dlopen`s `libcudart`, so it needs the CUDA runtime on the loader
  path (system module or the venv `nvidia-cuda-runtime` wheel on `LD_LIBRARY_PATH`,
  which also lets CuPy + JAX + Vesin share one process with no system module).
- A **differentiable *device*** witness is still open here, but **`vesin-torch` is
  the natural candidate**: confirmed by its author, it runs on **GPU** and is
  autograd-differentiable (an earlier note that it was CPU-only was wrong). A
  `vesin-torch` device adapter would give `differentiable=True` on-device — the
  matscipy/ALCHEMI/Vesin-CuPy paths are all `differentiable=False`. (TorchMD-Net
  §5.5 or a Reactant in-graph builder §5.4 are alternatives.)
- ~~Native matscipy `needs_rebuild` kernel to drop the CuPy dependency.~~ **Done**
  — see the implementation finding above.
- **`NeighborListPlugin.device_implementation=`** upstream, so the device capability
  is advertised through the same plugin record instead of a side-factory.
- **A device-resident consumer demo** (TorchSim step, or ase-jax swapping JAX-MD's
  `neighbor_fn` for protocol edges) to exercise the contract end-to-end in a compiled
  loop.
