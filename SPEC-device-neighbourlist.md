# Spec: Device-resident neighbour-list capability for ASE plugins

**Status:** provisional / experimental — strawman for prototyping and steering-committee discussion
**Owner:** James Kermode
**Target:** layer on top of existing in-flight work, do not replace it
**Build order:** DEVICE-FIRST. The device capability is the deliverable; the host path
already exists in !4163 / PR #2. Validate the protocol against THREE device backends
(matscipy-neighbours, Vesin, ALCHEMI Toolkit-Ops) so it is demonstrably not
matscipy-specific. Three independent implementations is the answer to "premature
standardisation."

---

## 0. Context Claude Code must load first

This work sits on top of two existing, open contributions. **Read them before writing
anything** — the goal is to *extend*, not re-architect.

1. **ASE MR !4163** — "Add pluggable neighbour-list backends (v4): protocol, plugin,
   skin wrapper". Targets the `ase4-plugins-all-calculators` integration branch.
   Defines the *host* neighbour-list protocol (`NeighborListFunction`-style callable),
   the plugin registration mechanism, and a Verlet-skin reuse wrapper.
   - Inspect: the protocol/ABC module, the skin wrapper, the plugin registration
     (entry-point group — follow the same pattern ASE uses for `ase.ioformats`).

2. **matscipy-neighbours PR #2** — "Add pip packaging (scikit-build-core) and an ASE
   neighbour-list plugin". Registers matscipy-neighbours as a host backend implementing
   the !4163 protocol. CPU path; rejects `self_interaction=True` rather than silently
   differing.
   - matscipy-neighbours also has a **native DLPack device API** (CUDA/HIP cell list,
     emits `i, j, D, S` as DLPack, consumable by torch/JAX/CuPy). That native API is the
     thing this spec bridges into ASE — it is NOT currently exposed through the ASE
     plugin, only through matscipy-neighbours' own functions.

**ASE conventions to honour (verify against the actual tree, do not trust memory):**
- Quantity strings: `"ijdDS"` — `i`,`j` endpoint indices; `d` scalar distance;
  `D` edge vectors; `S` integer cell shifts. `D = positions[j] - positions[i] + S @ cell`.
- Neighbour list sorted by first index `i`; order of `j` within an `i` not guaranteed.
- Existing skin/update semantics: rebuild when
  `((positions - ref_positions)**2).sum(1).max() > skin**2` (note: against skin, the
  matscipy/ASE convention; confirm whether !4163 uses skin or skin/2 and match it).
- External packages register via entry points (precedent: `ase.ioformats` /
  `ExternalIOFormat`). Use the neighbour-list entry-point group !4163 defines.
- Design philosophy (from ASE's calculator-interface proposal): do not force the
  lowest common denominator. Capabilities are opt-in; host-only backends stay simple.

---

## 1. Goal

Define an **optional device capability** that a neighbour-list backend MAY implement,
allowing edge data to be built and maintained **on-device** and exchanged **via DLPack**,
so that a device-resident calculator (torch/JAX/Reactant MLIP) can run geometry
optimisation or MD **without host round-trips per step**.

Non-goals:
- Do NOT make ASE depend on torch/jax/cupy/cuda. All device I/O is duck-typed over the
  DLPack protocol (`__dlpack__`, `__dlpack_device__`). ASE ships zero GPU code.
- Do NOT change the host protocol from !4163. This is purely additive.
- Do NOT claim device residency through the existing host `Atoms`-in/ndarray-out path.
  That path is host-resident by definition; this capability is a separate, discoverable
  interface.

### Why this is needed (one paragraph for the design note)
Today every group doing device-resident MLIP MD reinvents the device-resident neighbour
graph handoff ad hoc (e.g. MACE issue #1296 bolts a torch GPU neighbour list straight
into AtomicData; the water-ice / dense-hydrogen papers reimplement matscipy's logic as
vectorised tensor ops — several with O(N^2) broadcast distance, which blows up memory on
large cells). A neutral, DLPack-based capability owned by ASE consolidates this the way
extxyz consolidated trajectory I/O. Timing matters: the MACE rewrite's interfaces are
still fluid, so a contract defined now can be adopted rather than retrofitted.

---

## 2. The capability protocols

Add a NEW module (do not edit the host protocol module): `protocols_device.py`
(or wherever !4163 keeps its protocols — match its location/naming). Mark every public
symbol experimental in its docstring.

### 2.1 `DeviceNeighborList` (runtime-checkable Protocol)

A backend implementing this advertises it can build/maintain a device-resident list.
Calculators discover it via `isinstance(backend, DeviceNeighborList)`.

Required members:

- `device -> tuple[int, int]`
  DLPack `(device_type, device_id)` the backend operates on. **Consumers MUST check this
  matches their own device before assuming zero-copy.** Mismatch => copy-or-error; the
  backend never silently crosses devices.

- `differentiable -> bool`
  `True` iff `D` is produced inside a graph differentiable w.r.t. input positions (rare
  for external kernels; matscipy-neighbours device path => `False`; a fully-traced
  in-graph builder, e.g. a Reactant backend, could be `True`). Autograd consumers MUST
  request only `i,j,S` and recompute `D` themselves when `differentiable` is `False`.

- `build_device(positions, cell, pbc, cutoff, quantities="ijS", *,
   self_interaction=False, max_capacity=None, stream=None) -> DeviceNeighborResult`
  - `positions`: `(n_atoms, 3)` device array, DLPack.
  - `cell`: `(3, 3)` device array, DLPack.
  - `pbc`: `tuple[bool, bool, bool]`.
  - Builds on-device; returns a `DeviceNeighborResult` whose arrays are DLPack-exportable
    and resident on `self.device`. **Does not copy to host.**
  - `max_capacity`: if given, edge arrays are padded to exactly `max_capacity` rows so the
    output **shape is stable across rebuilds** (required for torch.compile / CUDA-graph /
    XLA consumers). Padding entries flagged via `mask()`; `n_edges` gives the logical
    count. If `None`, arrays are tight (variable length).
  - `stream`: consumer's stream handle (DLPack stream convention); see §3.

- `needs_rebuild(positions, *, skin, stream=None) -> <device scalar>`
  - Returns an **on-device** 0-d DLPack array (bool/uint8), truthy iff max displacement
    since last build exceeds the skin threshold.
  - **CRITICAL DESIGN POINT:** the reduction runs on device and returns a device scalar.
    The wrapper does NOT pull positions to host to decide. A fully-compiled loop consumes
    this scalar inside its control flow with no host round-trip; an eager consumer syncs
    explicitly. Returning a Python `bool` here would force a per-step host sync and defeat
    residency — do not do it.

### 2.2 `DeviceNeighborResult` (runtime-checkable Protocol)

Handle to on-device neighbour data; arrays exported lazily via DLPack.

- `n_edges -> int` — logical edge count (<= `max_capacity` if padded).
- `did_overflow -> bool | <device scalar>` — true if a fixed `max_capacity` was exceeded
  (real edge count > capacity). Mirrors JAX-MD's `did_buffer_overflow`. A compiled loop checks
  this and triggers reallocation with a larger capacity rather than silently dropping edges.
  When eager, a Python bool is fine; for in-graph use, expose as a device scalar like
  `needs_rebuild`. NEVER silently truncate on overflow — that is missing-neighbour corruption.
- `padded -> bool`
- `get(quantity: str) -> <DLPack array>` — one of `"i","j","S","D"` as a DLPack-exporting
  device array. Zero-copy view into backend buffers; **valid until the next
  `build_device` on this object**. Document this lifetime explicitly — it is the
  use-after-free trap.
- `mask() -> <DLPack array> | None` — boolean `(max_capacity,)` device array marking valid
  edges when padded, else `None`.

### 2.3 Quantity / layout contract
- `i, j`: integer endpoint indices, sorted by `i`.
- `S`: integer cell shifts, shape `(n_edges, 3)`, row-major.
- `D`: float edge vectors `(n_edges, 3)`, row-major, `D = r[j] - r[i] + S @ cell`.
  Only meaningful for non-autograd consumers unless `differentiable` is `True`.
- Row-major is mandated. Column-major consumers (Julia/Reactant) must account for the
  documented axis convention — call this out so a transpose bug is a contract issue, not a
  silent error.

**HARD REQUIREMENT — quantity string is an UNORDERED request.** The three target backends
disagree on output order: matscipy `"ijdDS"`, Vesin `"ijSd"` (S before d), ALCHEMI dense
or sparse-COO. Therefore `DeviceNeighborResult.get(quantity)` is keyed by NAME, and any
tuple-returning convenience MUST honour the order the caller requested per-call. NEVER
assume positional order across backends. This is the single most likely silent-corruption
bug; test it explicitly (§7).

**DESIGN FORK — batch dimension (flag to a human, do not decide alone).** ALCHEMI and
TorchSim are natively batched ("one GPU, many systems"); matscipy-neighbours and Vesin are
single-system. Options:
  (a) Protocol is single-system; batching is the caller's concern. Simplest; matches the
      two single-system backends; ALCHEMI used one-system-at-a-time loses its batch edge.
  (b) Protocol exposes an OPTIONAL batch capability (a separate marker, e.g.
      `BatchedDeviceNeighborList`) that ALCHEMI/TorchSim implement and others don't.
      Avoids forcing batch semantics on non-batched backends (the lowest-common-denominator
      trap ASE's own calculator-interface proposal warns against) while letting batched
      consumers use their native path.
Lean: (b) — single-system core protocol + optional batched extension. But this is a real
fork; surface it, don't bury it in code.

---

## 3. Stream / synchronisation contract

ASE defines **no stream policy of its own beyond "honour DLPack"**. Producer and consumer
each fulfil the DLPack stream handshake (the `stream` field / `__dlpack__(stream=...)`).
If either ignores it, behaviour is undefined (races showing up as nondeterministic
forces). Document this; do not try to reinvent stream management inside ASE.

Open question to flag, NOT to resolve unilaterally in code: whether ASE should own *any*
device-scalar→host-bool helper (`bool_from_device_scalar`) — doing so edges ASE toward
knowing about device frameworks. Lean: push it to the consumer; the wrapper's
`update_device` is a *reference* implementation device-resident calculators may replace
wholesale. Leave a clearly-marked seam.

---

## 4. Skin wrapper extension

Extend !4163's skin wrapper so it serves both cases from one class. Do not fork it.

- Host backend (no `DeviceNeighborList`): existing host displacement check, unchanged.
- Device backend: `update_device(positions, cell, pbc, *, stream=None, max_capacity=None)`:
  - first call => `build_device`, cache the `DeviceNeighborResult`.
  - subsequent => `needs_rebuild` (device scalar). Eager consumer syncs at a single
    documented point; compiled consumer branches in-graph. Rebuild only when truthy.
  - When reused: nothing leaves the device.

Pseudocode lives in §"Reference skin wrapper" of the design discussion; implement to match
!4163's actual class structure and naming.

---

## 5. Three device backends (build all three; this is the core of the work)

Implement each as a `DeviceNeighborList` in ITS OWN package / adapter module, NOT in ASE.
ASE only holds the protocol. Each discovered via `isinstance`. Build easiest-verified first;
the deliverable is all three behind one protocol.

### 5.1 matscipy-neighbours (backend #1 — you control it)
Over the existing native DLPack device cell list.
- `device` from the CUDA/HIP context the kernel runs on. `differentiable = False`.
- `build_device`: call native device cell list; wrap outputs as `DeviceNeighborResult`.
  Tight output first; `max_capacity` padding second.
- `needs_rebuild`: device-side reduction of max squared displacement vs cached build-time
  positions (kept on device); return device scalar. Reuse the Verlet-skin reference buffer.
- Register as an *optional* capability of the same plugin PR #2 adds. Host (PR #2) and
  device backends share the plugin name, distinguished by capability.
- DOCUMENT LOUDLY: routing a GPU model through the **host** `get_neighbor_list` path gives
  host round-trips every rebuild; residency requires the device capability.

### 5.2 Vesin (backend #2 — Luthaf; real, GPU-capable, torch interface exists)
VERIFIED: has `vesin-torch` (TorchScript C++/Python), ~14% CUDA in tree, benchmarks on
NVIDIA GPU. API: `NeighborList(cutoff, full_list).compute(points, box, periodic,
quantities="ijSd")`. Note `"ijSd"` order (S before d) — see §2.3 unordered requirement.
- VERIFY FIRST (do not assume): does `vesin-torch` return outputs as **device-resident CUDA
  tensors** you can `torch.from_dlpack` / `__dlpack__` without a host copy, or compute on GPU
  and hand back host arrays? This decides whether Vesin satisfies the DEVICE protocol or only
  the host one. Probe `.device` of the returned tensors, or read `vesin-torch` source. If
  device-resident: implement `DeviceNeighborList` (set `differentiable` per whether the
  torch interface produces vectors differentiable w.r.t. positions — verify).
- If host-only today: implement the HOST protocol for Vesin and note in the design note that
  Vesin device residency is pending upstream — still a useful data point.
- BSD-3; arXiv 2508.15704 (metatensor/metatomic). Luthaf is a plausible upstream collaborator
  on the protocol itself; shape the contract so vesin-torch can adopt it natively.

### 5.3 ALCHEMI Toolkit-Ops (backend #3 — NVIDIA; strongest residency fit)
Repo: github.com/NVIDIA/nvalchemi-toolkit-ops · Docs: nvidia.github.io/nvalchemi-toolkit-ops
VERIFIED (v0.3.0): O(N) cell-list neighbour construction; **native PyTorch tensor AND JAX
array support with `torch.compile` and `jax.jit` compatibility**; dense OR sparse-COO output;
batched (one GPU, many systems); built on NVIDIA Warp; two NL variants (naive, cell).
- Best proof of the protocol: `torch.compile`/`jax.jit` compatibility means the neighbour op
  sits INSIDE a compiled loop — exactly the residency the protocol targets. NVIDIA's stated
  motivation is eliminating host-to-device transfer. Same goal, independently built.
- `differentiable`: treat `False` for index/shift output unless vectors are produced in-graph
  — verify against the op's gradient behaviour.
- Batched: this is where the §2.3 batch fork bites. If you adopt optional
  `BatchedDeviceNeighborList`, ALCHEMI implements it; the single-system core wraps batch=1.
  Use ALCHEMI as the concrete driver for deciding the batch design.
- Output-format fork: map sparse-COO to the `i,j,S` contract directly; dense adjacency is a
  different shape. Lean COO-only for the core protocol (matches matscipy/Vesin), dense as a
  backend-specific extra.

Three backends, one protocol: matscipy (author), Vesin (ecosystem), ALCHEMI (hardware
vendor). That spread is the strongest possible argument the abstraction is real and
adoptable — the right framing for the steering committee and the MACE rewrite devs.

### 5.4 (optional, later) Julia/Reactant native list
In-graph builder, `differentiable = True` — closes the loop with the ACE-to-MLIR / Reactant
work. Document the PJRT/DLPack handoff; wire fully only if cheap. NOT required for validation.

### 5.5 TorchMD-Net neighbour engine (backend candidate — strong differentiable exemplar)
`torchmdnet.extensions` (arXiv 2402.17660). VERIFIED: torch Autograd extension, C++/CUDA
forward, naive-O(N^2) AND cell-list-O(N), rectangular + triclinic PBC. Custom backward in
PyTorch with SECOND-derivative support (needed for force-training), so it is a genuine
`differentiable = True` device backend — better than waiting on §5.4 for that property.
Crucially, it already solves CUDA-graph compatibility the way §2.1 `max_capacity` prescribes:
static-shaped I/O via an upper-bound pair count with `-1` padding for unused pairs. This is a
FOURTH independent witness to the padding design — cite it in the design note as prior art
that validates the choice. Good second or third backend to implement.

### 5.6 NNPOps (candidate, caveated — small-system regime)
Fused CUDA kernels incl. spatial-binning neighbour lists (via OpenMM/TorchANI). CAVEAT from
profiling (arXiv 2603.04092): latency-optimised for SMALL neighbourhoods, scales poorly at
larger sizes; really an AEV-acceleration package, not a general NL provider. Mention as a
candidate with the small-system caveat; do not make it a primary target.

### 5.7 Not candidates (record so they are not re-litigated)
- `cuml`/RAPIDS kNN: k-nearest, not cutoff-with-shifts — wrong primitive for PBC NLs.
- Per-paper hand-rolled vectorised cell lists (water-ice, XL-BOMB ELLPACK/COO, etc.): the
  fragmentation the protocol REPLACES, not backends to wrap.
- `Linux-cpp-lisp` ASE→TorchScript gist: NequIP-adjacent, "mostly untested," superseded.

## 5b. Target CONSUMERS (design toward these, they are not backends)
- **TorchSim** (arXiv / AI Sci. 2025): PyTorch-native, BATCHED simulation engine
  (`integrate`/`optimize`/`static` runners; `SimState` = batched tensors). ALCHEMI is already
  integrating into it. This is the device-resident compiled MD loop your edge data feeds. Its
  batched-by-construction `SimState` is the reason to favour the OPTIONAL BATCHED EXTENSION
  (§2.3 option b): to be adopted by the engine winning PyTorch-MLIP-MD, batch support is
  effectively required, not optional. Design the protocol so a TorchSim model step can consume
  a `DeviceNeighborResult` directly.
- **JAX side — `mlip` v2 / `e3j` / JAX-MD** (InstaDeep, arXiv 2605.22698, full release ~June
  2026): JAX/Pallas+CUDA, sits on JAX-MD's `partition.py` neighbour list. Not a drop-in
  backend for this contract, but the JAX-world counterpart; ALCHEMI's JAX-array support is the
  bridge. Note it so the design note isn't torch-only and the protocol's DLPack/JAX path is
  exercised, not just torch.
- **ase-jax** (abhijeetgangan/ase-jax, Apache-2.0, v0.1.0a1 Jun 2026, early/heavy-dev):
  `JaxMDCalculator` wraps a JAX-MD energy function behind an ASE calculator, with JAX-MD's own
  `neighbor_fn` under the hood. This is the JAX-side concrete instance of EXACTLY the pattern
  this protocol formalises — ASE calculator delegating to a device-resident, framework-native
  neighbour list, with ASE seeing only energy/forces at the boundary. TODAY it keeps the NL
  inside JAX-MD (architecture: loop+NL in framework, ASE as shell) rather than sharing it
  through a common ASE contract — i.e. it IS the gap this protocol fills. Use it as the
  CHEAPEST JAX end-to-end residency demo: swap JAX-MD's built-in `neighbor_fn` for edges from
  this protocol (ALCHEMI JAX path, or matscipy-neighbours via `jax.dlpack`) feeding a JAX-MD
  energy function. Far cheaper integration target than `mlip` v2. Gangan is in the
  TorchSim/MLIP orbit — plausible collaborator and design-note prior art.

### JAX-MD neighbour-list design lessons (fold into §2.1 max_capacity)
JAX-MD's `partition.py` uses capacity-based allocation with an OVERFLOW signal
(`did_buffer_overflow`) and reallocation, plus fractional-vs-real coordinates and
minimum-image-only (cutoff < half shortest box height). Two takeaways for the protocol:
- **Overflow flag, not silent truncation.** TorchMD-Net pads with `-1`, ALCHEMI uses fixed
  capacity, JAX-MD signals overflow + reallocates — three independent witnesses that
  fixed-capacity device NLs are universal AND that a `did_overflow` signal is needed. Add an
  overflow flag to `DeviceNeighborResult` (and/or `build_device`) so `max_capacity` exhaustion
  is detectable, not corrupting. A compiled loop checks the flag and triggers reallocation
  exactly as JAX-MD does. THIS IS A CONCRETE §2.1 REFINEMENT.
- Document the minimum-image / cutoff-vs-box-height constraint as a backend-reportable
  limitation (some backends only support cutoff < L/2); consumers must be able to query it.

---

## 7. Tests

- **Equivalence (cross-backend):** for (random + periodic + mixed-pbc) systems, ALL THREE
  device backends' output (brought to host, canonicalised: sort edges by (i,j,S)) must match
  each other AND the host matscipy oracle — integer-exact (i,j,S), float tolerance (D). This
  one test is the heart of the validation: it proves the protocol means the same thing across
  author/ecosystem/vendor backends. Reuse matscipy's repeat-supercell invariance test.
- **Quantity-order independence:** request the same quantities in different orders and across
  backends with different native orders (matscipy `ijdDS`, Vesin `ijSd`); assert `get(name)`
  returns the right array regardless. This catches the §2.3 silent-corruption bug.
- **Skin correctness:** over a short trajectory, results with skin reuse match per-step
  rebuild (no missed edges) for skin large enough; and a too-small skin still never misses
  (rebuild triggers correctly).
- **Residency:** assert no host transfer happens on reused steps (e.g. monkeypatch/spy the
  DLPack-to-host path, or check the device scalar is consumed without a `.cpu()`/sync on
  reuse). This is the test that actually protects the design's purpose.
- **Device-match:** requesting from a consumer on a different device raises (or copies, per
  policy) — never silently corrupts.
- **Padding & overflow:** with `max_capacity`, shape is constant across rebuilds with varying
  real edge counts; `mask()`/`n_edges` consistent. Deliberately undersize `max_capacity` and
  assert `did_overflow` fires (and that NO edges are silently dropped without the flag) — the
  reallocation-trigger path JAX-MD-style.

---

## 8. Staging / branches (DEVICE-FIRST)

1. §2 device protocols as a provisional module on the !4163 branch. No host-path change.
2. **matscipy-neighbours device backend (§5.1)** off PR #2 — your own, fastest to verify.
   Get the residency test (§7) green here first; it's the load-bearing proof.
3. **ALCHEMI backend (§5.3)** — second, because its `torch.compile`/`jax.jit` compatibility
   most directly exercises the compiled-loop residency the protocol exists for, and it forces
   the batch + output-format decisions early while they're cheap to change.
4. **Vesin backend (§5.2)** — third; verify its device-residency status first, implement
   device or host accordingly.
5. Cross-backend equivalence + quantity-order tests (§7) across all three.
6. Design note: motivation §1, THREE-backend validation (author/ecosystem/vendor), and the
   contestable decisions (device-scalar `needs_rebuild` return type; optional `max_capacity`
   padding; batch fork §2.3; output-format COO-vs-dense; whether ASE owns any
   device-scalar→host helper). For the steering committee AND the MACE rewrite devs — the
   interface is the deliverable; adoption by the rewrite is the win.

---

## 9. What to flag back to a human (do not decide alone)
- **Batch dimension** (§2.3): single-system core + optional `BatchedDeviceNeighborList`, or
  batch-aware core? ALCHEMI/TorchSim want batched; matscipy/Vesin/TorchMD-Net are
  single-system. Lean (b) optional extension — but note TorchSim (§5b) is batched-by-
  construction and is the target consumer, so batch support is effectively required for
  adoption, not a nice-to-have. Decide with ALCHEMI + a TorchSim consumer sketch in hand.
- **Output format**: COO-only core (matches matscipy/Vesin) with dense as backend-specific,
  vs dense in the core protocol? ALCHEMI offers both.
- Exact skin threshold convention (skin vs skin/2) — must match !4163.
- Whether `needs_rebuild` should also accept cell changes (NPT) — probably yes; positions
  AND cell deltas trigger rebuild. Confirm against host wrapper's logic.
- The `bool_from_device_scalar` ownership question (§3).
- Whether to expose `D` at all from a non-differentiable device backend, or force consumers
  to build it (lean: expose it, document the autograd caveat). Only the Julia/Reactant
  in-graph backend (§5.4) can honestly set `differentiable = True`.
