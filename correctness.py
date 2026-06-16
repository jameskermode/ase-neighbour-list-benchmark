"""Equivalence check across neighbour-list backends.

Run BEFORE any timing; the benchmark is gated on this passing -- a fast wrong
answer is worthless.

For each small test system we:

1. Compute the canonical full list (i, j, d, D, S) from every available backend.
2. Verify each backend's own self-consistency: D == positions[j] - positions[i]
   + S @ cell, and d == |D|, to a tight tolerance.
3. Canonicalise each backend's edge set by sorting on (i, j, Sx, Sy, Sz), then
   assert every backend's edge set is identical to the reference backend's.
4. Assert the matched D vectors agree to 1e-10.

On any mismatch we print a minimal diff (a handful of offending edges) and fail
(non-zero exit / pytest assertion).

Usable two ways::

    uv run python correctness.py        # standalone, prints a report
    uv run pytest correctness.py        # as a test
"""

from __future__ import annotations

import sys

import numpy as np

from backends import resolve_backends
from systems import build

#: (system kind, target atom count) -- kept small so the gate is fast.
TEST_SYSTEMS = [("cubic", 500), ("lowsym", 500), ("slab", 500)]
#: Cutoffs to check equivalence at (Angstrom).
TEST_CUTOFFS = [3.0, 5.0]

CONTRACT_TOL = 1e-10   # D == positions[j]-positions[i]+S@cell
AGREE_TOL = 1e-10      # D agreement between backends


# --------------------------------------------------------------------------- #
# Canonicalisation
# --------------------------------------------------------------------------- #
def _edge_keys(i, j, S):
    """Return a structured array of (i, j, Sx, Sy, Sz) and the sort order."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    S = np.asarray(S, dtype=np.int64)
    keys = np.empty(len(i), dtype=[("i", "i8"), ("j", "i8"),
                                   ("sx", "i8"), ("sy", "i8"), ("sz", "i8")])
    keys["i"], keys["j"] = i, j
    keys["sx"], keys["sy"], keys["sz"] = S[:, 0], S[:, 1], S[:, 2]
    order = np.argsort(keys, order=["i", "j", "sx", "sy", "sz"])
    return keys, order


def canonicalise(i, j, d, D, S):
    """Sort an edge list into canonical (i, j, Sx, Sy, Sz) order."""
    keys, order = _edge_keys(i, j, S)
    return {
        "keys": keys[order],
        "i": np.asarray(i)[order],
        "j": np.asarray(j)[order],
        "d": np.asarray(d)[order],
        "D": np.asarray(D)[order],
        "S": np.asarray(S)[order],
    }


def check_self_consistency(atoms, res, name):
    """Verify the contract D == pos[j]-pos[i]+S@cell and d == |D| for one backend."""
    pos = atoms.get_positions()
    cell = np.asarray(atoms.cell)
    expected_D = pos[res["j"]] - pos[res["i"]] + res["S"] @ cell
    dD = np.abs(res["D"] - expected_D).max() if len(res["i"]) else 0.0
    dd = np.abs(res["d"] - np.linalg.norm(res["D"], axis=1)).max() if len(res["i"]) else 0.0
    problems = []
    if dD > CONTRACT_TOL:
        problems.append(f"D contract off by {dD:.2e} (tol {CONTRACT_TOL:.0e})")
    if dd > CONTRACT_TOL:
        problems.append(f"d != |D| by {dd:.2e} (tol {CONTRACT_TOL:.0e})")
    return problems


def _minimal_diff(ref, other, ref_name, other_name, limit=6):
    """Return a short human-readable description of how two edge sets differ."""
    ref_set = {tuple(k) for k in ref["keys"]}
    oth_set = {tuple(k) for k in other["keys"]}
    only_ref = sorted(ref_set - oth_set)[:limit]
    only_oth = sorted(oth_set - ref_set)[:limit]
    lines = [
        f"edge-set mismatch: {ref_name} has {len(ref_set)} edges, "
        f"{other_name} has {len(oth_set)}",
        f"  ({len(ref_set - oth_set)} only in {ref_name}, "
        f"{len(oth_set - ref_set)} only in {other_name})",
    ]
    for k in only_ref:
        lines.append(f"  only in {ref_name}: (i,j,S)={k}")
    for k in only_oth:
        lines.append(f"  only in {other_name}: (i,j,S)={k}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Core comparison
# --------------------------------------------------------------------------- #
def compare_backends(kind, n_target, cutoff, verbose=True):
    """Compare all available backends on one (system, cutoff). Return error list."""
    atoms = build(kind, n_target)
    available = [(b, ok, why) for (b, ok, why) in resolve_backends() if ok]
    skipped = [(b.name, why) for (b, ok, why) in resolve_backends() if not ok]

    if verbose:
        tag = f"[{kind} N={len(atoms)} rc={cutoff}]"
        print(f"{tag} backends: {', '.join(b.name for b, _, _ in available)}"
              + (f"  (skipped: {', '.join(n for n, _ in skipped)})" if skipped else ""))

    if len(available) < 2:
        return [f"{kind}/{cutoff}: need >=2 available backends, have "
                f"{[b.name for b, _, _ in available]}"]

    errors = []
    results = {}
    for b, _, _ in available:
        i, j, d, D, S = b.compute(atoms, cutoff)
        res = canonicalise(i, j, d, D, S)
        results[b.name] = res
        for p in check_self_consistency(atoms, res, b.name):
            errors.append(f"{b.name} on {kind}/{cutoff}: {p}")

    ref_name = available[0][0].name
    ref = results[ref_name]
    for name, res in results.items():
        if name == ref_name:
            continue
        if len(res["keys"]) != len(ref["keys"]) or \
                not np.array_equal(res["keys"], ref["keys"]):
            errors.append(f"{kind}/{cutoff}: " +
                          _minimal_diff(ref, res, ref_name, name))
            continue
        dmax = np.abs(res["D"] - ref["D"]).max() if len(ref["i"]) else 0.0
        if dmax > AGREE_TOL:
            errors.append(f"{kind}/{cutoff}: {name} vs {ref_name} D disagree by "
                          f"{dmax:.2e} (tol {AGREE_TOL:.0e})")
        elif verbose:
            print(f"    {name:12s} == {ref_name}: {len(res['i'])} edges, "
                  f"D max-diff {dmax:.1e}")
    return errors


def run_all(verbose=True):
    """Run every (system, cutoff) check. Return the full error list."""
    all_errors = []
    for kind, n in TEST_SYSTEMS:
        for cutoff in TEST_CUTOFFS:
            all_errors.extend(compare_backends(kind, n, cutoff, verbose=verbose))
    return all_errors


# --------------------------------------------------------------------------- #
# pytest entry points
# --------------------------------------------------------------------------- #
def test_backends_equivalent():
    errors = run_all(verbose=False)
    assert not errors, "neighbour lists disagree:\n" + "\n".join(errors)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main():
    errors = run_all(verbose=True)
    print()
    if errors:
        print("CORRECTNESS FAILED:")
        for e in errors:
            print("  " + e.replace("\n", "\n  "))
        return 1
    print("CORRECTNESS PASSED: all available backends agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
