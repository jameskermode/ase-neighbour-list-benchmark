# GPU handover (run on an NVIDIA / AMD machine)

Paste the prompt below into a fresh Claude Code session on the GPU machine. It is
self-contained. Context: this benchmark and `matscipy-neighbours` were developed
on an Apple/Metal Mac where the GPU path can't run (the package's GPU backend is
**CUDA/HIP only — no Metal/MPS**), so the GPU backend has only ever been
*scaffolded* (it auto-skips). This is the first real GPU run.

---

```
Task: build matscipy-neighbours with CUDA and run its GPU neighbour-list tests + the GPU benchmark backend, then report timings vs CPU.

Background: matscipy-neighbours (https://github.com/libAtoms/matscipy-neighbours) is a compiled neighbour-list package with an optional CUDA/HIP GPU backend (reached via CuPy device positions, not an ASE Atoms object). A benchmark repo compares it against ASE/matscipy/vesin. Both were developed on an Apple/Metal Mac where the GPU path can't run (CUDA/HIP only, no Metal), so the GPU backend has only ever been scaffolded (auto-skips). Your job is the first real GPU run.

Setup (use uv; clone the two repos as siblings — the benchmark's pyproject has a relative path source `../matscipy-neighbours`):
  mkdir -p ~/gits && cd ~/gits
  git clone https://github.com/jameskermode/ase-neighbour-list-benchmark
  git clone https://github.com/libAtoms/matscipy-neighbours
  # PR #2 adds the pip packaging + ASE plugin; use that branch until it merges to main:
  (cd matscipy-neighbours && git checkout add-ase-plugin-and-pip-packaging)

Find your GPU arch:  nvidia-smi --query-gpu=compute_cap --format=csv,noheader   (e.g. 8.0 -> use 80)

Build + deps (in ase-neighbour-list-benchmark):
  uv sync                                  # CPU baseline
  uv pip install --no-deps --reinstall \
      -C cmake.define.ENABLE_CUDA=ON \
      -C cmake.define.CMAKE_CUDA_ARCHITECTURES=<ARCH> ../matscipy-neighbours
  uv pip install cupy-cuda12x              # match the installed CUDA toolkit (or cupy-cuda11x)

Run:
  1. Benchmark (the correctness gate copies device results to host and asserts identical edge sets vs ASE before timing):
       uv run python benchmark.py --sizes 4000,32000,108000 --cutoff 5.0 \
         --backends ase,matscipy-neighbours,matscipy-neighbours-gpu --out results/
     Confirm `matscipy-neighbours-gpu` is NOT skipped (it self-skips if CuPy/CUDA/the GPU build is missing - if skipped, debug that first).
  2. Package's own GPU tests (separate CMake build, BUILD_TESTING defaults on):
       cmake -S ../matscipy-neighbours -B ../matscipy-neighbours/build-cuda -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<ARCH>
       cmake --build ../matscipy-neighbours/build-cuda --parallel
       ctest --test-dir ../matscipy-neighbours/build-cuda --output-on-failure        # incl. test_neighbour_list_gpu
       PYTHONPATH=../matscipy-neighbours/build-cuda:../matscipy-neighbours/language_bindings/python \
         uv run pytest ../matscipy-neighbours/tests/test_dlpack.py                    # device DLPack round-trip

Report back: GPU vs CPU build times (and the CPU matscipy-neighbours/vesin/matscipy numbers from the same run for context), whether the correctness gate + ctest + test_dlpack passed, your GPU model/arch + CUDA/CuPy versions, and any GPU-vs-CPU edge-set discrepancies. Note that benchmark GPU timing is end-to-end (host->device positions + device build + device->host result copy).

Then (ask me first): add a GPU row/section to ase-neighbour-list-benchmark/FINDINGS.md and commit. The benchmark repo is mine (jameskermode/ase-neighbour-list-benchmark); the matscipy-neighbours packaging/plugin is in PR libAtoms/matscipy-neighbours#2 - do not push to matscipy-neighbours without asking.
```

---

For AMD/HIP instead of CUDA: swap `ENABLE_CUDA`/`CMAKE_CUDA_ARCHITECTURES` for
`ENABLE_HIP`/`CMAKE_HIP_ARCHITECTURES` (e.g. `gfx90a`) and install the ROCm CuPy
build. See the GPU section of `README.md` for the same recipe.
