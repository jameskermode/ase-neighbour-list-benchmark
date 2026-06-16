"""Environment + package-version capture, shared by benchmark and findings."""

from __future__ import annotations

import os
import platform
import subprocess


def cpu_model() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for pkg in ("ase", "matscipy", "vesin", "scipy", "numpy", "matplotlib"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "not installed"
    return out


def thread_env() -> dict[str, str]:
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    )
    return {k: os.environ.get(k, "<unset>") for k in keys}


def collect() -> dict:
    return {
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "logical_cores": os.cpu_count(),
        "python": platform.python_version(),
        "versions": package_versions(),
        "threads": thread_env(),
    }


def format_block(info: dict) -> str:
    lines = [
        f"platform      : {info['platform']}",
        f"cpu           : {info['cpu']}",
        f"logical cores : {info['logical_cores']}",
        f"python        : {info['python']}",
        "versions      : " + ", ".join(f"{k}={v}" for k, v in info["versions"].items()),
        "thread env    : " + ", ".join(f"{k}={v}" for k, v in info["threads"].items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_block(collect()))
