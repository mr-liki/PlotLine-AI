"""Report system, Python environment and NVIDIA GPU status.

python scripts/check_environment.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from roadmarking.healthcheck import _run, check_gpu, check_python


def nvidia_smi() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found"
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return res.stdout.strip() or res.stderr.strip()


def main() -> int:
    print(f"nvidia-smi: {nvidia_smi()}")
    free_gb = shutil.disk_usage(".").free / 1024**3
    print(f"free disk (project drive): {free_gb:.1f} GB")
    failed = False
    for name, fn in (("python_environment", check_python), ("gpu", check_gpu)):
        r = _run(name, fn)
        failed |= not r.ok
        print(f"[{r.status.upper():4}] {name}{' - ' + r.message if r.message else ''}")
        for k, v in r.details.items():
            print(f"         {k}: {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
