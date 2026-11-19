"""Approved code is locked: centerlines (configs/centerline_lock.yaml), closed polygons
(configs/polygon_lock.yaml) and the Plotline-AI application (configs/app_lock.yaml)."""

import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.parametrize("lock", ["centerline_lock.yaml", "polygon_lock.yaml", "app_lock.yaml"])
def test_locked_files_unchanged(lock):
    data = yaml.safe_load((ROOT / "configs" / lock).read_text(encoding="utf-8"))
    changed = [p for p, digest in data["files"].items() if _digest(ROOT / p) != digest]
    assert not changed, f"Locked files changed without updating {lock}: {changed}"
