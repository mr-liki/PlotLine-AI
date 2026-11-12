"""Targeted correction sprint: shared locations for Model A (frozen) and Model B (step 100)."""

from pathlib import Path

from regsafe_common import BASE as REGSAFE
from regsafe_common import MODELS, ROOT, dump, read, sha256

PRIOR_RUN = REGSAFE / "training/20261112T112000443125Z"
BASE = ROOT / "outputs/reports/correction_sprint"


def model_b_dir():
    """Resolve Model B from the sealed run's own records, never by guessing."""
    info = read(PRIOR_RUN / "training_info.json")
    gates = read(PRIOR_RUN / "checkpoint_gates.json")
    steps = sorted(int(s) for s in gates)
    assert steps == [100], steps
    path = Path(info["candidate_dir"]) / "step0100"
    assert (path / "model/model.safetensors").is_file(), path
    return path


def model_a_dir():
    path = MODELS / "final_candidate/checkpoint_step0600"
    assert (path / "model/model.safetensors").is_file()
    return path


def file_hashes(folder):
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
        for p in sorted(Path(folder).rglob("*"))
        if p.is_file() and p.suffix in {".safetensors", ".json"}
    }


__all__ = ["BASE", "PRIOR_RUN", "ROOT", "dump", "file_hashes", "model_a_dir", "model_b_dir", "read"]
