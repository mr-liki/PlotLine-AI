"""Hash-checked local training state including optimizer, scheduler, scaler and RNGs."""

import random
from pathlib import Path

import numpy as np
import torch

from roadmarking.rasterization.pipeline import dump, sha256


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_training_state(path, optimizer, scheduler, scaler, step, config, dataset_sha256):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": "phase5.4-resumable-v1",
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "rng": capture_rng(),
        "config": config,
        "dataset_sha256": dataset_sha256,
    }
    torch.save(state, path)
    dump(
        path.with_suffix(".integrity.json"),
        {"sha256": sha256(path), "step": step, "dataset_sha256": dataset_sha256},
    )


def load_training_state(path, optimizer, scheduler, scaler, dataset_sha256):
    """Load only this project's own saved state, after checksum and dataset checks."""
    import json

    path = Path(path)
    integrity = json.loads(path.with_suffix(".integrity.json").read_text())
    if sha256(path) != integrity["sha256"] or integrity["dataset_sha256"] != dataset_sha256:
        raise ValueError("Checkpoint integrity or dataset version mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["version"] != "phase5.4-resumable-v1" or state["dataset_sha256"] != dataset_sha256:
        raise ValueError("Incompatible saved training state")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state
