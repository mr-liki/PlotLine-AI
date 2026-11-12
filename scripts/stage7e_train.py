"""Stage 7E: train the 2x2 arms (3 seeds each) exactly as the sealed 7C protocol says.

Each run: default warm start, 600 steps, checkpoints 200/400/600 with full resumable state,
then fixed-setting inference of every predeclared checkpoint on the development areas.
Exposure (strata, unique crops/IDs, Stage-6 crops), runtime and GPU memory are recorded.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

from phase5_11_common import ROOT, load
from phase5_11_export import check_phase511
from roadmarking.rasterization.pipeline import dump, sha256
from roadmarking.training.expanded_continuous import run_training
from stage1_scoring import DEFAULT, FROZEN, run_inference
from stage7_common import OUT, S6_DEV

MODELS = ROOT / "models/mask2former_swin_s/stage7"
INFER = OUT / "inference"


def dev_sets():
    p57 = load(ROOT / "outputs/reports/phase5_7_expanded/latest.json")
    m57 = load(Path(p57["dataset"]) / "metadata/dataset_manifest.json")
    p59 = load(ROOT / "outputs/reports/phase5_9/latest.json")
    m59 = load(Path(p59["dataset"]) / "manifest.json")
    return {
        "legacy": (Path(p57["dataset"]), [c for c in m57["crops"] if c["split"] == "val"]),
        "new": (Path(p59["dataset"]), m59["crops"]),
        "s6": (S6_DEV.parent, load(S6_DEV)["crops"]),
    }


def exposure(run_dir, dataset):
    log = load(run_dir / "training_log.json")
    manifest = load(Path(dataset) / "metadata/dataset_manifest.json")
    crops = {c["crop_id"]: c for c in manifest["crops"]}
    seen = [crops[e["crop_id"]] for e in log]
    ids = {u for c in seen for u in c["global_instance_ids"]}
    train_ids = {
        u for c in manifest["crops"] if c["split"] == "train" for u in c["global_instance_ids"]
    }
    return {
        "steps": len(log),
        "strata": dict(Counter(e["sampling_stratum"] for e in log)),
        "unique_crops_seen": len({c["crop_id"] for c in seen}),
        "stage6_crop_draws": sum(c.get("stage") == "6" for c in seen),
        "unique_ids_seen": len(ids),
        "train_ids_total": len(train_ids),
        "id_coverage": len(ids & train_ids) / max(1, len(train_ids)),
        "loss_first_last_50": [
            sum(e["loss"] for e in log[:50]) / 50,
            sum(e["loss"] for e in log[-50:]) / 50,
        ],
        "max_gradient_norm": max(e["gradient_norm"] for e in log),
        "nonfinite_losses": sum(not math.isfinite(e["loss"]) for e in log),
        "peak_allocated_bytes": max(e["peak_allocated_bytes"] for e in log),
        "seconds_total_steps": sum(e["seconds"] for e in log),
    }


def main():
    protocol_path = OUT / "7C_protocol.json"
    protocol = load(protocol_path)
    for path, digest in protocol["code_sha256"].items():
        assert sha256(ROOT / path) == digest, f"code changed since protocol: {path}"
    shared = protocol["shared"]
    only = set(sys.argv[1:])
    sets = dev_sets()
    for arm, spec in protocol["arms"].items():
        dataset = Path(spec["dataset"])
        assert (
            sha256(dataset / "metadata/dataset_manifest.json") == protocol["datasets_sha256"][arm]
        )
        for seed in shared["seeds"]:
            name = f"{arm}_s{seed}"
            if only and name not in only:
                continue
            run_dir = MODELS / name
            final = run_dir / f"checkpoint_step{shared['steps']:04d}"
            if not (final / "bundle.json").exists():
                if run_dir.exists():
                    raise SystemExit(f"{name}: incomplete run directory; resume manually")
                run_training(
                    dataset,
                    DEFAULT / "checkpoint_step0050/model",
                    DEFAULT / "checkpoint_step0050/processor",
                    run_dir,
                    FROZEN,
                    max_steps=shared["steps"],
                    stop_after=shared["steps"],
                    learning_rate=shared["optimizer"]["AdamW_lr"],
                    sampling_cycle=list(shared["sampling_cycle"]),
                    dataset_check=check_phase511,
                    evaluation_interval=200,
                    initial_loss_scale=shared["optimizer"]["fp16_initial_scale"],
                    expected_baseline_counts={"fp": 132, "fn": 11},
                    extra_pools={
                        "p511_solid": lambda c: c.get("phase") in ("5.11", "5.12")
                        and 0 in c["labels"],
                        "p511_dash": lambda c: c.get("phase") in ("5.11", "5.12")
                        and 1 in c["labels"],
                    },
                    seed=seed,
                    criterion_kwargs=shared["fix_on"] if spec["bilinear_fix"] else None,
                )
            for step in protocol["shared"]["checkpoints_saved"][1:]:
                ckpt = run_dir / f"checkpoint_step{step:04d}"
                for key, (root, crops) in sets.items():
                    run_inference(ckpt, root, crops, INFER / name / f"step{step:04d}" / key)
            segment = load(run_dir / f"segment_to{shared['steps']}.json")
            record = {
                "run": name,
                "arm": arm,
                "seed": seed,
                "weights_sha256": {
                    s: load(run_dir / f"checkpoint_step{s:04d}/bundle.json")["weight_sha256"]
                    for s in shared["checkpoints_saved"]
                },
                "segment_elapsed_seconds": segment["segment_elapsed_seconds"],
                "cuda_peak_allocated_bytes": segment["cuda_peak_allocated_bytes"],
                "cuda_peak_reserved_bytes": segment["cuda_peak_reserved_bytes"],
                "exposure": exposure(run_dir, dataset),
                "training_config_sha256": sha256(run_dir / "training_config.json"),
            }
            dump(OUT / "runs" / f"{name}.json", record)
            print(json.dumps({k: record[k] for k in ("run", "segment_elapsed_seconds")}))


if __name__ == "__main__":
    main()
