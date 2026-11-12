"""One bounded synthetic-plus-real experiment using reconstructed procedural geometry."""

import csv
import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from stage9d_common import ROOT, dump, frozen_checks, read, sha256, verify_frozen
from stage9e_lib import (
    Sampler,
    development_crops,
    evaluate,
    gate_check,
    real_context,
    train_arm,
    with_root,
)
from stage9e_recipes import LOW, TOL, stop_rule

BASE = ROOT / "outputs/reports/geometry_supervised_sprint"
FOLDER = "roadway_mixed_500_geometry_annotated"
SOURCE = ROOT / FOLDER / FOLDER
HOLDOUT = {"worn_lane", "mixed_intersection", "zebra", "arrow_left", "text"}
SEED = 9076
STEPS = 300
TEACHER = {"mask": 5.0, "class": 2.0}


def load_png(path):
    with Image.open(path) as image:
        result = np.array(image)
    assert result.shape == (512, 512), path
    return result


def prepare(out):
    raw = {p.name: p for p in (ROOT / "roadway_mixed_raw_test_500").rglob("*.png")}
    taxonomy = read(SOURCE / "CLASS_MAP.json")["classes"]
    assert taxonomy["0"] == "lane_dashed" and taxonomy["1"] == "lane_solid"
    with (SOURCE / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        manifest = list(csv.DictReader(stream))
    assert len(manifest) == len({r["image"] for r in manifest}) == len(raw) == 500
    folder = out / "converted"
    folder.mkdir()
    train, dev = [], []
    audit = Counter()
    splits = {"train": Counter(), "development": Counter()}
    for row in manifest:
        name = row["image"]
        cid = Path(name).stem
        image = SOURCE / "images" / name
        assert sha256(image) == sha256(raw[name]), name
        assert row["matched_original_generator"] == "True"
        rgb = load_png(image)
        assert rgb.dtype == np.uint8
        sem, inst, paint = (
            load_png(SOURCE / key / name)
            for key in ("semantic_masks", "instance_masks", "paint_masks")
        )
        annotation = SOURCE / "annotations_json" / f"{cid}.json"
        metadata = read(annotation)
        assert metadata["image"] == name and metadata["width"] == metadata["height"] == 512
        objects = metadata["objects"]
        assert len(objects) == int(row["objects"])
        assert {int(n) for n in np.unique(inst) if n} <= {r["id"] for r in objects}
        valid = (sem == 0) & (inst == 0) & (paint == 0)
        targets, labels = [], []
        for obj in objects:
            cls = obj["class_id"]
            assert taxonomy[str(cls)] == obj["class_name"]
            mask = inst == obj["id"]
            actual_pixels = int(mask.sum())
            if actual_pixels != obj["pixel_count"]:
                audit["json_instance_count_disagreements"] += 1
                audit["json_instance_count_absolute_delta"] += abs(
                    actual_pixels - obj["pixel_count"]
                )
            consistent = mask & (sem == cls + 1) & (paint != 0)
            audit["object_pixel_disagreements_ignored"] += int((mask & ~consistent).sum())
            audit[f"source_{obj['class_name']}"] += 1
            if cls not in (0, 1):
                audit["unsupported_object_pixels_ignored"] += int(mask.sum())
                continue
            if cls == 0:
                # The source records grouped dash sequences. Recover distinct contiguous
                # pieces on pre-dropout paint geometry, ignoring small or ambiguous shards.
                components, n = label(consistent, structure=np.ones((3, 3), bool))
                pieces = (components == i for i in range(1, n + 1))
            else:
                pieces = (consistent,)
            for piece in pieces:
                if piece.sum() < (16 if cls == 0 else 4):
                    audit["small_supported_components_ignored"] += 1
                    continue
                valid |= piece
                targets.append(piece)
                labels.append(1 - cls)
        masks = np.asarray(targets, bool).reshape(len(targets), 512, 512)
        assert valid.any() and not (masks & ~valid).any()
        assert len(labels) < 100
        path = folder / f"{cid}.npz"
        np.savez_compressed(
            path,
            rgb=np.repeat(rgb[..., None], 3, axis=2),
            masks=masks,
            labels=np.asarray(labels, int),
            validity=valid,
        )
        role = "development" if row["source_theme"] in HOLDOUT else "train"
        splits[role].update({"images": 1, "solid": labels.count(0), "dash": labels.count(1)})
        record = dict(
            crop_id=cid,
            size=512,
            labels=labels,
            file=str(path),
            sha256=sha256(path),
            _root=folder,
            geography="synthetic",
            theme=row["source_theme"],
            source_hashes={
                "image": sha256(image),
                "annotation": sha256(annotation),
                **{
                    key: sha256(SOURCE / key / name)
                    for key in ("semantic_masks", "instance_masks", "paint_masks")
                },
            },
            source_supported_pixels=int((sem == 1).sum() + (sem == 2).sum()),
            source_paint_pixels=int((paint != 0).sum()),
            scoring_pixels=int(valid.sum()),
        )
        (dev if role == "development" else train).append(record)
    dump(
        out / "geometry_audit.json",
        dict(
            input=SOURCE.as_posix(),
            source_generator_claim="Vendor README; replay code not supplied",
            originals_hash_match=500,
            counts=dict(audit),
            split_counts=splits,
            train=[{**r, "_root": str(folder)} for r in train],
            development=[{**r, "_root": str(folder)} for r in dev],
        ),
    )
    assert len(train) == 400 and len(dev) == 100
    assert all(splits[role][c] for role in splits for c in ("solid", "dash"))
    return train, dev


class MixedSampler:
    """Four real and two synthetic draws per six steps; balance synthetic line classes."""

    def __init__(self, crops, mode, seed):
        real = [c for c in crops if c["geography"] != "synthetic"]
        synthetic = [c for c in crops if c["geography"] == "synthetic"]
        self.real = Sampler(real, mode, seed)
        self.synthetic = [
            Sampler([c for c in synthetic if cls in c["labels"]], "uniform", seed + cls + 1)
            for cls in (0, 1)
        ]

    def draw(self, step):
        cycle, offset = divmod(step - 1, 6)
        if offset in (2, 5):
            cls = int(offset == 5)
            return f"synthetic_class{cls}", self.synthetic[cls].draw(cycle + 1)[1]
        real_offset = offset - int(offset > 2)
        pool, crop = self.real.draw(cycle * 4 + real_offset + 1)
        return f"real_{pool}", crop

    def sizes(self):
        return {
            **self.real.sizes(),
            **{f"synthetic_class{c}": s.sizes()["all"] for c, s in enumerate(self.synthetic)},
        }


def main():
    torch.set_num_threads(4)
    out = BASE / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    print(out, flush=True)
    checks = frozen_checks()
    verify_frozen(checks)
    syn_train, syn_dev = prepare(out)
    real = real_context()
    real_train = with_root(real["full"], real["dataset"])
    real_dev = development_crops(real["ctx"])
    syn_development = [(c["_root"], c) for c in syn_dev]
    frozen = Path(checks[0]["path"]).parent
    model_dir = ROOT / "models/mask2former_swin_s" / f"geometry_sprint_{out.name}"
    dump(
        out / "protocol.json",
        dict(
            frozen=checks,
            seeds=[SEED],
            steps=STEPS,
            eval_every=100,
            lrs=LOW,
            dataset_roles="500 development-exposed synthetic images, with theme holdouts. "
            "300 remains development-exposed. T7/S8 protected.",
            geometry="Class 1 lane_solid -> 0, class 0 lane_dashed -> 1. Separate pre-dropout "
            "connected dash components of >=16 pixels. Ignore unsupported paint and disagreements.",
            sampler="Four real and two synthetic draws per cycle. Use Stage 9E real cycle.",
            class_only_support="Synthetic supported positives and verified generator background. "
            "Ignore unsupported classes. Draw synthetic crops with supported positives.",
            supervised="Valid-only loss, exact bilinear, soft teacher on valid points.",
            teacher_weights=TEACHER,
            holdout_themes=sorted(HOLDOUT),
            real_dev_ids=[c["crop_id"] for _, c in real_dev],
            historical_real_gates=TOL,
            early_stop="Any real raw/V1 solid or dash pixel F1 drop >0.10 at 100-step checkpoints.",
            synthetic_gate="Mean V1 solid/dash F1 improves by >=0.02 and neither drops by >0.02.",
            selection="Require full budget and pass real, synthetic, confusion, continuity "
            "and reload gates. No promotion on reused development evidence alone.",
            synthetic_nodata="Unavailable. Unknown paint is ignored, never called NoData.",
            deterministic=True,
        ),
    )
    processor = Mask2FormerImageProcessor.from_pretrained(
        frozen.parent / "processor", local_files_only=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        frozen, local_files_only=True
    ).cuda()
    baseline = evaluate(model, processor, real_dev)
    synthetic_baseline = evaluate(model, processor, syn_development)
    synthetic_baseline.pop("nodata", None)
    dump(out / "baseline_real.json", baseline)
    dump(out / "baseline_synthetic.json", synthetic_baseline)
    del model
    info, evaluations, _ = train_arm(
        "geometry_sprint",
        out,
        frozen,
        processor,
        real_dev,
        real_train + syn_train,
        sampler_mode="cycle",
        sampler_factory=MixedSampler,
        lrs=LOW,
        steps=STEPS,
        eval_every=100,
        seed=SEED,
        teacher_weights=TEACHER,
        baseline=baseline,
        stop_rule=stop_rule,
        save_dir=model_dir,
        deterministic=True,
        strict_determinism=True,
    )
    last = max(evaluations)
    gates, real_ok = gate_check(evaluations[last], baseline, TOL)
    model, loading = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_dir / "model",
        local_files_only=True,
        output_loading_info=True,
    )
    assert not any(loading.values())
    assert model.config.id2label == {0: "longitudinal_solid", 1: "longitudinal_dash"}
    assert model.criterion.empty_weight.shape == (3,)
    model = model.cuda()
    synth = evaluate(model, processor, syn_development)
    synth.pop("nodata", None)
    dump(out / "candidate_synthetic.json", synth)
    gain = {
        c: synth["v1"]["pixel"][c]["f1"] - synthetic_baseline["v1"]["pixel"][c]["f1"]
        for c in ("solid", "dash")
    }
    synth_ok = sum(gain.values()) / 2 >= 0.02 and min(gain.values()) >= -0.02
    status = "RECOGNITION_REGRESSION" if not real_ok or last != STEPS else "NO_GENERALIZATION_GAIN"
    if real_ok and last == STEPS and synth_ok:
        status = "REAL_WORLD_VALIDATION_REQUIRED"
    verify_frozen(checks)
    dump(
        out / "decision.json",
        dict(
            status=status,
            last_step=last,
            early_stopped=info["stopped_early_at"] is not None,
            real_gates=gates,
            real_passed=real_ok,
            synthetic_v1_f1_gain=gain,
            synthetic_improvement_gate=synth_ok,
            clean_loader_class_and_buffers_passed=True,
            candidate=str(model_dir),
            promoted=False,
            protected_tests_used=False,
            protocol_sha256=hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest(),
        ),
    )
    print(out, status, flush=True)


if __name__ == "__main__":
    main()
