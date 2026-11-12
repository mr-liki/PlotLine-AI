"""Audit the supplied synthetic text corpus without modifying its sources."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/reports/stage9c"
SOURCE = ROOT / "road_text_lidar_synthetic_500"
FOLDERS = {
    "images": ".png",
    "masks": ".png",
    "clean_masks": ".png",
    "labels": ".txt",
    "metadata": ".json",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def image(path):
    with Image.open(path) as raster:
        if raster.format != "PNG" or raster.mode != "L" or raster.size != (512, 512):
            raise ValueError(f"Unexpected raster format, mode or size: {path}")
        return np.array(raster)


def quantiles(values):
    return [round(float(value), 4) for value in np.percentile(values, (5, 50, 95))]


def inspect(paths, split, name, index):
    intensity, raw, clean_raw = (image(paths[key]) for key in ("images", "masks", "clean_masks"))
    if any(np.any((raster != 0) & (raster != 255)) for raster in (raw, clean_raw)):
        raise ValueError("Nonbinary target")
    visible, clean = raw != 0, clean_raw != 0
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata != index[(split, name)] or metadata["image_size"] != [512, 512]:
        raise ValueError("Combined annotation or dimensions disagree")
    if metadata["synthetic"] is not True or np.any(visible & ~clean):
        raise ValueError("Invalid provenance or visible paint outside intended paint")
    if intensity.min() == intensity.max():
        raise ValueError("Constant intensity raster")
    y, x = np.where(visible)
    bbox = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if x.size else None
    intended_box = Image.fromarray(clean_raw).getbbox()
    inclusive = (
        [intended_box[0], intended_box[1], intended_box[2] - 1, intended_box[3] - 1]
        if intended_box
        else None
    )
    if inclusive != metadata["bbox_xyxy"] or bool(x.size) != metadata["text_present"]:
        raise ValueError(
            "Metadata inclusive box disagrees with intended mask or visible presence differs"
        )
    if bool(metadata["transcription"]) != bool(x.size):
        raise ValueError("Transcription and text presence disagree")
    tokens = paths["labels"].read_text().split()
    if bbox:
        left, top, right, bottom = inclusive
        expected = (
            np.array([(left + right) / 2, (top + bottom) / 2, right - left + 1, bottom - top + 1])
            / 512
        )
        if (
            len(tokens) != 5
            or tokens[0] != "0"
            or not np.allclose(
                expected, [float(value) for value in tokens[1:]], rtol=0, atol=0.000001
            )
        ):
            raise ValueError("YOLO box disagrees with intended mask")
    elif tokens:
        raise ValueError("Negative has YOLO target")
    ignore = clean & ~visible
    background = ~clean
    record = dict(
        name=name,
        split=split,
        text=metadata["transcription"],
        visible_pixels=int(x.size),
        intended_pixels=int(clean.sum()),
        ignore_pixels=int(ignore.sum()),
        foreground_zero_pixels=int(np.sum(visible & (intensity == 0))),
        valid_background_zero_pixels=int(np.sum(background & (intensity == 0))),
        intensity_range=[int(intensity.min()), int(intensity.max())],
        foreground_intensity=quantiles(intensity[visible]) if bbox else None,
        background_intensity=quantiles(intensity[background]),
        visible_components_8=int(label(visible, structure=np.ones((3, 3)))[1]),
        intended_components_8=int(label(clean, structure=np.ones((3, 3)))[1]),
        intended_enclosed_hole_pixels=int(np.sum(binary_fill_holes(clean) & ~clean)),
        visible_touches_border=bool(bbox and (0 in bbox or 512 in bbox)),
        stroke_diameter_proxy_px=(
            quantiles(2 * distance_transform_edt(visible)[visible]) if bbox else None
        ),
        bbox=bbox,
    )
    return record, (intensity, raw, clean_raw, ignore)


def gallery(path, examples):
    sheet = Image.new("RGB", (720, len(examples) * 165 + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for j, title in enumerate(("Intensity", "Visible text", "Intended paint", "Ignore proposal")):
        draw.text((j * 180 + 5, 5), title, fill="black")
    for i, (record, channels) in enumerate(examples):
        y = i * 165 + 24
        for j, channel in enumerate(channels):
            pixels = channel.astype("uint8") * 255 if channel.dtype == bool else channel
            sheet.paste(
                Image.fromarray(pixels).resize((140, 140), Image.Resampling.NEAREST),
                (j * 180 + 5, y),
            )
        draw.text(
            (5, y + 143),
            f"{record['split']} {record['name']} {record['text'] or '(negative)'}",
            fill="black",
        )
    sheet.save(path)


def main():
    (root,) = [path.parent for path in SOURCE.rglob("annotations.json")]
    run = BASE / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    indexed = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    index = {(item["split"], item["file"]): item for item in indexed}
    files = {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    errors, samples, examples = [], [], []
    digests = defaultdict(lambda: defaultdict(list))
    phrases = defaultdict(set)
    seen = set()
    for split in ("train", "val", "test"):
        stems = {
            path.stem
            for key in FOLDERS
            for path in (root / split / key).iterdir()
            if path.is_file()
        }
        for stem in sorted(stems):
            name = stem + ".png"
            seen.add((split, name))
            paths = {key: root / split / key / (stem + suffix) for key, suffix in FOLDERS.items()}
            try:
                record, channels = inspect(paths, split, name, index)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                errors.append({"split": split, "name": name, "error": str(exc)})
                continue
            samples.append(record)
            if record["text"]:
                phrases[record["text"]].add(split)
            for key, channel in zip(("images", "masks", "clean_masks"), channels[:3], strict=True):
                digest = hashlib.sha256(channel.tobytes()).hexdigest()
                digests[key][digest].append([split, name, bool(channel.any())])
            positive_examples = sum(r["split"] == split and bool(r["text"]) for r, _ in examples)
            negative_examples = sum(not r["text"] for r, _ in examples)
            if (record["text"] and positive_examples < 2) or (
                not record["text"] and not negative_examples
            ):
                examples.append((record, channels))
    duplicates = {
        key: [group for group in groups.values() if len(group) > 1 and group[0][2]]
        for key, groups in digests.items()
    }
    empty_duplicates = {
        key: sum(len(group) for group in groups.values() if not group[0][2])
        for key, groups in digests.items()
    }
    split_summary = {}
    for split in ("train", "val", "test"):
        records = [record for record in samples if record["split"] == split]
        split_summary[split] = dict(
            samples=len(records),
            positive=sum(bool(r["text"]) for r in records),
            visible_pixels=sum(r["visible_pixels"] for r in records),
            ignore_pixels=sum(r["ignore_pixels"] for r in records),
        )
    results = dict(
        source=str(root),
        file_count=len(files),
        file_formats=dict(Counter(Path(p).suffix for p in files)),
        expected_samples=500,
        valid_samples=len(samples),
        combined_records=len(indexed),
        missing_combined_records=[list(key) for key in sorted(set(index) - seen)],
        combined_duplicate_records=len(indexed) - len(index),
        splits=split_summary,
        errors=errors,
        decoded_exact_duplicate_nonempty=duplicates,
        empty_rasters_by_type=empty_duplicates,
        phrases_shared_across_splits={
            phrase: sorted(splits) for phrase, splits in phrases.items() if len(splits) > 1
        },
        provenance_fields=sorted(set().union(*(r.keys() for r in indexed))),
        source_file_sha256=files,
        sample_audit=samples,
    )
    dump(run / "synthetic_text_dataset_audit.json", results)
    gallery(run / "synthetic_contact_sheet.png", examples)
    dump(BASE / "latest.json", {"run": str(run), "status": "AUDITED"})
    print(json.dumps({k: results[k] for k in ("valid_samples", "splits", "errors")}, indent=2))


if __name__ == "__main__":
    main()
