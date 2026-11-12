"""Run the Naksha PlotLine tracer as independent paint evidence for Phase 3.1 (read-only).

PlotLine (D:\\lidar_classifer_new\\plugins\\plotline) traces painted edge lines and lane
dashes from intensity with height checks. It shares no code or thresholds with the
Phase 3/3.1 road-surface selection, so its polylines are an independent check of whether
painted markings fall inside each corridor. The NumPy reference rasterizer is used (no
Vulkan). The source LAZ is only read; its SHA-256 is checked before and after.

  python scripts/run_plotline_reference.py Job_20260325_0838_Track01_3 Job_20260325_0838_Track01_8
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLUGIN = Path(r"D:\lidar_classifer_new\plugins\plotline")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(plugin: Path, name: str):
    # Keep the external reference plugin read-only, including its helper imports.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        f"_plotline_private_{name}", plugin / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", nargs="+")
    p.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/reports/phase3_1/reference_plotline")
    args = p.parse_args(argv)
    survey, snapshot, extraction = (
        load(args.plugin, n) for n in ("survey", "snapshot", "extraction")
    )
    gaps, dashes = load(args.plugin, "edge_gaps"), load(args.plugin, "dashes")
    version = json.loads((args.plugin / "manifest.json").read_text())["version"]
    args.out.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        laz = ROOT / "data/raw/pointclouds" / f"{source}.laz"
        before = sha256(laz)
        t = time.perf_counter()
        result = survey.run(
            dict(
                source_file=str(laz),
                rasterizer=None,
                workers=1,
                edge_connector=gaps.connect,
                dash_refiner=partial(dashes.refine, geometry=gaps),
            ),
            snapshot,
            extraction._bridge_occlusions,
            lambda n, text: print(f"{n:3d}% {text}", flush=True),
        )
        if sha256(laz) != before:
            raise RuntimeError(f"{laz} changed while tracing")
        result.update(
            wall_seconds=time.perf_counter() - t,
            plotline_version=version,
            source_sha256=before,
            rasterizer="numpy reference (no Vulkan)",
        )
        (args.out / f"{source}.json").write_text(json.dumps(result, default=float))
        print(
            json.dumps(
                {
                    "source": source,
                    "lines": len(result["lines"]),
                    "seconds": round(result["wall_seconds"], 1),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
