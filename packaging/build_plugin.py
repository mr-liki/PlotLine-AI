"""Build the NakshaAI-LiDAR plugin ZIP "PlotLine" that carries the complete Plotline-AI app.

Usage: python packaging/build_plugin.py [--runtime <conda env>] [--out dist]

ZIP layout (installed by NakshaAI-LiDAR's Plugin Manager; nothing of the host is changed):
    PlotLine/manifest.json        (first entry: the host takes the first *manifest.json)
    PlotLine/plugin.py, icon.png  (ribbon button; starts Plotline-AI as its own process)
    PlotLine/app/...              (same layout as the Plotline-AI installer: runtime, src,
                                   configs, models, data/raw/annotations, docs, launch.pyw)
"""

import argparse
import json
import os
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("models/mask2former_swin_s/production/v2_realistic_v1_fix_step0500")
SKIP_DIRS = {"__pycache__", "pkgs", ".git"}
SKIP_FILES = {".pyc"}


def walk(src: Path):
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix in SKIP_FILES or (p.parent.name == "conda-meta" and name == "history"):
                continue
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=str(Path.home() / "micromamba/envs/roadmarking-ai"))
    ap.add_argument("--out", default=str(ROOT / "dist"))
    a = ap.parse_args()
    manifest = json.loads((ROOT / "plugin/manifest.json").read_text(encoding="utf-8"))
    out = Path(a.out) / f"PlotLine-plugin-{manifest['version']}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.partial")
    runtime = Path(a.runtime)
    groups = [
        (runtime, "app/runtime"),
        (ROOT / "src", "app/src"),
        (ROOT / "configs", "app/configs"),
        (ROOT / MODEL, f"app/{MODEL.as_posix()}"),
        (ROOT / "data/raw/annotations/Highway", "app/data/raw/annotations/Highway"),
        (ROOT / "data/raw/annotations/Interchange", "app/data/raw/annotations/Interchange"),
    ]
    t0 = time.time()
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        z.write(ROOT / "plugin/manifest.json", "PlotLine/manifest.json")  # must be first
        z.write(ROOT / "plugin/plugin.py", "PlotLine/plugin.py")
        z.write(ROOT / "plugin/icon.png", "PlotLine/icon.png")
        z.write(ROOT / "packaging/launch.pyw", "PlotLine/app/launch.pyw")
        z.write(ROOT / "src/roadmarking/app/assets/plotline_ai.ico", "PlotLine/app/plotline_ai.ico")
        z.write(
            ROOT / "docs/Plotline-AI_User_Guide.md", "PlotLine/app/docs/Plotline-AI_User_Guide.md"
        )
        for src, dst in groups:
            for p in walk(src):
                if src.name in ("Highway", "Interchange") and p.suffix.lower() != ".gpkg":
                    continue  # only the reference GeoPackages (no CAD files)
                z.write(p, f"PlotLine/{dst}/{p.relative_to(src).as_posix()}")
                n += 1
    first = zipfile.ZipFile(tmp).namelist()[0]
    assert first == "PlotLine/manifest.json", first
    tmp.replace(out)
    print(f"{out} ({out.stat().st_size / 2**30:.2f} GB, {n} files, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
