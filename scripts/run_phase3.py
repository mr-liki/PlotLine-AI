"""Execute Phase 3 on verified intact sources; never recover damaged chunks."""

import argparse
import json
from pathlib import Path

from roadmarking.config import load_config
from roadmarking.rasterization.pipeline import INTACT, dump, process_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--output-mode",
        choices=("intensity", "legacy"),
        default="intensity",
        help="Clean images by default; legacy retains geometry processing",
    )
    p.add_argument("--inputs", nargs="+", help="Exact source stems or file paths")
    p.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    p.add_argument("--smoke", action="store_true", help="40 m representative Track01_3 region")
    p.add_argument("--gap-fill", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--previews", action="store_true")
    p.add_argument("--skip-validated", action="store_true")
    p.add_argument("--road-roi", type=Path, help="Polygon GeoJSON coordinates in native XY")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.output_mode == "intensity":
        from roadmarking.visualization.intensity_pipeline import run

        if args.road_roi is not None or args.gap_fill is True:
            p.error(
                "Clean mode retains context and disables fill. "
                "Use --output-mode legacy for geometry/ROI processing."
            )
        if args.inputs:
            for item in args.inputs:
                path = Path(item)
                if (
                    path.is_file()
                    and path.resolve().parent != cfg.path("data.raw_pointclouds").resolve()
                ):
                    p.error("Use the approved source files in data.raw_pointclouds")
        result = run(
            cfg.root, sources=args.inputs, bounds=args.bounds, limit=1 if args.smoke else None
        )
        print(json.dumps({k: result[k] for k in ("status", "run_id", "output_root", "counts")}))
        return
    inputs = args.inputs or ([next(iter(INTACT))] if args.smoke else list(INTACT))
    results = []
    report_dir = cfg.path("outputs.reports") / "phase3"
    for item in inputs:
        path = Path(item)
        if not path.is_file():
            path = cfg.path("data.raw_pointclouds") / (Path(item).stem + ".laz")
        try:
            fill = cfg.raw["phase3"]["gap_fill"] if args.gap_fill is None else args.gap_fill
            result = process_source(
                cfg,
                path,
                bounds=args.bounds,
                smoke=args.smoke,
                gap_fill=fill,
                previews=args.previews,
                skip_validated=args.skip_validated,
                roi_path=args.road_roi,
            )
            results.append(result)
            print(
                json.dumps(
                    {k: result[k] for k in ("source", "mode", "selected_points", "tiles", "qc")}
                )
            )
        except Exception as exc:
            dump(
                report_dir / "last_failure.json",
                dict(
                    source=str(path),
                    error=f"{type(exc).__name__}: {exc}",
                    completed_results=results,
                ),
            )
            raise
    mode = "smoke" if args.smoke else ("region" if args.bounds else "full")
    dump(report_dir / f"{mode}_run.json", results)


if __name__ == "__main__":
    main()
