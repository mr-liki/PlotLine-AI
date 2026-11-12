"""Phase 3 orchestration: strict scans -> sparse partitions -> aligned products."""

from __future__ import annotations

import hashlib
import json
import platform
import threading
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import psutil
import rasterio
from affine import Affine
from rasterio.features import shapes

from roadmarking.io.pointcloud import StrictReadReport, iter_pointcloud_chunks, read_header
from roadmarking.preprocessing.intensity import IntensityCalibration, calibrate_histogram
from roadmarking.preprocessing.road_roi import load_road_roi
from roadmarking.preprocessing.road_surface import (
    CorridorRefiner,
    RefinementCriteria,
    RoadCriteria,
    SurfaceAccumulator,
)
from roadmarking.rasterization.rasterizer import (
    NODATA,
    RasterAccumulator,
    fill_isolated,
    north_up_transform,
    pixel_to_world,
    world_to_pixel,
)
from roadmarking.rasterization.tiler import sparse_tile_origins, tile_id

VERSION = "phase3-v1"
VERSION_V2 = "phase3.1-v2"
INTACT = {"Job_20260325_0838_Track01_3": 30_509_631, "Job_20260325_0838_Track01_8": 32_870_412}
POINT_DTYPE = np.dtype(
    [("xyz", "<f8", (3,)), ("intensity", "<u2"), ("source_index", "<u8"), ("state", "u1")]
)
BLOCK = 512


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class MemoryMonitor:
    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        p = psutil.Process()
        while not self.stop.is_set():
            self.peak = max(self.peak, p.memory_info().rss)
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def write_tif(path, image, transform, crs, *, config_version=VERSION):
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = NODATA if image.dtype.kind == "f" else (-1 if image.dtype.kind == "i" else None)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=image.dtype,
        transform=transform,
        crs=crs,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    ) as ds:
        ds.write(image, 1)
        ds.update_tags(
            CRS_STATUS="SOURCE_CRS" if crs else "CRS_UNCONFIRMED", CONFIG_VERSION=config_version
        )


def region_mask(xyz, bounds):
    if bounds is None:
        return np.ones(len(xyz), bool)
    x0, y0, x1, y1 = bounds
    return (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)


def scan(path, chunk_size, timings, report):
    iterator = iter(iter_pointcloud_chunks(path, chunk_size, report))
    while True:
        t = time.perf_counter()
        try:
            item = next(iterator)
        except StopIteration:
            timings["reading_input_seconds"] += time.perf_counter() - t
            break
        timings["reading_input_seconds"] += time.perf_counter() - t
        yield item


def read_mosaic(block_dir, blocks, row, col, height, width, names):
    defaults = dict(
        maximum=(-1, np.int32),
        mean=(NODATA, np.float32),
        normalized=(NODATA, np.float32),
        count=(0, np.uint32),
        observed=(0, np.uint8),
        saturation=(0, np.uint32),
        road_state=(0, np.uint8),
        road_confidence=(0, np.uint8),
        filled=(NODATA, np.float32),
        filled_mask=(0, np.uint8),
    )
    out = {k: np.full((height, width), defaults[k][0], defaults[k][1]) for k in names}
    for br in range(max(0, row // BLOCK), (row + height - 1) // BLOCK + 1):
        for bc in range(max(0, col // BLOCK), (col + width - 1) // BLOCK + 1):
            if (br, bc) not in blocks:
                continue
            rr, cc = br * BLOCK, bc * BLOCK
            r0, c0 = max(row, rr), max(col, cc)
            r1, c1 = min(row + height, rr + BLOCK), min(col + width, cc + BLOCK)
            if r1 <= r0 or c1 <= c0:
                continue
            with np.load(block_dir / f"b{br}_{bc}.npz") as data:
                for name in names:
                    out[name][r0 - row : r1 - row, c0 - col : c1 - col] = data[name][
                        r0 - rr : r1 - rr, c0 - cc : c1 - cc
                    ]
    return out


def preview(report_dir, source, raw, selected, layers, transform, resolution):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    views = [
        (raw, "Raw all-surface maximum", "gray"),
        (layers["road_state"], "Geometry: 0 rejected / 1 uncertain / 2 ROI", "viridis"),
        (layers["normalized"], "Normalized max (observed only)", "gray"),
        (layers["mean"], "Raw mean (selected candidates)", "gray"),
        (layers["maximum"], "Raw maximum (selected candidates)", "gray"),
        (np.log1p(layers["count"]), "log(1 + point count)", "viridis"),
        (layers["observed"], "Observed mask (raw measurements)", "gray"),
        (layers["filled_mask"], "Interpolated provenance mask", "gray"),
        (layers["filled"], "Final intensity; see provenance", "gray"),
    ]
    h, w = raw.shape
    extent = [transform.c, transform.c + w * resolution, transform.f - h * resolution, transform.f]
    for ax, (arr, title, cmap) in zip(axes.ravel(), views, strict=True):
        invalid = arr < 0
        image = np.ma.array(arr, mask=invalid)
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("magenta")
        if "Raw" in title:
            valid = arr[arr >= 0]
            vmax = float(np.percentile(valid, 99.5)) if len(valid) else 65535
            ax.imshow(
                image,
                cmap=cm,
                extent=extent,
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
        else:
            ax.imshow(image, cmap=cm, extent=extent, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{source} | {resolution*100:g} cm/pixel | CRS_UNCONFIRMED\n"
        "Magenta = no measurements; candidates need manual road review"
    )
    fig.tight_layout()
    fig.savefig(report_dir / "representative_comparison.png", dpi=140)
    plt.close(fig)


def process_source(
    cfg,
    path,
    *,
    bounds=None,
    smoke=False,
    gap_fill=False,
    previews=False,
    skip_validated=False,
    roi_path=None,
    namespace="phase3",
    road_method="v1",
    refinement=None,
    calibration_override=None,
    roi_assert_native_frame=False,
):
    """Process one intact source.

    Phase 3 behaviour is the default (``namespace="phase3"``, ``road_method="v1"``).
    Phase 3.1 uses ``road_method="v2"`` (corridor refinement) and writes to its own
    namespace; ``calibration_override`` keeps a previous run's intensity calibration so
    normalized values remain comparable between versions.
    """
    if road_method not in ("v1", "v2"):
        raise ValueError("road_method must be 'v1' or 'v2'")
    version = VERSION if road_method == "v1" else VERSION_V2
    refine_criteria = RefinementCriteria(**(refinement or {}))
    started = time.perf_counter()
    source = path.stem
    if source not in INTACT:
        raise ValueError("Phase 3 permits only the two Phase-2 verified intact sources")
    header = read_header(path)
    if header.point_count != INTACT[source]:
        raise ValueError("Source count differs from verified Phase 2 input")
    settings = cfg.raw["phase3"].copy()
    if smoke and bounds is None:
        old = json.loads((cfg.path("outputs.reports") / "lidar_inspection_report.json").read_text())
        rec = next(x for x in old["files"] if x["file"] == path.name)
        bounds = rec["previews"][0]["bounds"]
    if bounds is not None and (
        len(bounds) != 4
        or not np.isfinite(bounds).all()
        or bounds[0] >= bounds[2]
        or bounds[1] >= bounds[3]
    ):
        raise ValueError("bounds must be finite xmin ymin xmax ymax")
    resolution = cfg.raw["rasterization"]["ground_sampling_distance_m"]
    tile_size = cfg.raw["tiling"]["tile_size_px"]
    overlap = cfg.raw["tiling"]["tile_overlap_px"]
    code_files = [
        Path(__file__),
        *Path(__file__).parent.parent.glob("preprocessing/*.py"),
        *Path(__file__).parent.glob("*.py"),
        Path(__file__).parent.parent / "io/pointcloud.py",
    ]
    identity = dict(
        version=version,
        namespace=namespace,
        road_method=road_method,
        refinement=asdict(refine_criteria) if road_method == "v2" else None,
        calibration_override=calibration_override,
        roi_assert_native_frame=bool(roi_assert_native_frame),
        source_sha256=sha256(path),
        source_size=path.stat().st_size,
        settings=settings,
        bounds=bounds,
        gap_fill=gap_fill,
        previews=previews,
        resolution=resolution,
        tile_size=tile_size,
        overlap=overlap,
        roi_sha256=sha256(roi_path) if roi_path else None,
        code_sha256=hashlib.sha256(
            "".join(sha256(p) for p in sorted(set(code_files))).encode()
        ).hexdigest(),
    )
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    mode = "smoke" if smoke else ("region" if bounds else "full")
    key = f"{source}/{mode}_{run_id}"
    root = cfg.path("data.processed") / namespace
    dirs = {k: root / k / key for k in ("rasters", "tiles", "metadata", "road_masks")}
    reports = cfg.path("outputs.reports") / namespace / key
    for d in [*dirs.values(), reports]:
        d.mkdir(parents=True, exist_ok=True)
    completion = dirs["metadata"] / "completion.json"
    if skip_validated and completion.exists():
        done = json.loads(completion.read_text())
        if done["identity"] == identity and all(
            Path(p).is_file() and sha256(p) == digest for p, digest in done["artifacts"].items()
        ):
            return json.loads((reports / "processing.json").read_text())
    # Appended partitions must never reuse leftovers from an incomplete run.
    partitions = dirs["road_masks"] / f"point_partitions_{time.time_ns()}"
    partitions.mkdir()
    chunk_size = settings["chunk_size"]
    timings = dict(reading_input_seconds=0.0)
    roi = None
    roi_report = None
    if roi_path:
        roi, roi_report = load_road_roi(
            roi_path, header, assert_native_frame=roi_assert_native_frame
        )
    criteria = RoadCriteria(**settings["road_surface"])
    with MemoryMonitor() as monitor:
        t = time.perf_counter()
        acc = SurfaceAccumulator(header, criteria, bounds)
        first = StrictReadReport()
        for _, cloud in scan(path, chunk_size, timings, first):
            acc.add(cloud.xyz[region_mask(cloud.xyz, bounds)])
        model = acc.finish(roi if road_method == "v1" else None)
        del acc
        timings["road_extraction_seconds"] = (
            time.perf_counter() - t - timings["reading_input_seconds"]
        )
        refined = None
        if road_method == "v2":
            t = time.perf_counter()
            read_before = timings["reading_input_seconds"]
            refiner = CorridorRefiner(model, refine_criteria)
            refine_report = StrictReadReport()
            for _, cloud in scan(path, chunk_size, timings, refine_report):
                refiner.add(cloud.xyz[region_mask(cloud.xyz, bounds)])
            refined = refiner.finish(roi)
            del refiner
            timings["road_refinement_seconds"] = (
                time.perf_counter() - t - (timings["reading_input_seconds"] - read_before)
            )
            if refine_report.points_read != header.point_count:
                raise RuntimeError("Refinement pass did not read every source record")
        surface = refined if refined is not None else model
        coarse_t = north_up_transform(model.spec.x_min, model.spec.y_max, criteria.cell_m)
        write_tif(
            dirs["road_masks"] / "surface_state_1m.tif", surface.state, coarse_t, header.crs_wkt
        )
        if refined is not None:
            write_tif(
                dirs["road_masks"] / "surface_confidence_1m.tif",
                refined.confidence,
                coarse_t,
                header.crs_wkt,
            )
            np.savez_compressed(dirs["road_masks"] / "refinement_model.npz", **refined.arrays())
        np.savez_compressed(
            dirs["road_masks"] / "surface_model.npz",
            state=model.state,
            a=model.a,
            b=model.b,
            z=model.z,
            rms=model.rms,
            count=model.count,
        )
        features = [
            dict(type="Feature", properties={"surface_state": int(value)}, geometry=geometry)
            for geometry, value in shapes(surface.state, mask=surface.state > 0, transform=coarse_t)
        ]
        # GeoJSON without a CRS is conventionally WGS84: use explicit native JSON instead.
        dump(
            dirs["road_masks"] / "candidate_regions.native.json",
            dict(
                coordinate_system="native XY; CRS_UNCONFIRMED",
                features=features,
                grid=asdict(model.spec),
                parameters=asdict(criteria),
            ),
        )
        x0 = float(np.floor(header.mins[0] / resolution) * resolution)
        y0 = float(np.ceil(header.maxs[1] / resolution) * resolution)
        transform = north_up_transform(x0, y0, resolution)
        maxc, maxr = world_to_pixel(transform, header.maxs[0], header.mins[1])
        height, width = int(maxr) + 1, int(maxc) + 1
        hist = np.zeros(65536, np.uint64)
        raw_preview = None
        preview_bounds = bounds
        if previews and preview_bounds is None:
            old = json.loads(
                (cfg.path("outputs.reports") / "lidar_inspection_report.json").read_text()
            )
            preview_bounds = next(x for x in old["files"] if x["file"] == path.name)["previews"][0][
                "bounds"
            ]
        if previews:
            pc, pr = world_to_pixel(transform, preview_bounds[0], preview_bounds[3])
            ph = round((preview_bounds[3] - preview_bounds[1]) / resolution)
            pw = round((preview_bounds[2] - preview_bounds[0]) / resolution)
            if ph * pw > 4_000_000:
                ph, pw = min(ph, 1024), min(pw, 1024)
            raw_preview = RasterAccumulator((ph, pw))
        selected = uncertain = accepted = region_points = roi_excluded_points = 0
        partition_paths = {}
        second = StrictReadReport()
        t = time.perf_counter()
        prior_read = timings["reading_input_seconds"]
        intensity_seconds = 0.0
        for start, cloud in scan(path, chunk_size, timings, second):
            in_region = region_mask(cloud.xyz, bounds)
            state = surface.classify(cloud.xyz)
            state[~in_region] = 0
            if roi is not None:
                import shapely

                outside = ~shapely.contains_xy(roi, cloud.xyz[:, 0], cloud.xyz[:, 1])
                roi_excluded_points += int((outside & in_region).sum())
            # Compact record-aligned selections preserve exact source index mapping.
            np.savez_compressed(
                dirs["road_masks"] / f"selection_{start:010d}.npz",
                source_offset=np.uint64(start),
                length=np.uint32(len(state)),
                candidate=np.packbits(state > 0),
                confirmed=np.packbits(state == 2),
            )
            region_points += int(in_region.sum())
            cols, rows = world_to_pixel(transform, cloud.xyz[:, 0], cloud.xyz[:, 1])
            if previews:
                inside = (cols >= pc) & (cols < pc + pw) & (rows >= pr) & (rows < pr + ph)
                raw_preview.add(cols[inside] - pc, rows[inside] - pr, cloud.intensity[inside])
            keep = state > 0
            selected += int(keep.sum())
            uncertain += int((state == 1).sum())
            accepted += int((state == 2).sum())
            ti = time.perf_counter()
            hist += np.bincount(cloud.intensity[keep], minlength=65536).astype(np.uint64)
            intensity_seconds += time.perf_counter() - ti
            if not keep.any():
                continue
            records = np.empty(int(keep.sum()), POINT_DTYPE)
            records["xyz"] = cloud.xyz[keep]
            records["intensity"] = cloud.intensity[keep]
            records["source_index"] = start + np.flatnonzero(keep).astype(np.uint64)
            records["state"] = state[keep]
            br, bc = rows[keep] // BLOCK, cols[keep] // BLOCK
            keys = br * ((width + BLOCK - 1) // BLOCK) + bc
            order = np.argsort(keys, kind="stable")
            keys, records = keys[order], records[order]
            cuts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
            for lo, hi in pairwise(cuts):
                k = int(keys[lo])
                b = (k // ((width + BLOCK - 1) // BLOCK), k % ((width + BLOCK - 1) // BLOCK))
                target = partitions / f"b{b[0]}_{b[1]}.bin"
                with target.open("ab") as f:
                    records[lo:hi].tofile(f)
                partition_paths[b] = target
        timings["selection_partition_seconds"] = (
            time.perf_counter()
            - t
            - (timings["reading_input_seconds"] - prior_read)
            - intensity_seconds
        )
        if calibration_override is not None:
            calibration = IntensityCalibration(**calibration_override)
        else:
            calibration = calibrate_histogram(hist, **settings["intensity_percentiles"])
        timings["intensity_processing_seconds"] = intensity_seconds
        np.save(reports / "intensity_histogram.npy", hist)
        dump(
            dirs["metadata"] / "source.json",
            dict(
                identity=identity,
                header=dict(
                    scales=header.scales.tolist(),
                    offsets=header.offsets.tolist(),
                    mins=header.mins.tolist(),
                    maxs=header.maxs.tolist(),
                    crs_wkt=header.crs_wkt,
                ),
                transform=list(transform)[:6],
                crs_status="SOURCE_CRS" if header.crs_wkt else "CRS_UNCONFIRMED",
                normalization=calibration.as_dict(),
                point_partition_dtype=POINT_DTYPE.descr,
                selection_bit_order=(
                    "big; bit i maps to source_offset+i; " "only first length bits valid"
                ),
            ),
        )
        block_dir = dirs["rasters"] / "blocks"
        block_dir.mkdir(exist_ok=True)
        blocks = set(partition_paths)
        total_count = 0
        t = time.perf_counter()
        for (br, bc), part in sorted(partition_paths.items()):
            size = part.stat().st_size
            if size > settings["max_partition_bytes"]:
                raise MemoryError(f"Partition {part} exceeds memory budget; reduce block size")
            points = np.fromfile(part, dtype=POINT_DTYPE)
            cols, rows = world_to_pixel(transform, points["xyz"][:, 0], points["xyz"][:, 1])
            raster = RasterAccumulator((BLOCK, BLOCK))
            raster.add(cols - bc * BLOCK, rows - br * BLOCK, points["intensity"])
            layers = raster.layers(calibration)
            cc, rr = np.meshgrid(
                np.arange(bc * BLOCK, (bc + 1) * BLOCK), np.arange(br * BLOCK, (br + 1) * BLOCK)
            )
            x, y = pixel_to_world(transform, cc, rr)
            layers["road_state"] = surface.state_at(x, y).astype(np.uint8)
            layers["road_state"][(rr >= height) | (cc >= width)] = 0
            if refined is not None:
                layers["road_confidence"] = refined.confidence_at(x, y).astype(np.uint8)
                layers["road_confidence"][(rr >= height) | (cc >= width)] = 0
            np.savez_compressed(block_dir / f"b{br}_{bc}.npz", **layers)
            total_count += int(raster.count.sum(dtype=np.uint64))
            del points, raster, layers
        assert total_count == selected, "Partition raster count mismatch"
        # Fill using original neighbor blocks and a halo. Only immutable raw layers
        # are read in this stage, so processing order cannot change results.
        for br, bc in sorted(blocks):
            data = read_mosaic(
                block_dir,
                blocks,
                br * BLOCK - 1,
                bc * BLOCK - 1,
                BLOCK + 2,
                BLOCK + 2,
                ["normalized", "observed", "road_state"],
            )
            filled, provenance = fill_isolated(
                data["normalized"], data["observed"], data["road_state"] == 2, enabled=gap_fill
            )
            with np.load(block_dir / f"b{br}_{bc}.npz") as saved:
                layers = {k: saved[k] for k in saved.files}
            layers["filled"], layers["filled_mask"] = filled[1:-1, 1:-1], provenance[1:-1, 1:-1]
            np.savez_compressed(block_dir / f"b{br}_{bc}.npz", **layers)
            bt = transform @ Affine.translation(bc * BLOCK, br * BLOCK)
            for name, arr in layers.items():
                write_tif(block_dir / f"b{br}_{bc}_{name}.tif", arr, bt, header.crs_wkt)
        timings["rasterization_seconds"] = time.perf_counter() - t
        names = [
            "maximum",
            "mean",
            "count",
            "observed",
            "normalized",
            "saturation",
            "road_state",
            "filled",
            "filled_mask",
        ]
        if refined is not None:
            names.append("road_confidence")
        manifest = []
        t = time.perf_counter()
        for row, col in sparse_tile_origins(blocks, BLOCK, tile_size, overlap):
            h, w = min(tile_size, height - row), min(tile_size, width - col)
            if h <= 0 or w <= 0:
                continue
            layers = read_mosaic(block_dir, blocks, row, col, h, w, names)
            if not layers["observed"].any():
                continue
            tid = tile_id(source, row, col)
            target = dirs["tiles"] / tid
            tt = transform @ Affine.translation(col, row)
            paths = {}
            for name, arr in layers.items():
                file = target / f"{name}.tif"
                write_tif(file, arr, tt, header.crs_wkt)
                paths[name] = str(file)
            pixels = h * w
            road = layers["road_state"] > 0
            meta = dict(
                tile_id=tid,
                source_file=str(path),
                source_sha256=identity["source_sha256"],
                row_offset=row,
                col_offset=col,
                height=h,
                width=w,
                resolution=resolution,
                native_xy_bbox=[tt.c, tt.f - h * resolution, tt.c + w * resolution, tt.f],
                affine=list(tt)[:6],
                crs_status="SOURCE_CRS" if header.crs_wkt else "CRS_UNCONFIRMED",
                observed_pct=100 * int(layers["observed"].sum()) / pixels,
                interpolated_pct=100 * int(layers["filled_mask"].sum()) / pixels,
                road_surface_coverage_pct=100 * int(road.sum()) / pixels,
                confirmed_road_pct=100 * int((layers["road_state"] == 2).sum()) / pixels,
                normalization=calibration.as_dict(),
                configuration_version=version,
                road_extraction_version=road_method,
                run_id=run_id,
                layers=paths,
                phase4_status="NEEDS_ROAD_REVIEW" if uncertain else "ROI_REVIEWED_CANDIDATE",
            )
            dump(target / "metadata.json", meta)
            manifest.append(meta)
        timings["tile_writing_seconds"] = time.perf_counter() - t
        dump(dirs["metadata"] / "tile_manifest.json", manifest)
        # Unique block statistics: tile overlaps must not inflate observation counts.
        counts = dict(observed=0, missing_in_candidate=0, candidate_pixels=0, filled=0)
        for br, bc in sorted(blocks):
            with np.load(block_dir / f"b{br}_{bc}.npz") as a:
                road, obs = a["road_state"] > 0, a["observed"] > 0
                counts["observed"] += int(obs.sum())
                counts["candidate_pixels"] += int(road.sum())
                counts["missing_in_candidate"] += int((road & ~obs).sum())
                counts["filled"] += int(a["filled_mask"].sum())
        qc = validate_products(manifest, block_dir, blocks, transform, selected)
        if previews:
            layers = read_mosaic(block_dir, blocks, int(pr), int(pc), ph, pw, names)
            preview(
                reports,
                source,
                raw_preview.maximum,
                None,
                layers,
                transform @ Affine.translation(int(pc), int(pr)),
                resolution,
            )
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 4))
            bins = np.arange(0, 65536, 256)
            ax.step(bins, hist.reshape(256, 256).sum(axis=1), where="post")
            ax.axvline(calibration.low, color="red")
            ax.axvline(calibration.high, color="red")
            ax.set(xlabel="Raw intensity", ylabel="Selected measured points", title=source)
            fig.tight_layout()
            fig.savefig(reports / "intensity_histogram.png", dpi=140)
            plt.close(fig)
        peak = monitor.peak
    timings["total_processing_seconds"] = time.perf_counter() - started
    result = dict(
        source=source,
        mode=mode,
        run_id=run_id,
        input_valid_points=first.points_read,
        second_scan_valid_points=second.points_read,
        invalid_points=first.invalid_points,
        region_points=region_points,
        selected_points=selected,
        uncertain_points=uncertain,
        confirmed_points=accepted,
        rejected_region_points=region_points - selected,
        candidate_area_native_units2=float((surface.state > 0).sum() * criteria.cell_m**2),
        confirmed_area_native_units2=float((surface.state == 2).sum() * criteria.cell_m**2),
        road_method=road_method,
        configuration_version=version,
        roi=roi_report,
        roi_excluded_points=roi_excluded_points if roi is not None else None,
        refinement=refinement_summary(refined, criteria.cell_m) if refined is not None else None,
        criteria=asdict(criteria),
        geometry_diagnostics=dict(
            candidate_rms_m_percentiles=np.percentile(
                model.rms[surface.state > 0], [5, 50, 95]
            ).tolist(),
            candidate_slope_percentiles=np.percentile(
                np.hypot(model.a, model.b)[surface.state > 0], [5, 50, 95]
            ).tolist(),
            candidate_points_per_cell_percentiles=np.percentile(
                model.count[surface.state > 0], [5, 50, 95]
            ).tolist(),
            fit_method="local cell OLS with conservative residual and neighbor-step rejection",
            semantic_confidence="uncertain unless supplied road ROI",
        ),
        normalization=calibration.as_dict(),
        pixels=counts,
        missing_candidate_pct=100
        * counts["missing_in_candidate"]
        / max(1, counts["candidate_pixels"]),
        filled_candidate_pct=100 * counts["filled"] / max(1, counts["candidate_pixels"]),
        tiles=len(manifest),
        timings=timings,
        peak_sampled_rss_bytes=peak,
        platform=platform.platform(),
        crs_status="CRS_UNCONFIRMED" if not header.crs_wkt else "SOURCE_CRS",
        warnings=first.warnings,
        qc=qc,
        manifest=str(dirs["metadata"] / "tile_manifest.json"),
        reports=str(reports),
        output_directories={k: str(v) for k, v in dirs.items()},
        phase4_ready=False,
        blockers=[
            "Automatic geometry candidates need a confirmed road ROI/review",
            "Horizontal CRS, native units and vertical datum unconfirmed",
        ],
        limitations=[
            "Conservative cell OLS may reject mixed road/object cells",
            "Smooth non-road terrain remains an uncertain candidate",
            "File percentile scaling does not correct range/incidence variations",
            "No annotation ground truth; marking recall is unmeasured",
        ],
    )
    result["output_bytes"] = sum(
        f.stat().st_size for d in [*dirs.values(), reports] for f in d.rglob("*") if f.is_file()
    )
    finalize_started = time.perf_counter()
    # All files are hashed, including masks and selected source-coordinate partitions.
    artifacts = {
        str(f): sha256(f)
        for d in [*dirs.values(), reports]
        for f in d.rglob("*")
        if f.is_file() and f != completion and f != reports / "processing.json"
    }
    if sha256(path) != identity["source_sha256"]:
        raise RuntimeError("Source changed during processing")
    timings["finalization_seconds"] = time.perf_counter() - finalize_started
    timings["total_processing_seconds"] = time.perf_counter() - started
    dump(reports / "processing.json", result)
    artifacts[str(reports / "processing.json")] = sha256(reports / "processing.json")
    dump(completion, dict(identity=identity, artifacts=artifacts, qc=qc))
    return result


def refinement_summary(refined, cell_m):
    from roadmarking.preprocessing.road_surface import CONFIDENCE_NAMES, EVIDENCE_NAMES

    conf = np.bincount(refined.confidence.ravel(), minlength=len(CONFIDENCE_NAMES))
    return dict(
        cell_area_native_units2=cell_m**2,
        confidence_cells={CONFIDENCE_NAMES[i]: int(v) for i, v in enumerate(conf)},
        evidence_cells={
            name: int(((refined.evidence & bit) > 0).sum()) for bit, name in EVIDENCE_NAMES.items()
        },
        core_cells=int(refined.core.sum()),
        edge_refined_cells=int((refined.edge_ok & ~refined.core).sum()),
        criteria=asdict(refined.criteria),
        confidence_semantics="algorithm evidence categories; not calibrated probabilities",
    )


def validate_products(manifest, block_dir, blocks, transform, expected_points):
    """Read back every tile/layer against immutable global blocks, including overlaps."""
    if not manifest:
        raise ValueError("No usable tiles")
    checked = observed = raster_points = 0
    for br, bc in blocks:
        with np.load(block_dir / f"b{br}_{bc}.npz") as data:
            raster_points += int(data["count"].sum(dtype=np.uint64))
    if raster_points != expected_points:
        raise ValueError("Unique raster count mismatch")
    for tile in manifest:
        r, c, h, w = (tile[k] for k in ("row_offset", "col_offset", "height", "width"))
        reference = read_mosaic(block_dir, blocks, r, c, h, w, tile["layers"])
        for name, filename in tile["layers"].items():
            with rasterio.open(filename) as ds:
                image = ds.read(1)
                if not np.array_equal(image, reference[name]):
                    raise ValueError(f"Readback/overlap mismatch: {filename}")
                expected_t = transform @ Affine.translation(c, r)
                if not ds.transform.almost_equals(expected_t, precision=1e-9):
                    raise ValueError("Tile affine drift")
                if tile["crs_status"] == "CRS_UNCONFIRMED" and ds.crs is not None:
                    raise ValueError("Fabricated CRS")
                checked += 1
        obs = reference["count"] > 0
        if not np.array_equal(obs, reference["observed"] > 0):
            raise ValueError("Invented observation")
        fill = reference["filled_mask"] > 0
        if (fill & obs).any() or (fill & (reference["road_state"] != 2)).any():
            raise ValueError("Invalid interpolation provenance")
        if not np.array_equal(reference["filled"][obs], reference["normalized"][obs]):
            raise ValueError("Fill changed observations")
        observed += int(obs.sum())
    # Every nonempty block pixel must be covered by a generated tile.
    for br, bc in blocks:
        covered = np.zeros((BLOCK, BLOCK), bool)
        r0, c0 = br * BLOCK, bc * BLOCK
        for tile in manifest:
            r, c, h, w = (tile[k] for k in ("row_offset", "col_offset", "height", "width"))
            a, b = max(r, r0), max(c, c0)
            d, e = min(r + h, r0 + BLOCK), min(c + w, c0 + BLOCK)
            if d > a and e > b:
                covered[a - r0 : d - r0, b - c0 : e - c0] = True
        with np.load(block_dir / f"b{br}_{bc}.npz") as data:
            if ((data["count"] > 0) & ~covered).any():
                raise ValueError("Observed pixels excluded by tiling")
    return dict(
        passed=True,
        tile_layers_read_back=checked,
        exact_overlap_consistency=True,
        unique_raster_point_count=raster_points,
        all_observations_covered=True,
        coordinate_tolerance_native_units=1e-9,
    )
