"""Record the user's Phase 5.11 review feedback and add IGNORE zones for missed paint.

Visible paint that the client annotation does not outline must never become background.
Each reported spot becomes an ignore zone (neither target nor background). No label is
drawn from intensity. This records feedback only; it is NOT an approval attestation.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pyogrio
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCREENSHOTS = Path(
    r"C:\Users\nt01173\AppData\Local\Temp\claude\cb7f5e780d01fe8f"
    r"\5091df0b-1ee8-4746-9517-f840308f94ea\images"
)
USER_TEXT = (
    "See the attached images , in these areas the indentification is missed , part from that "
    ", everything else is perfect , I can say its recongnizing 99% , but still 1% missing"
)


def main():
    latest = load(ROOT / "outputs/reports/phase5_11/latest.json")
    out = Path(latest["report_dir"])
    spec = load(out / "new_region_candidates.json")
    tiles = {t["candidate_id"]: t for t in spec["tiles"]}
    audit = load(out / "unannotated_paint_audit.json")["candidates"]
    stop = [
        c
        for c in audit
        if c["region"] == "P511_08"
        and min(abs(c["centre_native"][0] - 3409.6), abs(c["centre_native"][0] - 3410.3)) < 2
        and abs(c["centre_native"][1] - 177211.7) < 2
    ]
    rows0 = min(c["bbox_px"][0] for c in stop)
    cols0 = min(c["bbox_px"][1] for c in stop)
    rows1 = max(c["bbox_px"][0] + c["bbox_px"][2] for c in stop)
    cols1 = max(c["bbox_px"][1] + c["bbox_px"][3] for c in stop)
    # (region, row0, row1, col0, col1, pad, screenshot, reason, measured evidence)
    zones = [
        (
            "P511_05",
            313,
            333,
            130,
            239,
            10,
            "1.png",
            "Unoutlined bright gore-nose paint beyond the chevron hatching",
            "two uncovered bright components, 408 observed px",
        ),
        (
            "P511_07",
            592,
            642,
            873,
            931,
            15,
            "2.png",
            "Faint lane dash without client outline (adjacent to vehicle shadow)",
            "one uncovered local-contrast component, 174 observed px",
        ),
        (
            "P511_08",
            rows0,
            rows1,
            cols0,
            cols1,
            60,
            "3.png",
            "STOP lettering and transverse stop bar (unsupported, not in client data)",
            f"{len(stop)} uncovered bright components from the unannotated-paint audit",
        ),
    ]
    records, rows = [], []
    for region, r0, r1, c0, c1, pad, shot, reason, evidence in zones:
        affine = Affine(*tiles[region]["affine"])
        r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
        r1, c1 = min(1024, r1 + pad), min(1024, c1 + pad)
        x0, y1 = affine * (c0, r0)
        x1, y0 = affine * (c1, r1)
        geom = box(x0, y0, x1, y1)
        rows.append(
            {
                "candidate_id": region,
                "reason": reason,
                "user_reported": 1,
                "policy": "IGNORE_NOT_TARGET_NOT_BACKGROUND",
                "geometry": geom,
            }
        )
        records.append(
            {
                "region": region,
                "pixel_window": [r0, r1, c0, c1],
                "native_bounds": list(geom.bounds),
                "area_native": geom.area,
                "reason": reason,
                "evidence": evidence,
                "export_rule": "validity &= ~(zone & ~approved_target): ignore only "
                "non-target pixels; client-outlined targets inside stay supervised",
                "screenshot": str(SCREENSHOTS / shot),
                "screenshot_sha256": sha256(SCREENSHOTS / shot),
            }
        )
    store = spec["candidate_store"]
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(rows, geometry="geometry", crs=None),
        store,
        layer="review_ignore_zones",
        driver="GPKG",
        promote_to_multi=True,
        append=False,
    )
    # Confirmation crops: actual intensity, client outlines (blue), ignore zone (orange).
    paint = gpd.read_file(store, layer="candidate_paint_footprints")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, rec in zip(axes, records, strict=True):
        tile = tiles[rec["region"]]
        with rasterio.open(tile["layers"]["normalized"]) as ds:
            gray, transform = np.clip(ds.read(1), 0, 1), ds.transform
        outline = rasterize(
            [(g.boundary, 1) for g in paint[paint.candidate_id == rec["region"]].geometry],
            gray.shape,
            transform=transform,
        ).astype(bool)
        r0, r1, c0, c1 = rec["pixel_window"]
        img = np.repeat(gray[..., None], 3, -1)
        img[outline] = (0.2, 0.5, 1.0)
        img[r0:r1, [c0, c1 - 1]] = (1.0, 0.55, 0.0)
        img[[r0, r1 - 1], c0:c1] = (1.0, 0.55, 0.0)
        m = 60
        ax.imshow(img[max(0, r0 - m) : r1 + m, max(0, c0 - m) : c1 + m])
        ax.set_title(f"{rec['region']}: {rec['reason']}", fontsize=8)
        ax.axis("off")
    fig.suptitle("User-reported missed paint -> IGNORE zones (orange); client outlines blue")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "galleries" / "review_ignore_zones.png", dpi=110)
    plt.close(fig)
    feedback = {
        "status": "REVIEW_FEEDBACK_RECORDED_APPROVAL_NOT_YET_RECORDED",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "user_message_verbatim": USER_TEXT,
        "interpretation": "User reports 3 areas where visible paint lacks client outlines and "
        "states the rest is correct. Statements 1, 2 and 4 are not contradicted; statement 3 "
        "fails at the reported spots, which are therefore excluded as ignore zones.",
        "ignore_zones": records,
        "candidate_store_sha256_after": sha256(store),
        "pending": "Explicit confirmation that all 16 regions are approved with these ignore "
        "zones excluded.",
    }
    dump(out / "review_feedback.json", feedback)
    print(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
