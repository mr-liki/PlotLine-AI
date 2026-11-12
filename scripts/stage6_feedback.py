"""Record the user's Stage-6 review feedback as ignore zones in the reviewer layer.

User-reported missed paint in S6_04 (two faint lane dashes and the uncovered part of one
dash) becomes ignore zones: never target, never background. No label is drawn from intensity.
This records feedback only; it is NOT an approval.
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

SCREENSHOT = Path(
    r"C:\Users\nt01173\AppData\Local\Temp\claude\cb7f5e780d01fe8f"
    r"\5091df0b-1ee8-4746-9517-f840308f94ea\images\4.png"
)
USER_TEXT = (
    "See I checked Stage 6 QGIS project, only in one image , the annotations got "
    "missed , only for few parts"
)
# (row0, row1, col0, col1, pad, reason) in S6_04 native pixels, from the faint-stroke search.
ZONES = [
    (556, 571, 257, 299, 25, "Faint lane dash without client outline (stroke 0)"),
    (665, 679, 541, 572, 25, "Faint lane dash without client outline (stroke 3)"),
    (597, 661, 552, 708, 12, "Bright paint beside and beyond an outlined dash (box 2)"),
]


def main():
    latest = load(ROOT / "outputs/reports/stage6_review/latest.json")
    out = Path(latest["report_dir"])
    spec = load(out / "new_region_candidates.json")
    tile = next(t for t in spec["tiles"] if t["candidate_id"] == "S6_04")
    affine = Affine(*tile["affine"])
    rows, records = [], []
    for r0, r1, c0, c1, pad, reason in ZONES:
        r0, c0, r1, c1 = (
            max(0, r0 - pad),
            max(0, c0 - pad),
            min(1024, r1 + pad),
            min(1024, c1 + pad),
        )
        x0, y1 = affine * (c0, r0)
        x1, y0 = affine * (c1, r1)
        geom = box(x0, y0, x1, y1)
        rows.append(
            {
                "candidate_id": "S6_04",
                "issue": reason,
                "note": "Drafted from the user's Stage-6 screenshot; ignore, not background",
                "geometry": geom,
            }
        )
        records.append(
            {"pixel_window": [r0, r1, c0, c1], "native_bounds": list(geom.bounds), "reason": reason}
        )
    edits = spec["review_edits_store"]
    pyogrio.write_dataframe(
        gpd.GeoDataFrame(rows, geometry="geometry", crs=None),
        edits,
        layer="reviewer_missing_paint",
        driver="GPKG",
        promote_to_multi=True,
        append=False,
    )
    paint = gpd.read_file(spec["candidate_store"], layer="candidate_paint_footprints")
    with rasterio.open(tile["layers"]["normalized"]) as ds:
        gray, transform = np.clip(ds.read(1), 0, 1), ds.transform
    outline = np.zeros(gray.shape, bool)
    for g in paint[paint.candidate_id == "S6_04"].geometry:
        outline |= rasterize([(g.boundary, 1)], gray.shape, transform=transform).astype(bool)
    img = np.repeat(gray[..., None], 3, -1)
    img[outline] = (0.2, 0.5, 1.0)
    for rec_ in records:
        r0, r1, c0, c1 = rec_["pixel_window"]
        img[r0:r1, [c0, c1 - 1]] = (1, 0.55, 0)
        img[[r0, r1 - 1], c0:c1] = (1, 0.55, 0)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(img[480:760, 150:760])
    ax.set_title("S6_04: user-reported missed paint -> IGNORE zones (orange); outlines blue")
    ax.axis("off")
    fig.tight_layout()
    (out / "galleries").mkdir(exist_ok=True)
    fig.savefig(out / "galleries" / "stage6_ignore_zones.png", dpi=90)
    plt.close(fig)
    dump(
        out / "review_feedback.json",
        {
            "status": "REVIEW_FEEDBACK_RECORDED_APPROVAL_NOT_YET_RECORDED",
            "recorded_utc": datetime.now(UTC).isoformat(),
            "user_message_verbatim": USER_TEXT,
            "screenshot_sha256": sha256(SCREENSHOT),
            "matched_region": "S6_04",
            "ignore_zones": records,
            "review_edits_sha256_after": sha256(edits),
            "candidate_store_sha256_unchanged": sha256(spec["candidate_store"])
            == spec["candidate_store_sha256"],
            "guardrail_note": "The bright line along the road edge in S6_04 is the guardrail, not "
            "paint; it remains observed background.",
            "pending": "Explicit approval of the 10 regions with these ignore zones, and the role "
            "of S6_01 (fresh development vs training).",
        },
    )
    print(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
