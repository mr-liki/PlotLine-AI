"""Locate the user-reported omission without using predictions or test metrics."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from phase5_11_common import ROOT, load
from roadmarking.rasterization.pipeline import dump

OUT = ROOT / "outputs/reports/stage8b"
SCREEN = Path(
    "C:/Users/nt01173/AppData/Local/Temp/codex-clipboard-7cc46804-1cb8-4fbb-8faa-9b8ef714ba76.png"
)


def main():
    if (OUT / "approval_manifest.json").exists():
        raise ValueError("Approval issued: preserve its bound location/review evidence")
    OUT.mkdir(parents=True, exist_ok=True)
    shot = cv2.imread(str(SCREEN))
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY), None)
    matches = []
    packages = [
        ROOT / "outputs/reports/stage7_new_test/20261107T051050944913Z/new_region_candidates.json",
        Path(load(ROOT / "outputs/reports/stage8_new_solid_review/latest.json")["report_dir"])
        / "new_region_candidates.json",
    ]
    for package in packages:
        spec = load(package)
        for tile in spec["tiles"]:
            path = Path(spec["output"]) / "review_previews" / f"{tile['candidate_id']}.png"
            im = cv2.imread(str(path))
            other_kp, other_des = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
            pairs = cv2.BFMatcher().knnMatch(des, other_des, k=2)
            good = [a for a, b in pairs if a.distance < 0.7 * b.distance]
            if len(good) < 4:
                continue
            a = np.float32([kp[m.queryIdx].pt for m in good])
            b = np.float32([other_kp[m.trainIdx].pt for m in good])
            affine, inliers = cv2.estimateAffinePartial2D(
                a, b, method=cv2.RANSAC, ransacReprojThreshold=2
            )
            if affine is None:
                continue
            residual = np.linalg.norm(np.c_[a, np.ones(len(a))] @ affine.T - b, axis=1)
            matches.append(
                {
                    "region": tile["candidate_id"],
                    "inliers": int(inliers.sum()),
                    "matches": len(good),
                    "median_inlier_residual_px": float(
                        np.median(residual[inliers.ravel().astype(bool)])
                    ),
                    "screenshot_to_preview_affine": affine.tolist(),
                    "preview": str(path),
                    "tile": tile,
                    "manifest": str(package),
                }
            )
    matches.sort(key=lambda r: r["inliers"], reverse=True)
    winner = matches[0]
    assert winner["inliers"] >= 20 and winner["inliers"] > matches[1]["inliers"] * 2
    affine = np.array(winner["screenshot_to_preview_affine"])
    endpoints = np.array([[220, 245, 1], [288, 217, 1]]) @ affine.T
    winner["approximate_dash_endpoints_tile_px"] = endpoints.tolist()
    # QGIS previews are exported at native tile dimensions; establish the exact grid.
    assert (
        Image.open(winner["preview"]).size == Image.open(winner["tile"]["images"]["grayscale"]).size
    )
    dump(
        OUT / "missing_mark_location.json",
        {
            "winner": winner,
            "other_matches": [
                {k: v for k, v in r.items() if k not in ("tile",)} for r in matches[1:]
            ],
            "method": "SIFT feature correspondences plus robust similarity transform against existing QGIS review previews. Screenshot points only locate the vicinity; annotation geometry must come from source observations.",  # noqa: E501
        },
    )
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [SCREEN, *packages]}
    dump(
        OUT / "human_review_acceptance.json",
        {
            "record_type": "USER_REPORTED_REVIEW_ACCEPTANCE_WITH_ONE_EXCEPTION",
            "basis": "Explicit user instruction: PLOTLINE-AI STAGE 8B in this chat; all eight T7 and both S8 regions inspected and all other reviewed marking geometry/classifications accepted.",  # noqa: E501
            "reviewed_regions": [f"T7_{i:02}" for i in range(1, 9)] + ["S8_01", "S8_02"],
            "exception": "One visible dash lacks an outline; correction itself has not been confirmed.",  # noqa: E501
            "new_geometry_approval": "PENDING_FOCUSED_CONFIRMATION",
            "final_hash_bound_approval_manifest": "NOT_ISSUED_PENDING_CORRECTION_CONFIRMATION",
            "evaluation_use_authorized": False,
            "input_sha256": inputs,
            "preservation": "Original review packages and source client files remain unchanged. Acceptance is recorded separately in Stage 8B.",  # noqa: E501
        },
    )
    print(json.dumps({k: v for k, v in winner.items() if k != "tile"}, indent=2))


if __name__ == "__main__":
    main()
