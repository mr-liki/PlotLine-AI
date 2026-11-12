"""Conservative longitudinal profile tracking, preserving boundary alternatives.

Strict denotes geometric SUPPORT, never human-confirmed roadway. Search geometry
includes every observed profile segment and disconnected alternatives. A bounded
pilot review window is retained if the alignment does not cover all observations.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from roadmarking.preprocessing.road_cross_sections import SectionSettings


def _strip(section, left, right, half_length):
    center = np.asarray(section["center"])
    normal, tangent = np.asarray(section["normal"]), np.asarray(section["tangent"])
    return Polygon(
        [
            center + left * normal - half_length * tangent,
            center + right * normal - half_length * tangent,
            center + right * normal + half_length * tangent,
            center + left * normal + half_length * tangent,
        ]
    )


def build_boundaries(sections, settings=None):
    cfg = settings or SectionSettings()
    strict, search, uncertain, rejected, insufficient = [], [], [], [], []
    tracks, diagnostics = {}, []
    half_length = min(cfg.spacing_m / 2, cfg.along_half_length_m)
    for section in sections:
        segments = section["segments"]
        previous = tracks.get(section["reference_id"])
        near = [segment for segment in segments if segment["near_reference"]]
        selected = near[0] if near else None
        reasons = []
        if selected and previous:
            station_gap = section["station"] - previous["station"]
            change = max(abs(selected[k] - previous[k]) for k in ("left", "right"))
            allowance = cfg.max_station_width_change_m * max(1, station_gap / cfg.spacing_m)
            if station_gap > 2 * cfg.spacing_m or change > allowance:
                reasons.append("width_or_station_discontinuity_preserved_as_uncertain")
                selected = None
        for segment in segments:
            polygon = _strip(section, segment["left"], segment["right"], half_length)
            search.append(polygon)
            if selected is segment:
                strict.append(polygon)
                # Edges are uncertainty bands, not precise surveyed boundaries.
                for edge in ("left", "right"):
                    uncertain.append(
                        _strip(
                            section,
                            segment[edge] - cfg.bin_m,
                            segment[edge] + cfg.bin_m,
                            cfg.spacing_m / 2,
                        )
                    )
            else:
                uncertain.append(polygon)
        offsets = section["offsets"]
        for offset, count, noise in zip(offsets, section["counts"], section["noise"], strict=True):
            polygon = _strip(
                section, offset - cfg.bin_m / 2, offset + cfg.bin_m / 2, cfg.spacing_m / 2
            )
            if count < section["adaptive_min_points"]:
                insufficient.append(polygon)
            elif noise is not None and noise > cfg.noise_cap_m:
                rejected.append(polygon)
                # Rejected geometry must not erase potential paint observations.
                search.append(polygon)
        if selected:
            tracks[section["reference_id"]] = dict(selected, station=section["station"])
        else:
            # Do not propagate a guessed edge across a missing or ambiguous station.
            tracks.pop(section["reference_id"], None)
            reasons.append("no_supported_reference_surface")
        diagnostics.append(
            {
                "reference_id": section["reference_id"],
                "station": section["station"],
                "width_m": selected["right"] - selected["left"] if selected else None,
                "alternatives": len(segments) - int(selected is not None),
                "uncertainty": reasons or ["smooth_edges_not_semantically_verified"],
            }
        )
    return {
        "strict": unary_union(strict),
        "search": unary_union(search),
        "uncertain": unary_union(uncertain),
        "rejected": unary_union(rejected),
        "insufficient": unary_union(insufficient),
        "diagnostics": diagnostics,
        "review_status": "PROPOSED_UNREVIEWED",
    }
