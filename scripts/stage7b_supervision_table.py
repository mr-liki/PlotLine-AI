"""Stage 7B: best-achievable fit of 1/4-grid logits to synthetic native lines (software test).

Synthetic shapes are diagnostics only, never training data.
"""

import sys

from phase5_11_common import ROOT
from roadmarking.rasterization.pipeline import dump

sys.path.insert(0, str(ROOT / "tests"))
from test_criterion_supervision_math import _fit, _line


def main():
    rows = []
    for width in (1, 2, 3, 5):
        for offset in (0, 1, 2, 3):
            target = _line(width, offset)
            on = target.flatten() > 0
            row = {"width_px": width, "width_m": width * 0.05, "row_offset_in_cell": offset}
            for name, bilinear in (("nearest", False), ("bilinear", True)):
                loss, prob = _fit(bilinear, target)
                row[name] = {
                    "bce": round(loss, 4),
                    "mean_line_probability": round(float(prob[on].mean()), 3),
                    "line_recall_at_0_5": round(float((prob[on] > 0.5).float().mean()), 3),
                    "background_fp_rate_at_0_5": round(float((prob[~on] > 0.5).float().mean()), 4),
                }
            rows.append(row)
    out = ROOT / "outputs/reports/stage7"
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "7B_thin_line_fit_table.json", {"rows": rows})
    for r in rows:
        print(
            r["width_px"],
            r["row_offset_in_cell"],
            r["nearest"]["line_recall_at_0_5"],
            r["bilinear"]["line_recall_at_0_5"],
        )


if __name__ == "__main__":
    main()
