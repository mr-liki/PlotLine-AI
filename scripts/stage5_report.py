"""Stage 5 report from saved results."""

from phase5_11_common import ROOT, load

OUT = ROOT / "outputs/reports/stage5_selftrain"


def main():
    r = load(OUT / "stage5_results.json")
    gate = r["pseudo_label_gate"]
    lines = [
        "# Stage 5: gated self-training",
        "",
        f"Protocol SHA-256 `{r['protocol_sha256']}`. Status: **{r['status']}**.",
        "",
        "| Teacher (seed, fold) | pseudo crops | accepted instances | precision vs client |",
        "|---|---:|---:|---:|",
    ]
    for k, v in gate.items():
        lines.append(
            f"| {k} | {v['pseudo_crops']} | {v['accepted_instances']} | "
            f"{v['precision_vs_client']:.3f} |"
        )
    lines += [
        "",
        "The first teacher's high-confidence pseudo-labels matched the client outlines at "
        "IoU≥0.5 for only 71.2% of accepted instances, below the predeclared 0.80 target. "
        "Under the protocol no student was trained, and self-training is **not adopted**. "
        "Training on about 29% wrong pseudo-labels would amplify the errors the roadmap "
        "warned about.",
        "The client outlines for these Stage-6 regions are not yet approved and were used "
        "only to measure pseudo-label precision. If self-training is revisited, it should "
        "follow a stronger teacher, such as one using bilinear sampling and stitching, and "
        "must clear the same gate.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
