"""Observed-mask-only solid-fragment reconciliation; never extrapolates or joins dashes."""


def reconcile(proposals, valid, union_solids=False, nms_iou=0.3, overlap_agreement=0.5):
    selected, events = [], []
    for item in sorted(proposals, key=lambda p: -p["score"]):
        item = {**item, "mask": item["mask"].astype(bool) & valid}
        if not item["mask"].any():
            continue
        merged = False
        for other in selected:
            if other["label"] != item["label"]:
                continue
            intersection = int((item["mask"] & other["mask"]).sum())
            iou = intersection / max(1, (item["mask"] | other["mask"]).sum())
            support = item["support"] & other["support"] & valid
            agreement = intersection / max(
                1, min((item["mask"] & support).sum(), (other["mask"] & support).sum())
            )
            if (
                union_solids
                and item["label"] == 0
                and intersection >= 4
                and agreement >= overlap_agreement
            ):
                other["mask"] = (other["mask"] | item["mask"]) & valid
                other["support"] = other["support"] | item["support"]
                events.append("union_observed_solid_fragments")
                merged = True
                break
            if iou > nms_iou:
                events.append("suppress_duplicate")
                merged = True
                break
        if not merged:
            selected.append(item)
    return selected, events
