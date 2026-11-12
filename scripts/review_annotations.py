"""Record an explicit human review, or export/import a provenance-preserving snapshot."""

import argparse
import json
from pathlib import Path

from roadmarking.dataset.annotation_store import STATUSES, record_review, snapshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations")
    parser.add_argument("--annotation-id")
    parser.add_argument("--reviewer-id")
    parser.add_argument("--status", choices=STATUSES)
    parser.add_argument("--checklist", type=Path)
    parser.add_argument("--snapshot-to", type=Path)
    args = parser.parse_args()
    if args.snapshot_to:
        print(snapshot(args.annotations, args.snapshot_to))
    else:
        if not all((args.annotation_id, args.reviewer_id, args.status)):
            parser.error("Human review requires annotation-id, reviewer-id and status")
        checklist = json.loads(args.checklist.read_text()) if args.checklist else {}
        print(
            record_review(
                args.annotations, args.annotation_id, args.reviewer_id, args.status, checklist
            )
        )
