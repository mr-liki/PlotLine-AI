"""Step 1: immutable comparison manifest for Model A (frozen) and Model B (rejected step 100)."""

import os
import stat
import sys
from datetime import UTC, datetime

from correction_common import BASE, PRIOR_RUN, dump, file_hashes, model_a_dir, model_b_dir, read
from regsafe_common import frozen_paths, sha256


def main():
    checks, _, _ = frozen_paths()  # raises if the registered frozen hashes changed
    a, b = model_a_dir(), model_b_dir()
    expected_a = next(c for c in checks if "final_candidate" in c["path"])["expected_sha256"]
    assert sha256(a / "model/model.safetensors") == expected_a
    prior = {
        name: sha256(PRIOR_RUN / name)
        for name in (
            "protocol.json",
            "decision.json",
            "checkpoint_gates.json",
            "checkpoint_evals.json",
            "baseline.json",
            "training_info.json",
            "training_log.json",
            "report.md",
        )
    }
    decision = read(PRIOR_RUN / "decision.json")
    manifest = dict(
        created=datetime.now(UTC).isoformat(),
        model_a=dict(
            role="original frozen baseline (default, teacher)",
            dir=str(a),
            registered_sha256=expected_a,
            files=file_hashes(a / "model") | file_hashes(a / "processor"),
        ),
        model_b=dict(
            role="preserved rejected step-100 candidate (warm start)",
            dir=str(b),
            resolved_from=str(PRIOR_RUN / "training_info.json"),
            files=file_hashes(b),
            optimizer_state_saved=any(b.rglob("*optimizer*")) or any(b.rglob("training_state*")),
            historical_decision=decision["status"],
            historical_protocol_sha256=decision["protocol_sha256"],
        ),
        prior_run_records=prior,
    )
    out = BASE / "preservation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "comparison_manifest.json"
    if path.exists():
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        old = read(path)
        assert old["model_a"]["files"] == manifest["model_a"]["files"]
        assert old["model_b"]["files"] == manifest["model_b"]["files"]
        print("manifest exists and matches", path)
        os.chmod(path, stat.S_IREAD)
        return
    dump(path, manifest)
    os.chmod(path, stat.S_IREAD)  # immutable on this filesystem (read-only attribute)
    print(path, sha256(path))


if __name__ == "__main__":
    sys.exit(main())
