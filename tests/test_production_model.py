"""Production model registry: consistency, hash verification and rollback entry."""

import pytest
import yaml

from roadmarking.models.production import ProductionModelError, read_registry, resolve


def test_registry_active_and_rollback_entries():
    reg = read_registry()
    assert reg["active"] == "v2_realistic_v1_fix_step0500"
    assert "v1_frozen_step0600" in reg["models"]
    for e in reg["models"].values():
        assert set(e["thresholds_v1"]) == {"0", "1"}
        assert e["classes"] == {0: "longitudinal_solid", 1: "longitudinal_dash"}


@pytest.mark.parametrize("name", ["v2_realistic_v1_fix_step0500", "v1_frozen_step0600"])
def test_registered_files_match_hashes(name):
    spec = resolve(name, verify=False)
    if not (spec.weights / "model.safetensors").is_file():
        pytest.skip("model weights not present in this checkout")
    assert resolve(name).name == name


def test_hash_mismatch_is_refused(tmp_path):
    reg = read_registry()
    reg["models"][reg["active"]]["weights_sha256"] = "0" * 64
    bad = tmp_path / "reg.yaml"
    bad.write_text(yaml.safe_dump(reg), encoding="utf-8")
    spec = resolve(verify=False, path=bad)
    if not (spec.weights / "model.safetensors").is_file():
        pytest.skip("model weights not present in this checkout")
    with pytest.raises(ProductionModelError):
        resolve(path=bad)


def test_unknown_active_is_refused(tmp_path):
    bad = tmp_path / "reg.yaml"
    bad.write_text("active: nope\nmodels: {}\n", encoding="utf-8")
    with pytest.raises(ProductionModelError):
        read_registry(bad)
