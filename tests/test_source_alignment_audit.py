"""Bounded metadata reads must not treat LAS sizes as LAZ payload sizes."""

import laspy
import numpy as np
import pytest
from scripts.audit_annotation_sources import header_record


@pytest.mark.parametrize("extension", ["las", "laz"])
def test_las_and_laz_headers_are_valid_without_reading_points(tmp_path, extension):
    data = laspy.create(point_format=6, file_version="1.4")
    data.x = np.linspace(1000, 1001, 100)
    data.y = np.linspace(2000, 2001, 100)
    data.z = np.full(100, 10)
    path = tmp_path / f"SYNTHETIC.{extension}"
    data.write(path)
    record = header_record(path)
    assert record["valid_header_bounds"]
    assert record["declared_records_fit_file"]
    assert record["compressed"] == (extension == "laz")
    assert record["point_count"] == 100
    assert not record["full_point_integrity_validated"]


@pytest.mark.parametrize("signature,offset", [(b"INDX", 375), (b"LASF", 0), (b"LASF", 2**31)])
def test_malformed_fixed_headers_are_rejected_before_prefetch(tmp_path, signature, offset):
    payload = bytearray(375)
    payload[:4] = signature
    payload[96:100] = offset.to_bytes(4, "little")
    path = tmp_path / "invalid.laz"
    path.write_bytes(payload)
    record = header_record(path)
    assert not record["valid_header_bounds"]
    assert "error" in record
