# Phase 3: LiDAR preprocessing and intensity rasterization

**Current primary output:** [clean grayscale intensity images](phase3_intensity_images.md).
`scripts/run_phase3.py` now defaults to a representative clean-image exporter.
The geometry workflow below requires `--output-mode legacy`; its multi-panel
previews are diagnostics rather than final model-input images.

Implemented and executed on the two Phase-2 intact sources, Track01_3 and Track01_8.
See [the measured delivery report](../outputs/reports/phase3/phase3_report.md) and
[JSON report](../outputs/reports/phase3/phase3_report.json) for counts, runtimes,
quality checks, preview links and current readiness. This is candidate preprocessing:
road semantics require manual confirmation. It does not detect road markings.

## Run

Use the existing activated `roadmarking-ai` environment, or prefix commands with
`micromamba run -n roadmarking-ai`. Do not invoke an unactivated environment Python:
its numerical libraries depend on the environment DLL search path.

```powershell
python scripts/run_phase3.py --output-mode legacy --config configs/default.yaml --smoke --previews
python scripts/run_phase3.py --output-mode legacy --config configs/default.yaml --previews
python scripts/run_phase3.py --output-mode legacy --inputs Job_20260325_0838_Track01_3 --bounds -56543.5 -88894.5 -56503.5 -88854.5 --previews
python scripts/run_phase3.py --output-mode legacy --road-roi path/to/confirmed_native_xy_roi.json --gap-fill --previews
python scripts/run_phase3.py --output-mode legacy --previews --skip-validated
python -m pytest --junitxml=outputs/reports/phase3/pytest.xml
python -m roadmarking.healthcheck --json outputs/reports/phase3/healthcheck.json
python scripts/report_phase3.py
```

The default run processes both intact files; `--inputs` restricts them. Any other
source stem or unexpected point count is rejected. LAS and LAZ streaming are supported
by the reusable strict reader, while the execution gate deliberately permits only the
verified client inputs. Smoke mode defaults to the 40 m Phase-2 representative window
of Track01_3. It bounds processing area and output, but scans the complete source twice
so even region processing never silently ignores corruption elsewhere in the file.
Full native coordinates are retained. Negative bounds work with `--bounds` as shown.

## Contracts and method

`iter_pointcloud_chunks` validates every record and raises on decoding errors, invalid
returns/GPS time, non-finite coordinates or coordinates outside the header's bounds
(with one scale-step tolerance). Positive scales, finite offsets and finite ordered
bounds are required. No coordinate magnitude rule is applied to an unknown native CRS;
a separate coarse-grid memory budget rejects implausible extents. Unlike the Phase-2
recovery reader, it never skips a damaged chunk or incorporates partial recoveries.

Surface extraction uses local coordinates and double-precision exact moments for
1-unit local plane fits. Slope, fit RMS, neighboring elevation discontinuities,
point-to-plane residual and connected-component area determine geometry candidates.
This is conservative OLS plus rejection, not a robust semantic classifier or lowest-Z
filter. Cars and vegetation with mixed height are usually rejected, but smooth nearby
terrain cannot be reliably distinguished from roads using geometry alone. The label
raster explicitly distinguishes 0 rejected, 1 uncertain candidate and 2 ROI-confirmed
smooth surface. Intensity has no effect on geometry selection, so paint intensity does
not itself cause rejection. Mixed-object cells can still omit real road points.

An optional Polygon, MultiPolygon, Feature or FeatureCollection contains coordinates
in the same native XY system. This is an explicitly supplied road-area ROI: its geometry
must be valid. Geometry within it is labeled ROI-confirmed; both points and pixels are
checked at the exact ROI boundary. No coordinates are reprojected or inferred.

Intensity calibration is shared per source/run: exact uint16 point histogram, p1/p99.5
clipped linear contrast with minimum raw span 1024. Raw measurements and saturated-point
counts remain available. It does not correct intensity physics. Local normalization is
not enabled without evidence for a valid spatial correction. A region run is calibrated
on that region and is a separate processing area; do not mix its images with full-file
images as though their normalization were identical. Individual tiles never recalibrate.

The 0.05-unit north-up grid has snapped origins, pixel-center world mapping, local-origin
index arithmetic and a documented 1e-8-pixel snap tolerance at quantization boundaries.
It is provisionally described as 5 cm pending native-unit confirmation. No CRS is assigned
when none is present. All metadata then says `CRS_UNCONFIRMED`.

The raster accumulator uses uint64 sums, uint32 counts, int32 maxima and float32 mean/
normalized layers. Duplicate records are measurements and count separately. Integer
aggregates are invariant to chunk sizes and ordering. Max NoData is -1; floating NoData
is -9999. Mask zero means false, with no separate mask NoData. Count zero means missing,
so measured zero intensity remains distinguishable from missing data.

Conservative fill is optional, default off. An enclosed one-pixel hole requires eight
observed neighbors, confirmed road across its 3 x 3 neighborhood, and normalized
neighbor range at most 0.08. It uses only original measurements and never alters the
observation mask. One-pixel halos ensure block seams agree. Larger holes and contrast
boundaries remain empty. Without a confirmed ROI no pixels qualify for filling.

## Storage and bounded memory

Products live under `data/processed/phase3/{rasters,tiles,metadata,road_masks}` with
source/mode/fingerprint subdirectories. Reports/previews live under
`outputs/reports/phase3/`. Phase-2 products and original LAZs are untouched.

Two strict input scans build a bounded coarse surface model, then select points and
partition them spatially on disk. The selected binary point records preserve exact
float64 XYZ, uint16 intensity, uint64 original source index, and uint8 confidence state;
`source.json` supplies their NumPy dtype descriptor. Packed per-chunk masks map all
original record indices, including rejected records. Use `np.unpackbits` with default
big bit order and truncate to stored `length`. No original point coordinates are modified.

The coarse-grid limit is eight million cells (ten float64 moment arrays plus fitted
arrays). Source chunks are 250,000 records. Fine rasters are 512 x 512 blocks, never a
full 5 cm bounding box. Each point partition must fit the configured 256 MiB cap or
processing fails explicitly. Overlapping 1024-pixel tiles with 256-pixel overlap are
slices of these common blocks; no per-tile resampling occurs. Both block and tile
GeoTIFFs preserve all layers and affines; block NPZs support exact QA readback. A combined
Phase-4 tile manifest is generated by the report script from the selected full runs.

The source/run fingerprint covers source SHA-256, processing module hashes, effective
settings, ROI bytes, resolution, tile geometry, preview request and region bounds.
`--skip-validated` only accepts a completion record if every saved artifact hash still
matches. Configuration/code changes get another namespace. Failed runs do not receive
a completion marker. Existing valid products are preserved. Sources are hashed again
at completion to verify that processing never changed the original bytes.

## Validation and limitations

The new tests include LAS/LAZ chunk mapping, strict failure, raised/sloped road geometry,
nearby smooth terrain, vegetation and roofs, ROI edge precision, raster statistics and
NoData, order/chunk invariance, fill provenance, block halos, deterministic overlapping
windows, partial edges, end-to-end product writing and cache tampering. Every real tile
layer is read back and compared exactly to the master blocks; transforms, count totals,
coverage and provenance are checked. Peak RAM is sampled process RSS, not an exact OS
high-water mark. Source-run timers include final artifact/source hashes; aggregate
report generation is outside those timers. No new dependencies or versions were needed.

Production Phase-4 conversion is pending confirmed road areas and source/annotation
coordinate alignment. The current candidates can support native-grid annotation review.
CRS, native units, vertical datum, sensor/platform, and corrected damaged source files
remain client requests. There is no ground truth to quantify road recall or marking
preservation, and the representative views do not verify every type of road symbol.
