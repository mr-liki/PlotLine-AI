# Plotline-AI: user guide

Plotline-AI turns a LiDAR point cloud (`.las` / `.laz`) into one GeoPackage (`.gpkg`) with the road markings drawn in 3D. You don't need to run scripts or prepare any images.

## Install

Run `Plotline-AI-1.2-Setup.exe`, and keep `Plotline-AI-1.2-Setup-1.bin` in the same folder.
- **No administrator rights needed:** it installs for the current user into `%LOCALAPPDATA%\Programs\Plotline-AI` (about 8 GB).
- **What it adds:** Start-menu shortcuts, an optional desktop shortcut and an uninstaller.
- **Self-contained:** the installed application brings its own Python runtime (CUDA PyTorch, GDAL, PDAL) and does not use or change any other Python.
- **Requirement:** an NVIDIA GPU with a current driver.
- **First start:** it takes about a minute longer while Python compiles its libraries once.

See `INSTALL.txt` next to the installer.

## Use it inside NakshaAI-LiDAR (PlotLine plugin)

1. In NakshaAI-LiDAR, open **Plugins › Manager** and install `PlotLine-plugin-2.0.zip`.
   - This takes about 10 minutes and replaces an older PlotLine plugin.
2. Click the **PlotLine** button (Custom Plugins). Plotline-AI opens in its own window and works exactly as described below.
   - Nothing is loaded into NakshaAI-LiDAR, and no file of NakshaAI-LiDAR is changed.
3. **NakshaAI-LiDAR pauses during a job.** When you click **Start Processing**, it is paused so the job gets the full CPU and GPU. Its window stays visible but does not respond until the job ends.
   - It resumes automatically when the job finishes, fails or is cancelled.
   - A watchdog also resumes it if Plotline-AI is closed or crashes.

## Start the application

Double-click the **Plotline-AI** shortcut in the Plotline-AI folder (`D:\Plotline-AI`). It runs `Launch_Plotline_AI.bat`; you can copy the shortcut to the desktop.

- The launcher uses the `roadmarking-ai` Python environment. If it is installed somewhere unusual, set the environment variable `PLOTLINE_PYTHON` to its `python.exe`.
- To start from a command line instead: `python -m plotline_ai`, with `src` on `PYTHONPATH`.
- A CUDA GPU (NVIDIA) is required: the accepted recognition model is validated on GPU.
- A typical 35-million-point file takes about 4 minutes; up to 4 parts of the file are processed in parallel. Close other heavy GPU programs (for example other LiDAR AI tools) for the best speed.

## Produce the GeoPackage

1. Click **Browse LAS/LAZ** and choose the point cloud.
2. (Recommended) Click **Add overlapping files** and select the other passes of the same road from the same survey.
   - One file usually covers only part of the road (one driving direction); the overlapping files are combined automatically.
   - On the review areas this raised the share of lines found from 36–86 % to 96–100 %.
3. Click **Save Output As**. The suggested name is `<input>_digitized.gpkg`.
4. Click **Start Processing**. The bar shows the measured progress of each stage:
   - Validating LiDAR → Extracting intensity → Running AI recognition → Digitizing road markings → Recovering 3D coordinates → Exporting GeoPackage → Validating output → Completed.
5. When it finishes, the window shows:
   - the output path;
   - the number of exported features;
   - the coordinate reference;
   - the processing time.

   Click **Open Output Folder** to see the file.

**Cancel Processing** stops at the next safe point. No output file is written, and the input files are never changed.

Only the final `.gpkg` is written to the destination you chose. Temporary files live in `%LOCALAPPDATA%\Plotline-AI\jobs` and are removed when the job ends. A short job log is kept in `%LOCALAPPDATA%\Plotline-AI\logs`.

## What is in the GeoPackage

| Layer | Geometry | Content |
|---|---|---|
| `Highway` | MultiLineString Z | Lane lines on the paint axis. Solid lines are continuous; each dash is its own feature. Fields follow the client's Longitudinal Marks file. |
| `Other Marks` | MultiPolygon Z | Painted areas as closed polygons: arrows, hatch bars, chevrons, colour bands, give-way and framed marks. A framed mark, such as a give-way triangle drawn as a painted border, is exported like the client's: two solid polygons, the outer and the inner outline (`OBS` flags `frame_outer` / `frame_inner`). Fields follow the client's Other Marks file. `COD_RST` is a suggestion. |
| `plotline_run_metadata` | table | Source files and their SHA-256, model and its hash, coordinate reference, and the fields that cannot be determined automatically. |

- Every feature has `COMENTARIO = "Ativo novo"` and an `OBS` note saying it was digitized automatically and is pending review.
- **Not produced:**
  - road text (letters and digits), pedestrian crossings and stop lines (no accepted recognition for these);
  - the client fields `POSICAO`, section codes (`COD_SUBLAN`, `SUBLANCO`, `NO`, `RAMO`), `ID_GRUPO`, and the solid-line subtype (guide line vs continuous line).

## Check the result in Global Mapper

1. **File → Open Data File(s)**, then select the original `.las`/`.laz`.
   - If Global Mapper asks for the projection, use the one stored in the file: PT-TM06/ETRS89 (EPSG:3763) with EGM2008 heights.
2. **File → Open Data File(s)**, then select the `_digitized.gpkg` and load both layers `Highway` and `Other Marks`.
   - Their CRS is EPSG:3763, the same horizontal frame as the point cloud.
3. Zoom to a painted line in the point cloud (colour the points by intensity). The `Highway` lines should lie on the centre of the painted lines.
   - On the pilot, the measured offset from the client's own lines was about 1–2 cm.
4. Open the **3D view**: the lines and polygons sit on the road surface.
   - Their heights come from the LiDAR road surface (EGM2008, metres) and matched the client's 3D lines within a few millimetres.

If a LAS/LAZ file has **no** coordinate reference, Plotline-AI keeps the native coordinates, writes the GeoPackage without a CRS and says so in the result window. In Global Mapper, assign the same projection to both layers.

## Automatic self-check

Every job checks its own result for the mistakes found in earlier reviews before the file is saved:

| Check | What it looks for |
|---|---|
| Unjoined hidden gaps | a solid line left broken where vehicles hid the road from the scanner |
| Ends past paint | a line end running over visible road with no paint |
| Junction paint polygons | the merged paint of converging lines exported as an `Other Marks` polygon |
| Overlapping lines | two lines drawn on top of each other for more than 2 m |
| Excess vertices | a line keeping vertices that change its shape by less than 1 cm |
| Lines without visible paint | a line whose axis is not a bright painted stripe |
| Lines through marks | a short line drawn between or across hatch bars / chevrons |
| Zone outline polygons | a polygon drawn around a hatch zone instead of one polygon per bar |
| Line paint polygons | a thin strip of line paint exported as an area mark |
| Off-lane polygons | an "area mark" outside the lanes (speckle on the verge, tiny one-sided blobs) |
| Marks over lines | an area mark drawn over the paint of a lane line next to it |
| Polygons not covering paint | an area mark whose outline overlaps its own paint below 75 % (IoU) |
| Off-centre lines | a stretch of a line (≥ 2 m) more than 2 cm from the centre of its paint, re-measured on the finished line |

- **Every line is placed on the exact centre of its paint stripe.**
  - Every 25 cm the brightness across the line is measured and averaged over ±1.5 m along it, which fills the gaps between scan lines.
  - The centre is the midpoint of the stripe's two edges.
  - The correction is smoothed over ±10 m, so the line follows curves, wide stripes and faint paint without wobbling.
  - On the test files the median distance to the paint centre is 0.4–0.5 cm, and 90 % of the line length is within 1.0–1.5 cm.

- **A short dash painted right beside a continuous line** is drawn as its own dashed line, not merged into a polygon.
- **An arrow touching a lane line** is cut back at the line. A line end that runs into an arrow is cut back to where it leaves the arrow.
- **Lines at the far edge of the scan**, where the road surface is sparse, are kept. Heights between measured points are interpolated along the line, never beyond them, and these lines are flagged `z_interpolated_between_measured_points`.

Every place reported in review is recorded in `tests/reference/reported_cases.json`. After each change, `python scripts/check_reported_cases.py <outputs…>` re-checks all of them on fresh outputs.

Before export, Plotline-AI also removes these mistakes automatically. Each limit was set so that no line or mark that the client also digitized was removed on the test files:

- **Lines:**
  - a line is removed if its axis is a bright stripe at under 30 % of its visible length, or has paint darker on both sides at under 40 % of it;
  - a short line inside a hatch zone is removed;
  - line ends are trimmed to the last visibly painted point, and isolated bright specks don't count as paint.
- **Area marks:**
  - a mark is removed if under 60 % of it is near bright paint;
  - thin strips of line paint, and junk outside the lanes, are removed;
  - hatch bars that touch each other or an untraced border are split into one polygon per bar.

- Lines joined across a vehicle-hidden gap carry `INFERRED_UNDER_OCCLUSION` in `OBS`. Their height across the gap is interpolated between the two visible ends.
- A gap with **visible** unpainted road is a real break and is never joined.
- If any check finds something, the result window lists how many places to review. The coordinates are in the job log (`self_check`), and the counts are in the `plotline_run_metadata` table.

## Messages you may see

- **"No CUDA GPU is available"**: run on a computer with the NVIDIA GPU and drivers.
- **"…could not be decoded (corrupt LAZ chunks)"**: more than 1 % of the file is unreadable. Re-export or re-copy the file.
- **"The output failed validation…"**: the GeoPackage did not pass the readback checks. Nothing was written to the destination; the log in `%LOCALAPPDATA%\Plotline-AI\logs` has the details.
- **"The selected LAS/LAZ files do not share the same coordinate reference system"**: only combine files of the same survey.

- **Every area mark is fitted to its own paint.** After its shape is chosen (a client template or straight edges), it is moved, turned and stretched slightly to best cover the mark's painted shape. Paint of lane lines next to it is excluded.
- **Centerlines never run inside a mark.** A line lying mostly inside an area mark is removed over the whole file, for example one traced along a give-way frame's border or inside an arrow.
- **Before each release**, run `PLOTLINE_REPORTED_CASES=1 pytest tests/test_app.py -k reported`. It re-runs every file reported in review and fails if any recorded mistake is back.
