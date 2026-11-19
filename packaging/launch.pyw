"""Plotline-AI launcher for the installed application (started by the Start-menu and desktop
shortcuts with the bundled runtime\\pythonw.exe, so no console window opens).

Layout of an installation:
    <app>\\runtime\\   the Python environment (CUDA PyTorch, GDAL, PDAL, ...)
    <app>\\src\\       Plotline-AI code      <app>\\configs\\  settings and locks
    <app>\\models\\    the accepted AI model <app>\\data\\raw\\annotations\\  client reference marks
"""

import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
RUNTIME = APP / "runtime"

# Data and plug-in paths of the bundled GDAL / PROJ (the conda activation would set these).
os.environ["GDAL_DATA"] = str(RUNTIME / "Library" / "share" / "gdal")
os.environ["GDAL_DRIVER_PATH"] = str(RUNTIME / "Library" / "lib" / "gdalplugins")
os.environ["PROJ_DATA"] = str(RUNTIME / "Library" / "share" / "proj")
os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"]
os.environ["PATH"] = os.pathsep.join(
    [
        str(RUNTIME),
        str(RUNTIME / "Library" / "bin"),
        str(RUNTIME / "Scripts"),
        os.environ.get("PATH", ""),
    ]
)
os.environ["PYTHONPATH"] = str(APP / "src")  # also for the parallel worker processes
sys.path.insert(0, str(APP / "src"))
os.chdir(APP)

if __name__ == "__main__":
    # Only the main process opens the window: the parallel workers (multiprocessing "spawn")
    # re-run this file to get the environment above, and must not start the app again.
    from roadmarking.app.gui import main

    main()
