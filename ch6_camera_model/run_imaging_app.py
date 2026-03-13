#!/usr/bin/env python3
"""
Run the camera imaging simulation from the refactored camera_model package.

This script does not modify ch6_camera_model/01_imaging_simulation.py.
Run from ch6_camera_model so the camera_model package is on the path:

  cd ch6_camera_model && python run_imaging_app.py

Or run the package as a module from repo root:

  python -m camera_model

(requires: PYTHONPATH=ch6_camera_model or run from ch6_camera_model)
"""
import sys
from pathlib import Path

# Ensure ch6_camera_model is on path so "camera_model" resolves when run as script
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from camera_model import run_app

if __name__ == "__main__":
    run_app()
