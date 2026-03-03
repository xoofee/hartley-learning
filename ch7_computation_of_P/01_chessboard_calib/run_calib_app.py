#!/usr/bin/env python3
"""
Run the chessboard calibration app.

Run from this directory so calib_app is on the path:

  cd ch7_computation_of_P/01_chessboard_calib && python run_calib_app.py

Or from repo root:

  python -m ch7_computation_of_P.01_chessboard_calib.run_calib_app

(requires PYTHONPATH to include ch7_computation_of_P)
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
# So that "calib_app" resolves when run as script from 01_chessboard_calib
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from calib_app import run_app

if __name__ == "__main__":
    run_app()
