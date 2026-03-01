"""
Camera model package: pinhole camera simulation and imaging demo.

Refactored from ch6_camera_model/01_imaging_simulation.py for scalability,
modularity (Single Responsibility, concept-based split), and maintainability.
New features (binocular, multi-camera, epipolar, angle-by-K) can be added via
the plugin registry without changing core code (Open-Closed principle).

Usage:
  from camera_model import run_app
  run_app()

  # Or run as script from ch6_camera_model: python -m camera_model
"""
from __future__ import annotations

# Core math and state
from . import scene
from . import rotation
from . import pinhole
from . import distortion
from . import geometry
from . import rendering
from . import state

# GUI entry point
from .app import run_app, MainWindow

__all__ = [
    "scene",
    "rotation",
    "pinhole",
    "distortion",
    "geometry",
    "rendering",
    "state",
    "run_app",
    "MainWindow",
]
