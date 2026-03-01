# PRD

refactor ch6_camera_model\01_imaging_simulation.py to this folder to make it
- scalable. More testing features will incoorperated to this model. so should use principles like OPEN-CLOSE principle/ plugins pattern
- modular. should obey Single Responsibily. should split it by concept group, responsibility, ...
- clean
- maintainable

in the future, more feature/demo will be added
- binocular
- multiple camera
- epipolar geometry demo
- get angle between two points by K
- ...

This project will be a learning/teaching demo

do not change the ch6_camera_model\01_imaging_simulation.py file, just moving/refactor the original code here.

---

## Refactored layout (done)

- **scene.py** — Scene geometry (square, triangle, rectangle in world coords).
- **rotation.py** — Euler ↔ rotation matrix (yaw/pitch/roll, R_wc, R_CW_BASE_CAM).
- **pinhole.py** — Intrinsics K, projection P, project_points, decompose_P.
- **distortion.py** — OpenCV-style lens distortion (apply/undistort, polygon edges).
- **geometry.py** — Backproject ray, camera pyramid, P row planes (plane–box intersection).
- **rendering.py** — Draw projected scene, vanishing points, world origin on image, rasterize.
- **state.py** — `CameraState`: intrinsics, pose (C, pitch/yaw/roll), distortion.
- **widgets/** — Qt widgets: matrix display/edit, camera params, distortion, rotation, C.
- **plugins/** — Open-closed: `Feature` base, **Demo** base + exclusive demos registry; built-in demos in **demos.py**.
- **app.py** — Main window, composes above; `run_app()` entry point.

Run (from `ch6_camera_model`): `python run_imaging_app.py` or `python -m camera_model`.

---

# plugins/feature/demos (done)

Demos area: **exclusive** buttons (only one demo active at a time). Each demo’s logic is independent; when a demo is turned off, `on_deactivated()` is called so it can release all related state.

- **plugins/registry.py** — `Demo` base class and `register_demo` / `get_demos` / `get_demo_by_id`.
- **plugins/demos/** — Built-in demos (one file per demo): `none_demo.py`, `p_row_planes.py`, `backproject.py`, `angulometer.py`; `__init__.py` exports `register_builtin_demos()` and `build_demos_button_group()`.

## 1 Angulometer

- Get angle between two points using K (backproject two image points to rays, then angle = arccos(d1·d2)).
- User can drag two points on the image (drag near a point to move it).
- Angle is shown **dynamically on the image plot** (yellow text between the two points).
- Both rays are drawn in the **3D plot** (lime and cyan).
- When the demo is turned off, all related state is released (`on_deactivated()` clears points and dragging).

## 2 …

Further demos (e.g. epipolar, binocular) can be added by implementing `Demo` and calling `register_demo()`; no changes to core app.

P row planes and single-point backproject are refactored into the same exclusive demos (P row planes, Backproject).