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
- **plugins/** — Open-closed: `Feature` base, registry, built-in (P planes, backproject).
- **app.py** — Main window, composes above; `run_app()` entry point.

Run (from `ch6_camera_model`): `python run_imaging_app.py` or `python -m camera_model`.

# plugins/feature/demos

make an area of demos
this area contains a lot of buttons, they eclusive. each button is a demo/mode
and there functions should not interven. That is make each function independent to avoid chaos of code

each button could be pushed or release (on/off)

## 1 angulometer

get angle between two points by K. the the user could drag two points
calculate the angle between the two points
should the angle dynamically on the image plot
should the ray of the two points in the 3d plot

when the button is off, release all the related objects

## 2 ....
in the future we may add demo/feature 2, 3, ... etc

refactor the previous P row planes / single point ray demo if necessary

remember to use a good archtecture as i say before