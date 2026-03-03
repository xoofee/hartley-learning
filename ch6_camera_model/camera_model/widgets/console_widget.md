# Console widget

REPL-style Python console: single text area, Enter runs the current line/block. No separate Run button.

## How it gets P, K, state, etc.

The console does **not** own the camera state. The main window passes a **namespace getter** (e.g. `MainWindow._get_console_namespace`). Each time you press Enter, the console calls that getter and gets a dict with:

- `state` – the app’s `CameraState` (intrinsics, pose, distortion)
- `P`, `K`, `R`, `t` – from `state.get_P()`, `state.get_K()`, `state.get_R_and_t()`
- `fig`, `ax3d`, `ax_img` – matplotlib figure and axes
- `square_pts`, `triangle_pts`, `rectangle_pts` – scene shapes
- `redraw` – call to refresh UI and plot
- `np` – numpy

So P, K, R, t are **computed** by the main app (state doesn’t store them; it recomputes from parameters).

## Persistent namespace

The console keeps a **persistent** namespace so imports and variables (e.g. `import pickle`, `x = 1`) survive across lines. Before **every** run it does `_persistent_ns.update(fresh)`, so names from the getter (including `P`, `K`, `R`, `t`) are refreshed from the main app.

## Caveat: in-place edits to P, K, R, t

Because `P`, `K`, `R`, `t` are **overwritten from the getter before each line**:

- `P[2,3] = 1.0` modifies the array that `P` currently refers to.
- On the **next** line, the namespace is updated and `P` is replaced by a new `state.get_P()`, so your edit is no longer in `P`.
- So `state.set_from_P(P)` on the next line receives the **unmodified** P, not your edited one.

**Ways to change the camera from the console:**

- Change **state** and redraw: `state.C_x = 5.0` then `redraw()`.
- Modify P then set in **one line**: `P[2,3]=1.0; state.set_from_P(P); redraw()` (so the same `P` is used before the next refresh).
- Or get a fresh P, modify, set: `P = state.get_P(); P[2,3]=1.0; state.set_from_P(P); redraw()` in one line.

## Other behavior

- **History** is selectable and copyable; only the line after the last `>>> ` is editable.
- **Selection + key**: typing with text selected appends at the end of the input.
- **Multi-line**: incomplete blocks get a `... ` prompt; buffer is kept until the block is complete.
