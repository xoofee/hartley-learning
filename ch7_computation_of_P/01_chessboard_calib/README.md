a calibration demo using opencv/python/pyqt/matplotlib

make the layout drag/dockable/resizable

1 camera preview
let user select camera (there may be several cameras on the system.)
the user will prefer the usb camera if they exist. the integrated camera of the laptop will not be selected if the usb camera exist
support open and close of the realtime preview

2 phone take
let the user take photo while preview is on

3 gallery
the photos will be store in a (fixed) local folder.
A gallery (thumbnail) to show all the photos in the folder
load the gallery at startup
user could remove items in the gallery
user could click to view the photo in the main window (may be in a separate tab if there are many windows)

4 log widget
as in the ch6_camera_model\camera_model

5 calibration
using chessboard
let the user to input the chessboard parameters
user could click button to

- calibrate the camera using all the images in the gallery
get K, distortion coefs, and output them to log
get R,t of each camera

output reproj error to log

6 3d plot
- show a 3d plot of the chessboard and camera pose. camera is shown in a paramid as in ch6_camera_model\camera_model

since in real image calibration, we do not know the sensor size, so the K only reflect the focal length to pixel size ratio
so, in 3d plot of camera pyramid, assume a fixed height, but make the height/bottom_width and height/bottom_length be consistent to the camera matrix K. Make the pyramid size appropriate in 3d plot, not too large, and not too small

draw the chessboard grid
draw the world xyz axis at it origin in RGB if possible, just like any 3d software
draw the axis of cameras in RGB too

make the code clean, modular, scalable, flexible, extensible.
More feature may be added to it later (Structure from Motion, ..., etc)

---

## Run

From this directory:

```bash
pip install -r requirements.txt
python run_calib_app.py
```

Or from repo root (with PYTHONPATH including `ch7_computation_of_P`):

```bash
python -m ch7_computation_of_P.01_chessboard_calib.run_calib_app
```

Photos are stored in `01_chessboard_calib/calib_images/`. Use **Camera preview** to select a camera (non-zero indices listed first, often USB), start preview, and **Take photo** to capture. **Gallery** shows thumbnails; click to view in the central area, **Remove** to delete. Set **Chessboard** inner corners and square size, then **Calibrate from gallery** to get K, distortion, R/t per image and reprojection error in the **Log**. The **3D plot** shows the chessboard grid and camera pyramids (aspect from K, fixed height).

# Applications

## realtime camera pose
now add this demos
make it plugin pattern like ch6_camera_model\camera_model\plugins
so that it is extensible and scalable and modular

the first demo will be realtime pose (of camera)
when this demo is turned on
do not show the cabliration camera pose (pyramid) in the 3d view
(still keep the chessboard)

when camera preview is on
detect the camera pose using the calibrated K and the realtime chessboard image
