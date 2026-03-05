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

# gallery refactor
rename the current gallery to a reusable dockable gallery 
and add a capture button to capture image photos to this calibration gallery and the folder

when click the thumbnail in the gallery, show the image in the centerwidget like any multi-document editor. make it closable by x button just in the same way as any multi-document desktop application

for the current calibration gallery, it saves to ch7_computation_of_P\01_chessboard_calib\images\calib

add a new gallery "work"
which bind to  ch7_computation_of_P\01_chessboard_calib\images\work

remove the original take phone button
because we already have a work gallery that could take photos


# demo "rotate image"

then add a demo to rotate image. When the "camera rotate" is on, and any image from the work opened in the centerwidget
the user could drag the image to rotate it.
the transform is H = K*R*inv(K)
R is from yaw/pitch of the camera. yaw pitch is get from the mouse drag. roll is always 0

just mimic the behavior of street view of google or any other ar app of panorama/photosphere:
the current mouse position and the start position could determine the R (with K). The yaw / pitch is not necessary to get R, you may just get R without calculation yaw pitch if you think it is appropriate

this make the point under the mouse always be the same point. so the mouse move is in opposite of camera actual move


use the K load from saved. is K is not available, warn the user and return to last demo or none demo


# demo refactor
add a dockable widget to container the pane of specific demo
The "Rotate image" should have a button to reset the rotation of the image (so no rotation, show the original image)



# Tools
Add a Tools menu
add a submenu to switch on/off of the image x y in status bar
format as integer when display in the status bar

## yaw pitch
show the realtime pitch yaw in degrees in status bar (keep one decimal place) for the active ImageViewWidget if K matrix is avaiable
the yaw pitch if converted from x,y in image 2d coordinate (above) and K
add a submenu in Tools to switch this feature on/off
This feature does not depended on the "camera rotate" demo, but could make some function reusable to reduce duplication

