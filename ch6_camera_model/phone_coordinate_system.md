### 1. Device / Sensor Coordinate Frame (IMU + Raw Camera)
This is the **body-fixed frame** attached to the phone hardware. It's the same for raw IMU readings and the camera image plane (rear camera in most AR apps).

Standard convention (shared by Android and iOS):
- **X** → positive to the **right** (when holding the phone in portrait, screen facing you)
- **Y** → positive **up** (toward the top of the phone in portrait)
- **Z** → positive **out of the screen** (toward you / away from the back camera)

Key notes:
- This is a **right-handed** coordinate system.
- When the phone is held **portrait, screen facing you**:
  - Gravity ≈ **[0, -9.81, 0]** m/s2 (pulls down along -Y).
  - Rear camera looks along **-Z** (backward/out of the back).
- In landscape (rotated 90° CCW), axes rotate accordingly, but the definition stays fixed to the hardware.
- Raw accelerometer/gyro data is reported in this frame.
- Camera intrinsics/extrinsics (focal length, principal point) are defined with respect to this frame: image u increases right (+X), v increases down (opposite to +Y).

### 2. Camera Coordinate Frame in AR / V-SLAM
In ARCore and ARKit (and most computer vision pipelines), the **camera frame** is aligned very closely with the device frame, but with a key flip for the optical direction:

- **X** → right (same as device +X)
- **Y** → down (opposite to device +Y — this is the big difference!)
- **Z** → forward (along the direction the camera is looking, usually **-device Z** for rear camera)

This is the classic **OpenCV / optical / computer-vision camera frame**:
- +Z = looking direction (forward)
- +X = right in image
- +Y = down in image (rows increase downward)


---

For indoor navigation using **Visual-Inertial Odometry (VIO)** or AR SDKs on Android (ARCore) or iOS (ARKit), while also having a fallback to pure OpenCV + your own VIO inspired by **VINS-Mono** or **ORB-SLAM(3)**, the coordinate convention choice is critical for consistency across sensors, tracking output, map building, and navigation logic.

### Recommended Convention: Optical / Computer-Vision Camera Frame (the one you already have in your code)

**X = right**  
**Y = down**  
**Z = forward** (looking direction of the camera)

This is the **standard in computer vision** (OpenCV default, most V-SLAM papers, feature tracking, reprojection error minimization, bundle adjustment, etc.).

#### Why this is the best choice for your mixed setup

| Aspect                          | Your optical frame (X right, Y down, Z forward) | ARCore/ARKit native (mostly X right, Y up, -Z forward) | Pure OpenCV / your custom VIO |
|---------------------------------|--------------------------------------------------|----------------------------------------------------------|-------------------------------|
| OpenCV `solvePnP`, `projectPoints`, `triangulatePoints` | Native (no flips needed)                        | Requires axis flip (Y → -Y or Z → -Z)                   | Native                        |
| VINS-Mono style VIO             | Matches common implementations (many use Z forward, Y down) | Needs remapping                                         | Matches                       |
| ORB-SLAM3                       | Matches (Z forward in camera model)             | Needs remapping                                         | Matches                       |
| Feature tracking / optical flow | Image y increases down → +Y down = natural     | Needs y-flip in image coords                            | Matches                       |
| Gravity vector                  | Gravity ≈ [0, +g, 0] (down = +Y)                | Gravity ≈ [0, -g, 0] (up = +Y)                          | Easy to align                 |
| Pitch sign                      | +pitch = look down (as in your code)            | Often +pitch = look up or opposite                      | Consistent with your function |
| Indoor navigation simplicity    | Forward = Z, down = Y → intuitive height = -Y or +Y | More rendering-oriented                                 | Best for path planning        |

In practice, almost all custom VIO / SLAM codebases (including many ports of VINS-Mono and ORB-SLAM to mobile) **internally use the optical frame** for everything: state vector, landmarks, poses, IMU preintegration alignment.
