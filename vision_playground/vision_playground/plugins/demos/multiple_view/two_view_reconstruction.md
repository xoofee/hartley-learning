
a two view reconstruction demo

# data source
user will select two images (and it will be opened in the center widget, this is already implemented), this demo will get data from the opened images.

# configurations
user can configure if K is used.
if K is used, get the global K

# Reconstruction Pipeline

   Feature Matching
        ↓
   F / E Estimation
        ↓
   Pose Recovery (R,t)
        ↓
   Triangulation
        ↓
   3D Visualization

# when user press the reconstruct button, the demo will:
1. get dense matches between the two images
2. calculate the fundamental matrix F or E, based on K known or not
3. calculate the rotation matrix R and the translation vector t
4. get the 3D points from the two images    (triangulate the matches)
5. visualize the 3D points with point color same as the image color (like Gaussian Splatting)
(3d visualize in the center widget with a closable tab just like the image tabs)
show the two camera poses in the 3d view (refer to the implementation in the 3d plot of calibration)

in the future, we will add visualization for 
epipolar lines, inlier/outlier matches, reprojection error

do not remove this comment.
Make the code modular and flexible and have a good architecture.
do not make a god file. do not repeat the same code in this vision playground app, unless it is only for this demo.


reference doc:



# 1. Overall Architecture

```
PyQt GUI
 ├─ Image viewer (already implemented)
 ├─ Reconstruction button
 ├─ Settings panel (use K or not)
 │
 └─ Reconstruction Pipeline
        ↓
   Feature Matching
        ↓
   F / E Estimation
        ↓
   Pose Recovery (R,t)
        ↓
   Triangulation
        ↓
   3D Visualization
```

---

# 2. Step-by-Step Implementation

## Step 1 — Dense matches

Use:

* ORB / SIFT (classic)
* or deep matchers (LoFTR, LightGlue)

Simple OpenCV approach:

```python
orb = cv2.ORB_create(5000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
```

---

# 3. Estimate F or E

If **K unknown**

```
F = cv2.findFundamentalMat(pts1, pts2, RANSAC)
```

If **K known**

```
E = cv2.findEssentialMat(pts1, pts2, K)
```

---

# 4. Recover camera pose

```
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

Camera matrices:

```
P1 = K [ I | 0 ]
P2 = K [ R | t ]
```

or if uncalibrated:

```
P1 = [I|0]
P2 from F
```

---

# 5. Triangulate 3D points

```
pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
pts3d = pts4d[:3] / pts4d[3]
```

---

# 6. Get point colors from image

```
colors = img1[pts1[:,1].astype(int), pts1[:,0].astype(int)]
colors = colors / 255.0
```

---

# 7. 3D Visualization with Matplotlib

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    pts3d[0],
    pts3d[1],
    pts3d[2],
    c=colors,
    s=1
)

plt.show()
```

This will give a **colored sparse point cloud** similar to SfM output.

---

# 8. Integrating with PyQt

Matplotlib embeds nicely inside PyQt:

```
FigureCanvasQTAgg
```

Example:

```
PyQt Widget
 └─ Matplotlib canvas
      └─ 3D scatter
```

So your app could have:

```
Left panel: configuration
Center: images
Right panel: 3D reconstruction
```

---

# 9. Expected Result

From two images you will get something like:

```
sparse colored 3D point cloud
+ camera pose
```

This is basically a **mini two-view Structure-from-Motion pipeline** similar to what **COLMAP** does internally (but massively simplified).

---

# 10. Important Limitations

Two-view reconstruction gives:

* **up-to-scale reconstruction**
* not very dense
* unstable if baseline is small

Also:

* Matplotlib handles **~100k points max** comfortably
* for bigger clouds you would need **OpenGL / PyQtGraph / Open3D**

---


in the future, we will add visualization for 
epipolar lines, inlier/outlier matches, reprojection error

reference:
---

# 1. Inlier / Outlier Matches

When estimating **F or E**, **RANSAC returns a mask** telling which matches are inliers.

### Example

```python
F, mask = cv2.findFundamentalMat(
    pts1,
    pts2,
    cv2.FM_RANSAC,
    ransacReprojThreshold=1.0,
    confidence=0.99
)

mask = mask.ravel()
```

Separate points:

```python
inliers1 = pts1[mask == 1]
inliers2 = pts2[mask == 1]

outliers1 = pts1[mask == 0]
outliers2 = pts2[mask == 0]
```

### Visualization

Overlay on image:

```python
plt.imshow(img1)

plt.scatter(inliers1[:,0], inliers1[:,1], c='lime', s=5)
plt.scatter(outliers1[:,0], outliers1[:,1], c='red', s=5)

plt.title("Green = Inliers, Red = Outliers")
```

Or draw match lines between images.

---

# 2. Epipolar Lines

Use OpenCV:

```python
cv2.computeCorrespondEpilines
```

### Compute lines in image2 corresponding to points in image1

```python
lines2 = cv2.computeCorrespondEpilines(
    pts1.reshape(-1,1,2),
    1,
    F
)
lines2 = lines2.reshape(-1,3)
```

Each line is:

```
ax + by + c = 0
```

### Draw them

```python
def draw_epilines(img, lines, pts):

    r,c = img.shape[:2]

    for line, pt in zip(lines, pts):

        a,b,c_line = line

        x0,y0 = 0, int(-c_line/b)
        x1,y1 = img.shape[1], int(-(c_line + a*x1)/b)

        cv2.line(img,(x0,y0),(x1,y1),(0,255,0),1)
        cv2.circle(img,tuple(pt.astype(int)),4,(0,0,255),-1)

    return img
```

Show in your GUI.

For teaching, it is nice to show:

```
click point → show corresponding epipolar line
```

---

# 3. Camera Frustums (3D)

Once you have **R,t**, you know camera poses.

Camera centers:

```
C1 = [0,0,0]
C2 = -R^T t
```

### Build a frustum model

Define camera pyramid in camera coordinates:

```python
scale = 0.2

frustum = np.array([
    [0,0,0],
    [-1,-1,1],
    [1,-1,1],
    [1,1,1],
    [-1,1,1]
]) * scale
```

Transform using pose.

For camera 2:

```
X_world = R.T @ (X_cam - t)
```

Or simpler:

```
X_world = R @ X_cam + t
```

depending on convention.

### Plot in Matplotlib

```python
ax.plot([C[0],corner[0]], [C[1],corner[1]], [C[2],corner[2]])
```

Connect corners to form pyramid.

This gives classic **SfM camera visualization**.

---

# 4. Reprojection Error

This is **one of the most educational metrics**.

### Project 3D points back to image

Using:

```
x = P X
```

Example:

```python
def project(P, X):

    X_h = np.vstack([X, np.ones(X.shape[1])])

    x = P @ X_h
    x = x[:2] / x[2]

    return x
```

### Compute error

```python
proj1 = project(P1, pts3d)
proj2 = project(P2, pts3d)

err1 = np.linalg.norm(proj1.T - pts1, axis=1)
err2 = np.linalg.norm(proj2.T - pts2, axis=1)

error = (err1 + err2) / 2
```

---

### Visualize error

Color the points:

```python
plt.scatter(
    pts1[:,0],
    pts1[:,1],
    c=error,
    cmap="jet"
)
plt.colorbar(label="Reprojection error")
```

Or threshold:

```
green = good
red = bad
```

---

# 5. Very Nice Teaching Feature (Highly Recommended)

Add **interactive point inspection**.

When user clicks a point:

show

```
• matched point
• epipolar line
• reprojection error
• 3D coordinate
```

Flow:

```
mouse click
    ↓
nearest feature
    ↓
highlight
    ↓
update 3D + epipolar
```

This makes the demo **very powerful for learning**.

---

# 6. Recommended Layout for Your App

Example:

```
+----------------------------------------+
| left image | right image               |
|            |                           |
|  matches   |  epipolar lines           |
+----------------------------------------+
|             3D viewer                  |
|   point cloud + camera frustums        |
+----------------------------------------+
```
