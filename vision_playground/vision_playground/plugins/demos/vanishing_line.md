## Perspective-Correct Height Measurement and Camera Calibration

This document outlines a geometric framework for recovering 3D physical height and camera positioning from a single calibrated 2D image. By utilizing the properties of vanishing points and ground-plane constraints, we can resolve the depth ambiguity inherent in monocular vision.

---

### 1. Problem Statement

In a 2D image, depth information is lost. A single pixel $p$ corresponds to a 3D ray originating from the camera center $C$, represented as $P = C + s(R^\top K^{-1}p)$. Without knowing the scale (depth $s$) or the camera's orientation ($R$), we cannot determine the physical height of objects. To solve this, we must establish a coordinate system relative to a known ground plane and define a reference scale.

---

### 2. The Solution: Ground-Plane Triangulation

The solution assumes the world origin lies on a flat ground plane ($Z=0$). By identifying a rectangle on this plane, we recover the camera's rotation $R$. By defining a vertical segment of known length (1 unit) between the ground and an object’s top, we triangulate the camera's 3D position $C$, effectively "scaling" the virtual environment to match physical reality.

---

### 3. The Process and Mathematical Framework

#### Step A: Establishing Orientation (Rotation $R$)

We determine the camera's tilt and roll by finding the "horizon" through two orthogonal vanishing points.

1. **Vanishing Points ($v_x, v_y$):** Let $A, B, C, D$ be the image coordinates of a ground rectangle. The vanishing points are found via the intersection of parallel lines in homogeneous space:

$$v_x = (A \times B) \times (C \times D), \quad v_y = (B \times C) \times (D \times A)$$


2. **Rotation Matrix Construction:** We back-project these to world-space directions $d_1$ and $d_2$ using the inverse intrinsic matrix $K^{-1}$:

$$d_1 = K^{-1}v_x, \quad d_2 = K^{-1}v_y$$



We normalize and orthogonalize these vectors to form the columns of $R$:

$$r_1 = \frac{d_1}{\|d_1\|}, \quad r_3 = \frac{r_1 \times d_2}{\|r_1 \times d_2\|}, \quad r_2 = r_3 \times r_1$$



The orientation is thus $R = [r_1, r_2, r_3]$.

#### Step B: Calibrating Camera Position ($C$)

We define a reference segment where $P_{world} = [0,0,0]^\top$ (base) and $Q_{world} = [0,0,1]^\top$ (top).

1. **Ray Generation:** For user clicks $p$ and $q$, the back-projected rays in world coordinates are:

$$\mathbf{d}_p = R^\top K^{-1} p, \quad \mathbf{d}_q = R^\top K^{-1} q$$


2. **Least Squares Triangulation:** Since $Q - P = [0, 0, 1]^\top$, and $P = C + s_1 \mathbf{d}_p$ and $Q = C + s_2 \mathbf{d}_q$, we solve for depths $s_1, s_2$:

$$\begin{bmatrix} -\mathbf{d}_{px} & \mathbf{d}_{qx} \\ -\mathbf{d}_{py} & \mathbf{d}_{qy} \\ -\mathbf{d}_{pz} & \mathbf{d}_{qz} \end{bmatrix} \begin{bmatrix} s_1 \\ s_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$


3. **Camera Center:** The camera position is $C = -s_1 \mathbf{d}_p$.

#### Step C: Dynamic Interaction (The "Drag" Logic)

Once $C$ and $R$ are known, we can update the image height dynamically as the user moves $p_{new}$.

1. **Ground Intersection:** Find where the new ray $\mathbf{d}_{p\_new}$ hits $Z=0$:

$$s = -\frac{C_z}{\mathbf{d}_{p\_new, z}}, \quad P_{world} = C + s \mathbf{d}_{p\_new}$$


2. **Constraint:** Set $Q_{world} = P_{world} + [0, 0, 1]^\top$.
3. **Reprojection:** Project $Q_{world}$ back to image coordinates $q_{new}$:

$$q_{new} \sim K R (Q_{world} - C)$$



This ensures $q_{new}$ stays perfectly aligned with the vertical vanishing point $v_z$ relative to $p_{new}$.
