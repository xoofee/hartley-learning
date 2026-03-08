"""
suppose the K matrix is known, and no distortion.
Let user select four corners of a rectangle (in the current image in centerwidget), say, A B C D, in the ground plane
so that the edges are AB BC CD DA
calculate the vanishing line l_inf

let the AB and CD intersect at X, let's define it the x axis vanishing point
Then the R could be inferred from the l_inf and the x axis vanishing point
(P = K[R|t])

let's define the world origin same as camera origin vertically so the ground plane is z=0 in world plane. and camera is at the x=0, y=0, and z = h

let user click two points on the image, say P and Q, and P is on the Ground, the PQ perpendicular to the ground plane

suppose the distance between P and Q is 1 (in world coordinates), then the world coordinates of P and Q could be inferred from the P and Q image coordinates

then let the user drag point of P, the point of Q will be dynamically updated so that the distance between P and Q is 1 (in world coordinates)
and PQ is perpendicular to the ground plane


Gemini refine:

## Perspective-Correct Height Measurement and Dynamic Interaction

### 1. Problem Statement

In a monocular camera setup (a single 2D image), estimating the physical world position and height of an object is an "ill-posed" problem because depth information is lost during projection.

To resolve this, we need to:

* **Establish a Coordinate System:** Define the ground plane ($Z=0$) and the camera’s position relative to it (the height $h$).
* **Restore Perspective:** Use vanishing points from a ground-plane rectangle to determine the camera's orientation (Rotation matrix $R$).
* **Enforce Geometry:** Ensure that when a user selects a point $P$ on the ground and an arbitrary top point $Q$, the system can infer a 3D height of $1$ unit while maintaining the geometric constraint that $PQ$ is perpendicular to the ground, even if the user's initial mouse clicks are imprecise.

---

### 2. The Calculation Process

#### Step A: Establishing Ground Orientation ($R$)

1. **Define the Ground Plane:** The user selects four corners of a rectangle ($A, B, C, D$) on the ground.
2. **Locate Vanishing Points:** Calculate the intersection of the two sets of parallel edges:
* $v_x = (A \times B) \times (C \times D)$
* $v_y = (B \times C) \times (D \times A)$


3. **Compute Rotation ($R$):** Use the intrinsic matrix $K$ to map these to world space.

$$r_1 = \frac{K^{-1}v_x}{\|K^{-1}v_x\|}, \quad r_2 = \frac{K^{-1}v_y}{\|K^{-1}v_y\|}, \quad r_3 = r_1 \times r_2$$



The vertical vanishing point $v_z$ is then the intersection of the rays defined by the camera's optical axis and the ground normal.

#### Step B: Calibrating the Camera Height ($h$)

When the user clicks the base $P$ and top $Q$, we must solve for $h$ to establish the "scale" of the world:

1. **Back-project:** Transform pixel coordinates to world rays: $\mathbf{D}_p = R^\top K^{-1} p$ and $\mathbf{D}_q = R^\top K^{-1} q$.
2. **Solve for $h$:** We treat the initial clicks as a calibration step. By setting the horizontal alignment error to zero, we derive $h$:

$$h = \frac{D_{px} D_{qz}}{D_{px} D_{qz} - D_{pz} D_{qx}}$$



This $h$ represents the camera's distance from the ground plane.

#### Step C: Dynamic Interaction (The "Drag" Logic)

Once the camera is calibrated, we can dynamically update the position of $Q$ as the user drags $P$:

1. **Map $P$ to Ground:** As the user drags the mouse to $p_{new}$, calculate the new world position $P_{world}$ at the intersection of the ray and the plane $Z=0$:

$$P_{world} = C - \frac{h}{D_{p\_new,z}} \mathbf{D}_{p\_new}$$


2. **Calculate $Q$:** Because the object is fixed to height $1$:

$$Q_{world} = P_{world} + [0, 0, 1]^\top$$


3. **Re-project:** Map $Q_{world}$ back into the image:

$$q_{new} = K R (Q_{world} - C)$$



*Note: The calculated $q_{new}$ will automatically appear to "snap" onto the line connecting $p_{new}$ and the vertical vanishing point $v_z$, ensuring perfect geometric verticality regardless of the user's initial click tolerance.*

implement it. remember to make the code modular and flexible and have a good architecture. we may have other demos that require interactive like "click to select, drag a point on it". and some demo do not need this

do not remove this comment
"""
