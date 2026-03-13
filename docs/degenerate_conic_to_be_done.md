
Degenerate conics. If the matrix C is not of full rank, then the conic is termed degenerate.
Degenerate point conics include two lines (rank 2), and a repeated line (rank
1).
Example 2.8. The conic
C = lmT + mlT
is composed of two lines l and m. Points on l satisfy lTx = 0, and are on the conic
since xTCx = (xTl)(mTx) + (xTm)(lTx) = 0. Similarly, points satisfying mTx = 0
also satisfy xTCx = 0. The matrix C is symmetric and has rank 2. The null vector is
x = l ×m which is the intersection point of l and m. 
Degenerate line conics include two points (rank 2), and a repeated point (rank 1).
For example, the line conic C
∗ = xyT + yxT has rank 2 and consists of lines passing
through either of the two points x and y. Note that for matrices that are not invertible
(C
∗)∗ = C.

is there pactical usage of degenarate conic?

https://chatgpt.com/c/69524771-1fe4-832b-9821-1fbf0930121f


Yes—**degenerate conics are not just a mathematical curiosity; they are *very practical*** in projective geometry, computer vision, and robotics. The key is to view them as **limit cases** or **compact representations of simpler geometric structures** (points, lines, intersections, unions).

Below I’ll connect them directly to *why Hartley & Zisserman introduce them* and *where you actually use them*.

---

## 1. Big picture intuition

A (non-degenerate) conic represents a **2D quadratic curve**:

* ellipse, hyperbola, parabola

A **degenerate conic** represents:

* a **pair of lines**
* a **single (double) line**
* a **pair of points**
* a **single (double) point**

So degenerate conics act as a **bridge between linear geometry (points/lines)** and **quadratic geometry**.

They let us:

* unify points, lines, and conics in **one algebraic framework**
* handle **limits**, **constraints**, and **intersections** cleanly

---

## 2. Practical usages (important ones)

### 2.1 Representing *two lines as one object*

Your example:
[
C = l m^T + m l^T
]

represents **exactly the union of two lines**.

#### Why is this useful?

* Instead of storing two separate lines, you store **one symmetric 3×3 matrix**
* You can:

  * intersect it with another conic
  * transform it with a homography
  * differentiate / optimize over it

📌 **Used in**:

* line-pair constraints
* vanishing line computation
* camera calibration steps

---

### 2.2 Vanishing points & vanishing lines (VERY practical)

Vanishing geometry is full of **degenerate conics**.

Example:

* Two parallel 3D directions → two vanishing points
* The **vanishing line of a plane** can be represented as a **degenerate dual conic**

Degenerate conics appear when:

* a plane at infinity is projected
* metric constraints collapse under perspective

📌 **Used in**:

* single-view metrology
* Manhattan world detection
* camera intrinsics estimation

---

### 2.3 Duality: point conic ↔ line conic

Degenerate **dual conics** (line conics) represent:

* all lines through a point
* all lines through either of two points

Example:
[
C^* = x y^T + y x^T
]

This represents:

> the set of lines passing through **x OR y**

📌 **Used in**:

* epipolar geometry
* line-based correspondence
* multi-view constraints

This shows up naturally in:

* epipoles
* degenerate fundamental matrices
* critical motion sequences

---

### 2.4 Epipolar geometry (critical!)

In **degenerate camera configurations**:

* pure rotation
* planar scenes
* camera centers aligned

The **fundamental matrix degenerates**, and its associated conic objects become degenerate.

Understanding degenerate conics helps you:

* diagnose failure cases
* detect planar degeneracy
* design fallback algorithms

📌 **Used in**:

* SLAM / VSLAM
* structure-from-motion
* homography vs fundamental matrix decision

---

### 2.5 Conics as constraints in optimization

In practice, conics are used as **soft constraints**:

* Fit a circle → becomes degenerate if radius → ∞
* Two nearly-parallel lines → conic approaches rank-2
* Noise pushes rank-3 → rank-2 boundary

Degenerate conics are **limit cases** during:

* least squares fitting
* bundle adjustment
* RANSAC hypothesis testing

📌 **Used in**:

* robust geometry fitting
* numerical stability analysis

---

### 2.6 VSLAM & landmarks (your question earlier 👀)

> *“does vslam use the concept of conic?”*

Yes—**implicitly**.

Examples:

* Circular landmarks (poles, wheels) → conics
* Degenerate cases:

  * circle seen edge-on → two lines
  * very distant circle → double line

Understanding degenerate conics explains:

* why pose recovery fails in certain viewpoints
* why orientation is recoverable but scale is not

---

## 3. Why Hartley insists on degenerate conics

Because they allow:

1. **Coordinate-free reasoning**
2. **Unified treatment of special cases**
3. **Algebraic duality (points ↔ lines)**
4. **Stable projective transformations**

In projective geometry:

> *Degeneracy is not an error — it is structure.*

---

## 4. Summary table

| Degenerate Conic   | Represents   | Practical use                   |
| ------------------ | ------------ | ------------------------------- |
| Rank 2 point conic | Two lines    | Vanishing geometry, calibration |
| Rank 1 point conic | Double line  | Limit of distant curves         |
| Rank 2 line conic  | Two points   | Epipolar constraints            |
| Rank 1 line conic  | Double point | Critical camera motions         |

---

## 5. One-sentence takeaway

> **Degenerate conics are how projective geometry encodes points and lines inside quadratic form machinery—making special cases analyzable, transformable, and optimizable.**

If you want, next I can:

* show **how a circle becomes two lines under perspective**
* connect degenerate conics to **homography vs fundamental matrix**
* give **code-level examples** (OpenCV / NumPy)

