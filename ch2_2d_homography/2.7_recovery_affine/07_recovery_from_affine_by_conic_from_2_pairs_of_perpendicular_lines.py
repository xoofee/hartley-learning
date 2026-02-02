"""
load ch2\2.7_recovery_affine\affinely_rectificed_floor.jpg
pop out the window to let the user click 6 points, say, A, B, C, D, E, F
get homogeneous coordinates of line AB and BC
line DE and EF
(by cross product)
so we have 2 pairs of lines, to satisfy l^T C m = 0
the first pair is AB and BC, the second pair is DE and EF

The full equation is:

[l1 l2 l3] [c11 c12 c13;  [m1
            c12 c22 c23;   m2
            c13 c23 c33]   m3]
            = [0 0 0]

expand and combine with c, write in a text friendly style

[ l1m1,
l1m2 + l2m1,
l1m3 + l3m1,
l2m2,
l2m3 + l3m2,
l3*m3 ] ·
[ c11, c12, c13, c22, c23, c33 ] = 0

for (already) affinely rectified floor, image, only c11 c12, c22 are non-zero

C =
[ c11 c12 0
c12 c22 0
0 0 0 ]


[ l1m1,
l1m2 + l2m1,
l2m2] ·
[ c11, c12, 0, c22, 0, 0 ] = 0

so, get null space of the matrix
Solve Ac = 0 via SVD, the last row of Vt is the solution, reshape to get the 2x2 submatrix of C, let's call it KKT

_, _, Vt = np.linalg.svd(A)
KKT = Vt[-1].reshape(2, 2)

then use LU decomposition to solve K*K^T = KKT (2x2 matrix)

then construct H from K
[K 0; 0 1]

transform the image using H
show the original and transformed image side by side

do not remove this comment.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# Global variables to store clicked points
clicked_points = []
fig = None
ax = None
image_rgb = None

def on_click(event):
    """Callback function for matplotlib mouse clicks"""
    global clicked_points, fig, ax, image_rgb
    
    if event.inaxes != ax:
        return
    
    if event.button == 1:  # Left mouse button
        if len(clicked_points) < 6:
            x, y = event.xdata, event.ydata
            clicked_points.append([x, y])
            
            # Label points as A, B, C, D, E, F
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
            label = labels[len(clicked_points) - 1]
            
            # Draw a circle at the clicked point
            circle = Circle((x, y), 10, color='lime', fill=True)
            ax.add_patch(circle)
            ax.text(x + 15, y - 15, label, 
                   color='lime', fontsize=12, weight='bold')
            fig.canvas.draw()
            
            print(f"Point {label}: ({x:.1f}, {y:.1f})")
            
            if len(clicked_points) == 6:
                print("All 6 points selected! Close the window to continue...")

def select_points(image):
    """Display image and let user click 6 points using matplotlib"""
    global clicked_points, fig, ax, image_rgb
    clicked_points = []
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title('Click 6 points: A, B, C, D, E, F\n(AB and BC form first pair, DE and EF form second pair)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("Click 6 points: A, B, C, D, E, F")
    print("Points A, B, C should form two lines (AB and BC) that are perpendicular in the metric space")
    print("Points D, E, F should form two lines (DE and EF) that are perpendicular in the metric space")
    plt.show()
    
    if len(clicked_points) != 6:
        print("Warning: Not all 6 points were selected!")
    
    return np.array(clicked_points, dtype=np.float32)

def point_to_homogeneous(p):
    """Convert 2D point to homogeneous coordinates"""
    return np.array([p[0], p[1], 1.0])

def line_from_points(p1, p2):
    """Calculate line from two points using cross product"""
    p1_h = point_to_homogeneous(p1)
    p2_h = point_to_homogeneous(p2)
    line = np.cross(p1_h, p2_h)
    # Normalize line
    if abs(line[2]) > 1e-10:
        line = line / line[2]
    return line

def solve_conic_from_line_pairs(line_pairs):
    """
    Solve for the conic C from line pairs.
    For affine space, C has the form:
    C = [c11 c12 0; c12 c22 0; 0 0 0]
    
    The constraint is l^T C m = 0 for each pair (l, m).
    This expands to: l1*m1*c11 + (l1*m2 + l2*m1)*c12 + l2*m2*c22 = 0
    
    Args:
        line_pairs: List of (l, m) tuples where l and m are lines in homogeneous coordinates
    
    Returns:
        KKT: 2x2 matrix representing the conic (C[:2, :2])
    """
    # Build constraint matrix A
    # Each row corresponds to one constraint: [l1*m1, l1*m2 + l2*m1, l2*m2] · [c11, c12, c22] = 0
    A = []
    for l, m in line_pairs:
        l1, l2, l3 = l
        m1, m2, m3 = m
        
        # For affine space, only c11, c12, c22 are non-zero
        # Constraint: l1*m1*c11 + (l1*m2 + l2*m1)*c12 + l2*m2*c22 = 0
        constraint = [
            l1 * m1,                    # coefficient for c11
            l1 * m2 + l2 * m1,          # coefficient for c12
            l2 * m2                     # coefficient for c22
        ]
        A.append(constraint)
    
    A = np.array(A)
    
    # Solve Ac = 0 via SVD
    # The last row of Vt is the solution
    _, _, Vt = np.linalg.svd(A)
    c_solution = Vt[-1]
    
    # Extract c11, c12, c22
    c11, c12, c22 = c_solution
    
    # Construct the 2x2 conic matrix KKT = C[:2, :2]
    KKT = np.array([
        [c11, c12],
        [c12, c22]
    ])
    
    return KKT

def decompose_KKT_to_K(KKT):
    """
    Decompose KKT = K * K^T where K is a 2x2 upper triangular matrix.
    This is done using Cholesky decomposition.
    
    Since KKT = K * K^T, we can use Cholesky decomposition if KKT is positive definite.
    Cholesky gives us L (lower triangular) where KKT = L * L^T.
    To get an upper triangular K, we set K = L^T.
    
    However, if KKT is not positive definite, we may need to adjust.
    For a conic representing perpendicularity constraints, KKT should be positive definite.
    
    Args:
        KKT: 2x2 symmetric matrix
    
    Returns:
        K: 2x2 upper triangular matrix such that K * K^T = KKT
    """
    try:
        # Cholesky decomposition: KKT = L * L^T where L is lower triangular
        # To get upper triangular K, we set K = L^T
        L = np.linalg.cholesky(KKT)
        K = L.T  # Transpose to get upper triangular
        return K
    except np.linalg.LinAlgError:
        # If Cholesky fails, KKT might not be positive definite
        # Try to make it positive definite by adding a small regularization
        KKT_reg = KKT + np.eye(2) * 1e-6
        try:
            L = np.linalg.cholesky(KKT_reg)
            K = L.T  # Transpose to get upper triangular
            return K
        except np.linalg.LinAlgError:
            # If still fails, use SVD-based approach and make K upper triangular
            # KKT = U * S * U^T (symmetric), we want K (upper triangular) such that K * K^T = KKT
            U, S, Vt = np.linalg.svd(KKT)
            # For symmetric matrix, U and V should be the same (up to sign)
            # Take absolute values of S to ensure positive
            S_sqrt = np.sqrt(np.maximum(S, 1e-10))
            # K_temp such that K_temp * K_temp^T = KKT
            K_temp = U @ np.diag(S_sqrt)
            # To get upper triangular K: do QR on K_temp^T to get K_temp^T = Q * R (R upper triangular)
            # Then we have K_temp = R^T * Q^T
            # We want K upper triangular such that K * K^T = KKT = K_temp * K_temp^T
            # One approach: use QR on K_temp to get K_temp = Q * R, but then K = R won't satisfy K*K^T = KKT
            # Better: Since we want K upper triangular, we can compute Cholesky on a more regularized version
            # Or: compute an upper triangular approximation
            # Simplest fallback: take upper triangular part and adjust
            K_approx = np.triu(K_temp)  # Take upper triangular part
            # Ensure the diagonal is positive and reasonable
            for i in range(2):
                if abs(K_approx[i, i]) < 1e-10:
                    K_approx[i, i] = 1e-6
                if K_approx[i, i] < 0:
                    K_approx[i, :] *= -1
            return K_approx

def construct_homography_from_K(K):
    """
    Construct homography H from K.
    H = [K 0; 0 1]
    
    Args:
        K: 2x2 matrix
    
    Returns:
        H: 3x3 homography matrix
    """
    H = np.eye(3)
    H[:2, :2] = K
    return H

def main():
    # 1. Load the affinely rectified floor image
    image_path = r'ch2\2.7_recovery_affine\affinely_rectificed_floor.jpg'
    
    # Try different possible paths
    if not os.path.exists(image_path):
        # Try relative to current directory
        image_path = 'affinely_rectificed_floor.jpg'
    if not os.path.exists(image_path):
        image_path = os.path.join('ch2', '2.7_recovery_affine', 'affinely_rectificed_floor.jpg')
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please make sure affinely_rectificed_floor.jpg exists in the ch2/2.7_recovery_affine directory")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # 2. Let user click 6 points (A, B, C, D, E, F)
    points = select_points(image)
    
    if len(points) != 6:
        print("Error: Need exactly 6 points")
        return
    
    A, B, C, D, E, F = points[0], points[1], points[2], points[3], points[4], points[5]
    
    print(f"\nPoints selected:")
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")
    print(f"D: {D}")
    print(f"E: {E}")
    print(f"F: {F}")
    
    # 3. Get lines AB, BC, DE, EF
    line_AB = line_from_points(A, B)
    line_BC = line_from_points(B, C)
    line_DE = line_from_points(D, E)
    line_EF = line_from_points(E, F)
    
    print(f"\nLines:")
    print(f"AB: {line_AB}")
    print(f"BC: {line_BC}")
    print(f"DE: {line_DE}")
    print(f"EF: {line_EF}")
    
    # 4. Form line pairs: (AB, BC) and (DE, EF)
    # These pairs should satisfy l^T C m = 0 where C is the conic
    line_pairs = [
        (line_AB, line_BC),  # First pair
        (line_DE, line_EF)    # Second pair
    ]
    
    # 5. Solve for the conic C (specifically the 2x2 submatrix KKT)
    KKT = solve_conic_from_line_pairs(line_pairs)
    print(f"\nConic matrix KKT (C[:2, :2]):\n{KKT}")
    
    # 6. Decompose KKT = K * K^T to get K
    K = decompose_KKT_to_K(KKT)
    print(f"\nMatrix K:\n{K}")
    
    # Verify: K * K^T should be close to KKT
    KKT_reconstructed = K @ K.T
    print(f"\nVerification - K * K^T:\n{KKT_reconstructed}")
    print(f"Difference from original KKT:\n{np.abs(KKT_reconstructed - KKT)}")
    
    # 7. Construct homography H from K
    H = construct_homography_from_K(K)
    print(f"\nHomography matrix H:\n{H}")
    
    # 8. Transform the image using H
    h, w = image.shape[:2]
    transformed_image = cv2.warpPerspective(image, np.linalg.inv(H), (w, h))
    
    # 9. Draw line segments and points on original image for visualization
    image_with_lines = image.copy()
    
    # Draw points
    point_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    point_colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0), 
                    (255, 0, 0), (255, 0, 0), (255, 0, 0)]
    for i, (pt, label, color) in enumerate(zip(points, point_labels, point_colors)):
        pt_int = tuple(map(int, pt))
        cv2.circle(image_with_lines, pt_int, 8, color, -1)
        cv2.putText(image_with_lines, label, (pt_int[0] + 10, pt_int[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw line segments AB, BC (green)
    A_int = tuple(map(int, A))
    B_int = tuple(map(int, B))
    C_int = tuple(map(int, C))
    cv2.line(image_with_lines, A_int, B_int, (0, 255, 0), 2)
    cv2.line(image_with_lines, B_int, C_int, (0, 255, 0), 2)
    
    # Draw line segments DE, EF (red)
    D_int = tuple(map(int, D))
    E_int = tuple(map(int, E))
    F_int = tuple(map(int, F))
    cv2.line(image_with_lines, D_int, E_int, (0, 0, 255), 2)
    cv2.line(image_with_lines, E_int, F_int, (0, 0, 255), 2)
    
    # 10. Show the two images side by side
    fig, axes = plt.subplots(2, 1, figsize=(15, 7))
    
    # Original image with lines and points
    axes[0].imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Affinely Rectified Image\n(Green: AB, BC; Red: DE, EF)', fontsize=12)
    axes[0].axis('off')
    
    # Transformed image
    axes[1].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Metric Recovered Image', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDone! Close the matplotlib window to exit.")

if __name__ == "__main__":
    main()
