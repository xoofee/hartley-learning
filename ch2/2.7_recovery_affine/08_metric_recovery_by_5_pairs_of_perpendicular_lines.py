"""
load ch2\2.3_projective_transform\building.jpg
pop out the window to let the user click 20 points, say, A1 B1 C1 D1, A2 B2 C2 D2, A3 B3 C3 D3, A4 B4 C4 D4, A5 B5 C5 D5
get homogeneous coordinates of lines
A1B1, C1D1
A2B2, C2D2,
 A3B3, C3D3, 
 A4B4, C4D4,
  A5B5, C5D5

(by cross product)
so we have 5 pairs of lines, to satisfy l^T C m = 0

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

get null space of the matrix
Solve Ac = 0 via SVD, the last row of Vt is the solution, reshape to get the C

C is 
[KK^T KK^T*v;
 v^T*KK^T v^T*KK^T*v
]

_, _, Vt = np.linalg.svd(A)
C = Vt[-1].reshape(3, 3)

KKT=C[:2, :2]

then use LU decomposition to solve K*K^T = KKT (2x2 matrix)

then get v

then construct H (from world to image) from
Ha = 
[K 0; 0 1]
Hp = [I 0; v^T 1]

H = Hp * Ha (from world to image)

transform the image using H
show the original and transformed image side by side

do not remove this comment.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json


import matplotlib
matplotlib.use("TkAgg")
# to avoid on windows
# File "c:\App\Python\env\xf\Lib\site-packages\matplotlib\backends\backend_qt.py", line 166, in _may_clear_sock rsock.recv(1) OSError: 

# Global variables to store clicked points
clicked_points = []
fig = None
ax = None
image_rgb = None
dynamic_line = None  # For dynamic line from last clicked point to cursor
permanent_lines = []  # Store permanent line objects
permanent_points = []  # Store point annotations

def on_click(event):
    """Callback function for matplotlib mouse clicks"""
    global clicked_points, fig, ax, image_rgb, dynamic_line, permanent_lines, permanent_points
    
    if event.inaxes != ax:
        return
    
    if event.button == 2:  # Middle mouse button
        if len(clicked_points) < 20:
            x, y = event.xdata, event.ydata
            clicked_points.append([x, y])
            
            # Label points as A1, B1, C1, D1, A2, B2, C2, D2, etc.
            pair_idx = (len(clicked_points) - 1) // 4
            point_idx = (len(clicked_points) - 1) % 4
            labels = ['A', 'B', 'C', 'D']
            label = f"{labels[point_idx]}{pair_idx + 1}"
            
            # Remove dynamic line if it exists
            if dynamic_line is not None:
                dynamic_line.remove()
                dynamic_line = None
            
            # Draw a cross at the clicked point
            cross_size = 10
            ax.plot([x - cross_size, x + cross_size], [y, y], 'lime', linewidth=4)
            ax.plot([x, x], [y - cross_size, y + cross_size], 'lime', linewidth=4)
            text_obj = ax.text(x + 15, y - 15, label, 
                   color='lime', fontsize=12, weight='bold')
            permanent_points.append((x, y, label, text_obj))
            
            # Draw permanent line when B or D is clicked (completing AB or CD)
            if point_idx == 1:  # B clicked - draw AB
                A = clicked_points[-2]
                B = clicked_points[-1]
                line_obj, = ax.plot([A[0], B[0]], [A[1], B[1]], 'lime', linewidth=4, alpha=0.7)
                permanent_lines.append(line_obj)
            elif point_idx == 3:  # D clicked - draw CD
                C = clicked_points[-2]
                D = clicked_points[-1]
                line_obj, = ax.plot([C[0], D[0]], [C[1], D[1]], 'lime', linewidth=4, alpha=0.7)
                permanent_lines.append(line_obj)
            
            fig.canvas.draw()
            
            print(f"Point {label}: ({x:.1f}, {y:.1f})")
            
            if len(clicked_points) == 20:
                print("All 20 points selected! Close the window to continue...")
                # Remove dynamic line if still exists
                if dynamic_line is not None:
                    dynamic_line.remove()
                    dynamic_line = None

def on_motion(event):
    """Callback function for mouse motion - draws dynamic line"""
    global clicked_points, fig, ax, image_rgb, dynamic_line
    
    if len(clicked_points) == 0:
        # Remove dynamic line if no points clicked yet
        if dynamic_line is not None:
            dynamic_line.remove()
            dynamic_line = None
            fig.canvas.draw_idle()
        return
    
    # Determine which point we're drawing from
    pair_idx = (len(clicked_points) - 1) // 4
    point_idx = (len(clicked_points) - 1) % 4
    
    # Show dynamic line from A to cursor, or from C to cursor
    if point_idx == 0:  # A clicked, show line from A to cursor
        A = clicked_points[-1]
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            # Remove old dynamic line
            if dynamic_line is not None:
                dynamic_line.remove()
            # Draw new dynamic line
            dynamic_line, = ax.plot([A[0], x], [A[1], y], 'lime', linewidth=3, linestyle='--', alpha=0.5)
            fig.canvas.draw_idle()
        else:
            # Mouse outside axes, remove dynamic line
            if dynamic_line is not None:
                dynamic_line.remove()
                dynamic_line = None
                fig.canvas.draw_idle()
    elif point_idx == 2:  # C clicked, show line from C to cursor
        C = clicked_points[-1]
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            # Remove old dynamic line
            if dynamic_line is not None:
                dynamic_line.remove()
            # Draw new dynamic line
            dynamic_line, = ax.plot([C[0], x], [C[1], y], 'lime', linewidth=3, linestyle='--', alpha=0.5)
            fig.canvas.draw_idle()
        else:
            # Mouse outside axes, remove dynamic line
            if dynamic_line is not None:
                dynamic_line.remove()
                dynamic_line = None
                fig.canvas.draw_idle()
    else:
        # B or D clicked, no dynamic line needed
        if dynamic_line is not None:
            dynamic_line.remove()
            dynamic_line = None
            fig.canvas.draw_idle()

def select_points(image):
    """Display image and let user click 20 points using matplotlib"""
    global clicked_points, fig, ax, image_rgb, dynamic_line, permanent_lines, permanent_points
    clicked_points = []
    dynamic_line = None
    permanent_lines = []
    permanent_points = []
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title('Click 20 points (MIDDLE mouse button): A1 B1 C1 D1, A2 B2 C2 D2, A3 B3 C3 D3, A4 B4 C4 D4, A5 B5 C5 D5\n(Each quadruple forms perpendicular lines AB and CD)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect the click and motion events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    
    print("Click 20 points using MIDDLE mouse button: A1 B1 C1 D1, A2 B2 C2 D2, A3 B3 C3 D3, A4 B4 C4 D4, A5 B5 C5 D5")
    print("Each quadruple (Ai, Bi, Ci, Di) should form two lines (AiBi and CiDi) that are perpendicular in the metric space")
    print("Dynamic lines will show from A to cursor, and from C to cursor")
    plt.show()
    
    if len(clicked_points) != 20:
        print("Warning: Not all 20 points were selected!")
    
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
    Solve for the full conic C from line pairs in projective space.
    C has the form:
    C = [c11 c12 c13; c12 c22 c23; c13 c23 c33]
    
    The constraint is l^T C m = 0 for each pair (l, m).
    This expands to: l1*m1*c11 + (l1*m2 + l2*m1)*c12 + l1*m3*c13 + l2*m2*c22 + (l2*m3 + l3*m2)*c23 + l3*m3*c33 = 0
    
    Args:
        line_pairs: List of (l, m) tuples where l and m are lines in homogeneous coordinates
    
    Returns:
        C: 3x3 symmetric matrix representing the conic
    """
    # Build constraint matrix A
    # Each row corresponds to one constraint: [l1*m1, l1*m2 + l2*m1, l1*m3 + l3*m1, l2*m2, l2*m3 + l3*m2, l3*m3] · [c11, c12, c13, c22, c23, c33] = 0
    A = []
    for l, m in line_pairs:
        l1, l2, l3 = l
        m1, m2, m3 = m
        
        # Full constraint for projective space
        constraint = [
            l1 * m1,                    # coefficient for c11
            l1 * m2 + l2 * m1,          # coefficient for c12
            l1 * m3 + l3 * m1,          # coefficient for c13
            l2 * m2,                    # coefficient for c22
            l2 * m3 + l3 * m2,          # coefficient for c23
            l3 * m3                     # coefficient for c33
        ]
        A.append(constraint)
    
    A = np.array(A)
    
    # Solve Ac = 0 via SVD
    # The last row of Vt is the solution
    _, _, Vt = np.linalg.svd(A)
    c_solution = Vt[-1]
    
    # Extract c11, c12, c13, c22, c23, c33
    c11, c12, c13, c22, c23, c33 = c_solution
    
    # Construct the 3x3 conic matrix C
    C = np.array([
        [c11, c12, c13],
        [c12, c22, c23],
        [c13, c23, c33]
    ])
    
    return C

def extract_v_from_C(C):
    """
    Extract v from C where C has the structure:
    C = [KK^T    KK^T*v;
         v^T*KK^T v^T*KK^T*v]
    
    We have:
    - KKT = C[:2, :2]
    - C[:2, 2] = KKT * v, so v = KKT^-1 * C[:2, 2]
    - C[2, 2] = v^T * KKT * v (for verification)
    
    Args:
        C: 3x3 conic matrix
    
    Returns:
        v: 2x1 vector
        KKT: 2x2 matrix
    """
    KKT = C[:2, :2]
    
    # Solve KKT * v = C[:2, 2] for v
    try:
        v = np.linalg.solve(KKT, C[:2, 2])
    except np.linalg.LinAlgError:
        # If KKT is singular, use pseudo-inverse
        v = np.linalg.pinv(KKT) @ C[:2, 2]
    
    return v, KKT

def decompose_KKT_to_K(KKT):
    return decompose_KKT_to_K_numpy(KKT)
    # return decompose_KKT_to_K_manual(KKT)

def decompose_KKT_to_K_manual(KKT):
    """manual implementation of decompose with upper triangular K
    let KKT = [r s; s t]
    K = [a b;    K^T = [a 0;
         0 c]           b c]
    KKT = K * K^T = [a^2 + b^2    bc;
                     bc          c^2]
    
    solve a b c
    c = sqrt(t)
    b = s / c
    a = sqrt(r - b^2)
    
    """

    # check conditions
    if KKT[1,1] < 0:
        raise ValueError("KKT is not positive definite")

    c = np.sqrt(KKT[1,1])
    b = KKT[0,1] / c

    if b**2 > KKT[0,0]:
        raise ValueError("KKT is not positive definite")

    a = np.sqrt(KKT[0,0] - b**2)

    K = np.array([[a, b], [0, c]])

    # validate KKT
    assert np.allclose(KKT, K@K.T, atol=1e-6)

    return K


def decompose_KKT_to_K_numpy(KKT):
    """
    Decompose KKT = K * K^T where K is a 2x2 upper triangular matrix.
    This is done using Cholesky decomposition.
    
    Since KKT = K * K^T, we can use Cholesky decomposition if KKT is positive definite.
    Cholesky gives us L (lower triangular) where KKT = L * L^T.
    To get an upper triangular K, we set K = L^T.
    
    Args:
        KKT: 2x2 symmetric matrix
    
    Returns:
        K: 2x2 upper triangular matrix such that K * K^T = KKT
    """
    try:
        # Cholesky decomposition: KKT = L * L^T where L is lower triangular
        # To get upper triangular K, we set K = L^T
        L = np.linalg.cholesky(KKT)
        # K = L.T  # Transpose to get upper triangular: ERROR!!!!! L.T@L != KKT!!!!
        K = L  # Transpose to get upper triangular
        return K
    except np.linalg.LinAlgError:
        # If Cholesky fails, KKT might not be positive definite
        # Try to make it positive definite by adding a small regularization
        KKT_reg = KKT + np.eye(2) * 1e-6
        try:
            L = np.linalg.cholesky(KKT_reg)
            # K = L.T  # Transpose to get upper triangular
            K = L
            return K
        except np.linalg.LinAlgError:
            # If still fails, use SVD-based approach and make K upper triangular
            U, S, Vt = np.linalg.svd(KKT)
            S_sqrt = np.sqrt(np.maximum(S, 1e-10))
            K_temp = U @ np.diag(S_sqrt)
            K_approx = np.triu(K_temp)  # Take upper triangular part
            # Ensure the diagonal is positive and reasonable
            for i in range(2):
                if abs(K_approx[i, i]) < 1e-10:
                    K_approx[i, i] = 1e-6
                if K_approx[i, i] < 0:
                    K_approx[i, :] *= -1
            return K_approx

def construct_homography_from_K_and_v(K, v):
    """
    Construct homography H from K and v.
    H = Hp * Ha (from world to image)
    where:
    Ha = [K 0; 0 1]
    Hp = [I 0; v^T 1]
    
    Args:
        K: 2x2 upper triangular matrix
        v: 2x1 vector
    
    Returns:
        H: 3x3 homography matrix (from world to image)
    """
    # Construct Ha
    Ha = np.eye(3)
    Ha[:2, :2] = K
    
    # Construct Hp
    Hp = np.eye(3)
    Hp[2, :2] = v
    
    # H = Hp * Ha (from world to image)
    H = Hp @ Ha
    
    return H

def save_points_to_json(points, json_path):
    """
    Save points to JSON file.
    
    Args:
        points: numpy array of shape (20, 2) containing the 20 points
        json_path: path to the JSON file
    """
    # Convert numpy array to list of lists
    points_list = points.tolist()
    
    data = {
        'points': points_list,
        'num_points': len(points_list)
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Points saved to {json_path}")

def load_points_from_json(json_path):
    """
    Load points from JSON file.
    
    Args:
        json_path: path to the JSON file
    
    Returns:
        numpy array of shape (20, 2) containing the 20 points, or None if loading fails
    """
    try:
        if not os.path.exists(json_path):
            return None
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        points_list = data.get('points', [])
        num_points = data.get('num_points', len(points_list))
        
        if num_points != 20 or len(points_list) != 20:
            print(f"Warning: Expected 20 points, got {num_points}")
            return None
        
        points = np.array(points_list, dtype=np.float32)
        print(f"Points loaded from {json_path}")
        return points
    
    except Exception as e:
        print(f"Error loading points from {json_path}: {e}")
        return None

def main():
    # 1. Load the building image
    image_path = r'ch2\2.3_projective_transform\building.jpg'
    
    # Try different possible paths
    if not os.path.exists(image_path):
        # Try relative to current directory
        image_path = 'building.jpg'
    if not os.path.exists(image_path):
        image_path = os.path.join('ch2', '2.3_projective_transform', 'building.jpg')
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please make sure building.jpg exists in the ch2/2.3_projective_transform directory")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # 2. Try to load cached points, otherwise let user click 20 points
    json_path = r'ch2\2.7_recovery_affine\08_metric_recovery_by_5_pairs_of_perpendicular_lines.points.json'
    
    # Try different possible paths for JSON file
    if not os.path.exists(json_path):
        json_path = os.path.join('ch2', '2.7_recovery_affine', '08_metric_recovery_by_5_pairs_of_perpendicular_lines.points.json')
    
    points = load_points_from_json(json_path)
    
    if points is None:
        # No cached points, let user click
        print("No cached points found. Please click 20 points...")
        points = select_points(image)
        
        if len(points) != 20:
            print("Error: Need exactly 20 points")
            return
        
        # Save points to cache
        save_points_to_json(points, json_path)
    else:
        print(f"Using cached points from {json_path}")
        if len(points) != 20:
            print("Error: Cached points do not have exactly 20 points")
            return
    
    # 3. Get lines for each pair
    line_pairs = []
    
    for i in range(5):
        idx = i * 4
        A = points[idx]
        B = points[idx + 1]
        C = points[idx + 2]
        D = points[idx + 3]
        
        line_AB = line_from_points(A, B)
        line_CD = line_from_points(C, D)
        line_pairs.append((line_AB, line_CD))
    
    print(f"\nLine pairs formed: {len(line_pairs)} pairs")
    
    # 4. Solve for the full conic C
    C = solve_conic_from_line_pairs(line_pairs)
    print(f"\nConic matrix C:\n{C}")
    
    # 5. Extract v and KKT from C
    v, KKT = extract_v_from_C(C)
    print(f"\nKKT (C[:2, :2]):\n{KKT}")
    print(f"v:\n{v}")
    
    # 6. Decompose KKT = K * K^T to get K
    K = decompose_KKT_to_K(KKT)
    print(f"\nMatrix K:\n{K}")
    
    # Verify: K * K^T should be close to KKT
    KKT_reconstructed = K @ K.T
    print(f"\nVerification - K * K^T:\n{KKT_reconstructed}")
    print(f"Difference from original KKT:\n{np.abs(KKT_reconstructed - KKT)}")
    
    # 7. Construct homography H from K and v
    H = construct_homography_from_K_and_v(K, v)
    print(f"\nHomography matrix H (from world to image):\n{H}")
    
    # 8. Transform the image using H (from world to image, so we use H directly, not H^-1)
    h, w = image.shape[:2]
    # transformed_image = cv2.warpPerspective(image, np.diag([0.6, 0.6, 1]) @ np.linalg.inv(H), (w, h))
    transformed_image = cv2.warpPerspective(image, np.linalg.inv(H), (w, h))
    
    # 9. Draw line segments and points on original image for visualization
    image_with_lines = image.copy()
    
    # Color scheme: different color for each pair
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # Draw points and line segments
    for i in range(5):
        idx = i * 4
        A = points[idx]
        B = points[idx + 1]
        C = points[idx + 2]
        D = points[idx + 3]
        color = colors[i]
        
        # Draw points
        for j, (pt, label) in enumerate(zip([A, B, C, D], [f'A{i+1}', f'B{i+1}', f'C{i+1}', f'D{i+1}'])):
            pt_int = tuple(map(int, pt))
            cv2.circle(image_with_lines, pt_int, 8, color, -1)
            cv2.putText(image_with_lines, label, (pt_int[0] + 10, pt_int[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw line segments AB, CD
        A_int = tuple(map(int, A))
        B_int = tuple(map(int, B))
        C_int = tuple(map(int, C))
        D_int = tuple(map(int, D))
        cv2.line(image_with_lines, A_int, B_int, color, 2)
        cv2.line(image_with_lines, C_int, D_int, color, 2)
    
    # 10. Show the two images side by side
    fig, axes = plt.subplots(2, 1, figsize=(15, 7))
    
    # Original image with lines and points
    axes[0].imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image with 5 Perpendicular Line Pairs', fontsize=12)
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

