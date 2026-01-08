"""
1 load the image like in ch2\2.7_recovery_affine\03_affine_recovery_of_a_floor.py
2 let the user click 3 colinear points on the image, say, A, B, C, note B is between A and C
3 a = AB , b = BC in eucludian distance
4 A is origin
5 so we have three point in 1d space. in homogeneous coordinates, they are [0, 1], [a, 1], [a+b, 1]
6 let the user input an original ratio in world coordinate say, r for BC/BA in world coordinate
   the world coordinate is [0, 1], [1, 1], [1+r, 1]
7 so we have three pair points, which could be used to find the homography matrix H (2x2)

refer to this function, adapter it to 2d matrix

def findHomography(src_points, dst_points):
    print(f"src_points: {src_points}")
    print(f"dst_points: {dst_points}")
    # H, _ = cv2.findHomography(src_points, dst_points)

    xs = src_points[:, 0]
    ys = src_points[:, 1]

    xd = dst_points[:, 0]
    yd = dst_points[:, 1]

    A = np.zeros((8, 9))
    for i in range(4):
       A[i*2, :] = [xs[i], ys[i], 1, 0, 0, 0, -xd[i]*xs[i], -xd[i]*ys[i], -xd[i]]
       A[i*2+1, :] = [0, 0, 0, xs[i], ys[i], 1, -yd[i]*xs[i], -yd[i]*ys[i], -yd[i]]
    
    H = np.linalg.solve(A[:, :8], -A[:, 8])
    H = np.concatenate([H, [1]])
    H = H.reshape(3, 3)
        
    # U, S, Vt = np.linalg.svd(A)
    # H = Vt[-1, :].reshape(3, 3)
    # H = H / H[2, 2]

    return H


7 use H to convert the point at infinite [1, 0] to the 1d line coordinate, say, [p, q]
then the coordinate of the point at infinite is [p/q, 1]
use A [0, 1] and B [a, 1] along with the image pixel point to interpolate the pixel point for [p/q, 1]

8 print the interpolated point pixel coordinate.

"""

import cv2
import numpy as np
import os

import matplotlib
matplotlib.use("TkAgg")
# to avoid on windows
# File "c:\App\Python\env\xf\Lib\site-packages\matplotlib\backends\backend_qt.py", line 166, in _may_clear_sock rsock.recv(1) OSError: [WinError 10038] 在一个非套接字上尝试了一个操作。

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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
        if len(clicked_points) < 3:
            x, y = event.xdata, event.ydata
            clicked_points.append([x, y])
            
            # Label points as A, B, C
            labels = ['A', 'B', 'C']
            label = labels[len(clicked_points) - 1]
            
            # Draw a circle at the clicked point
            circle = Circle((x, y), 10, color='lime', fill=True)
            ax.add_patch(circle)
            ax.text(x + 15, y - 15, label, 
                   color='lime', fontsize=12, weight='bold')
            fig.canvas.draw()
            
            print(f"Point {label}: ({x:.1f}, {y:.1f})")
            
            if len(clicked_points) == 3:
                print("All 3 points selected! Close the window to continue...")

def select_points(image):
    """Display image and let user click 3 collinear points using matplotlib"""
    global clicked_points, fig, ax, image_rgb
    clicked_points = []
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title('Click 3 collinear points: A, B, C\n(B is between A and C)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("Click 3 collinear points: A, B, C (B is between A and C)")
    plt.show()
    
    if len(clicked_points) != 3:
        print("Warning: Not all 3 points were selected!")
    
    return np.array(clicked_points, dtype=np.float32)

def find_homography_1d(src_points, dst_points):
    """
    Find 1D homography matrix H (2x2) from 3 point correspondences using DLT (Direct Linear Transform) with SVD.
    
    For 1D homography: [x', w']^T = H * [x, w]^T where H is 2x2
    x' = h11*x + h12*w
    w' = h21*x + h22*w
    
    Using the constraint: x'*(h21*x + h22*w) = w'*(h11*x + h12*w)
    Rearranging: x'*h21*x + x'*h22*w - w'*h11*x - w'*h12*w = 0
    
    This gives us one constraint per point: [-w'*x, -w'*w, x'*x, x'*w] * [h11, h12, h21, h22]^T = 0
    """
    # Build constraint matrix A (3x4) for Ah = 0
    # where h = [h11, h12, h21, h22]^T
    A = np.zeros((3, 4))
    
    for i in range(3):
        x = src_points[i, 0]
        w = src_points[i, 1]
        xp = dst_points[i, 0]
        wp = dst_points[i, 1]
        
        # Constraint: x'*(h21*x + h22*w) = w'*(h11*x + h12*w)
        # Rearranged: -w'*x*h11 - w'*w*h12 + x'*x*h21 + x'*w*h22 = 0
        A[i, 0] = -wp * x   # coefficient for h11
        A[i, 1] = -wp * w   # coefficient for h12
        A[i, 2] = xp * x    # coefficient for h21
        A[i, 3] = xp * w    # coefficient for h22
    
    # Solve Ah = 0 using SVD
    # The solution is the right singular vector corresponding to the smallest singular value
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # Last row of Vt (corresponds to smallest singular value)
    
    # Reshape to 2x2 matrix
    H = h.reshape(2, 2)
    
    # Normalize (optional, but good practice)
    if abs(H[1, 1]) > 1e-10:
        H = H / H[1, 1]
    
    return H

def main():
    # 1. Load the image
    image_path = r'ch2\2.7_recovery_affine\floor.jpg'
    
    # Try different possible paths
    if not os.path.exists(image_path):
        image_path = 'floor.jpg'
    if not os.path.exists(image_path):
        image_path = os.path.join('ch2', '2.7_recovery_affine', 'floor.jpg')
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please make sure floor.jpg exists in the ch2/2.7_recovery_affine directory")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # 2. Let user click 3 collinear points (A, B, C)
    points = select_points(image)
    
    if len(points) != 3:
        print("Error: Need exactly 3 points")
        return
    
    A, B, C = points[0], points[1], points[2]
    
    print(f"\nPoints selected:")
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")
    
    # 3. Calculate distances: a = AB, b = BC
    a = np.linalg.norm(B - A)
    b = np.linalg.norm(C - B)
    
    print(f"\nDistances:")
    print(f"a = AB = {a:.2f}")
    print(f"b = BC = {b:.2f}")
    
    # 4. A is origin, so in 1D homogeneous coordinates:
    # [0, 1], [a, 1], [a+b, 1]
    projected_points_1d = np.array([
        [0, 1],      # A
        [a, 1],      # B
        [a + b, 1]   # C
    ], dtype=np.float64)
    
    print(f"\n1D image coordinates (homogeneous):")
    print(f"A: {projected_points_1d[0]}")
    print(f"B: {projected_points_1d[1]}")
    print(f"C: {projected_points_1d[2]}")
    
    # 6. Get user input for ratio r = BC/BA in world coordinates
    print("\nEnter the ratio r = BC/BA in world coordinates:")
    r = float(input("r = "))
    
    # World coordinates: [0, 1], [1, 1], [1+r, 1]
    # A at 0, B at 1, C at 1+r
    # So BA = 1, BC = r, and BC/BA = r
    world_points_1d = np.array([
        [0, 1],      # A
        [1, 1],      # B
        [1 + r, 1]   # C
    ], dtype=np.float64)
    
    print(f"\n1D world coordinates (homogeneous):")
    print(f"A: {world_points_1d[0]}")
    print(f"B: {world_points_1d[1]}")
    print(f"C: {world_points_1d[2]}")
    print(f"Note: BC/BA in world = {r}")
    
    # 7. Find 2x2 homography matrix H
    H_1d = find_homography_1d(world_points_1d, projected_points_1d)
    print(f"\n1D Homography matrix H (2x2):\n{H_1d}")
    
    # 7. Use H to convert point at infinity [1, 0] to 1D line coordinate
    point_infinity_1d_world = np.array([1.0, 0.0])  # Point at infinity
    point_transformed_1d_image = H_1d @ point_infinity_1d_world
    p, q = point_transformed_1d_image[0], point_transformed_1d_image[1]
    
    print(f"\nPoint at infinity [1, 0] transformed to [{p:.6f}, {q:.6f}]")
    
    if abs(q) > 1e-10:
        point_infinity_coord_1d_image = p / q
        print(f"Coordinate of point at infinity: [{point_infinity_coord_1d_image:.6f}, 1]")
        
        # 8. Interpolate pixel coordinate for [p/q, 1] using A and B
        # In 1D image coordinates: A is at [0, 1] (pixel A), B is at [a, 1] (pixel B)
        # We need to find the pixel coordinate for [point_infinity_coord, 1] in world coordinates
        
        
        # Now interpolate the actual pixel coordinate
        # We know: A (pixel) corresponds to 0 in 1D, B (pixel) corresponds to a in 1D
        # So if point is at coord in 1D, we interpolate:
        # pixel = A + (coord / a) * (B - A)
        if abs(a) > 1e-10:
            pixel_coord = A + (point_infinity_coord_1d_image / a) * (B - A)
            print(f"\nInterpolated pixel coordinate: ({pixel_coord[0]:.2f}, {pixel_coord[1]:.2f})")
            
            # Calculate bounding box to include image and vanishing point
            h, w = image.shape[:2]
            margin = 50  # Extra margin around the image
            
            # Find min/max coordinates
            all_points = [A, B, C, pixel_coord]
            min_x = min(p[0] for p in all_points) - margin
            max_x = max(p[0] for p in all_points) + margin
            min_y = min(p[1] for p in all_points) - margin
            max_y = max(p[1] for p in all_points) + margin
            
            # Ensure image bounds are included
            min_x = min(min_x, 0)
            max_x = max(max_x, w)
            min_y = min(min_y, 0)
            max_y = max(max_y, h)
            
            # Calculate padding needed
            pad_left = max(0, int(-min_x))
            pad_top = max(0, int(-min_y))
            pad_right = max(0, int(max_x - w))
            pad_bottom = max(0, int(max_y - h))
            
            # Create extended canvas
            extended_h = h + pad_top + pad_bottom
            extended_w = w + pad_left + pad_right
            extended_image = np.ones((extended_h, extended_w, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Place original image on extended canvas
            extended_image[pad_top:pad_top+h, pad_left:pad_left+w] = image
            
            # Adjust all coordinates for extended canvas
            A_ext = A + np.array([pad_left, pad_top])
            B_ext = B + np.array([pad_left, pad_top])
            C_ext = C + np.array([pad_left, pad_top])
            vp_ext = pixel_coord + np.array([pad_left, pad_top])
            
            # Draw the three points
            for pt, label in [(A_ext, 'A'), (B_ext, 'B'), (C_ext, 'C')]:
                pt_int = tuple(map(int, pt))
                cv2.circle(extended_image, pt_int, 8, (0, 255, 0), -1)
                cv2.putText(extended_image, label, (pt_int[0] + 10, pt_int[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw the vanishing point
            vp_int = tuple(map(int, vp_ext))
            cv2.circle(extended_image, vp_int, 12, (0, 0, 255), -1)
            cv2.circle(extended_image, vp_int, 12, (255, 255, 255), 2)  # White outline
            cv2.putText(extended_image, 'VP', (vp_int[0] + 15, vp_int[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw extended line through A, B, C to vanishing point
            # Calculate line direction from A to B
            line_dir = B_ext - A_ext
            line_dir = line_dir / np.linalg.norm(line_dir)
            
            # Extend line from C towards vanishing point (or backwards if VP is before A)
            # Find which direction to extend
            if np.dot(vp_ext - C_ext, line_dir) > 0:
                # VP is in front of C, extend forward
                line_end = vp_ext
                line_start = A_ext - line_dir * 200  # Extend backwards from A
            else:
                # VP is behind A, extend backwards
                line_start = vp_ext
                line_end = C_ext + line_dir * 200  # Extend forward from C
            
            # Draw the extended line
            cv2.line(extended_image, tuple(map(int, line_start)), tuple(map(int, line_end)), (255, 0, 0), 2)
            
            # Draw a thicker line segment through the original image area
            # Find intersection with image boundaries
            img_bounds = [
                (pad_left, pad_top),  # top-left
                (pad_left + w, pad_top),  # top-right
                (pad_left + w, pad_top + h),  # bottom-right
                (pad_left, pad_top + h)  # bottom-left
            ]
            
            # Draw line segment within image bounds with different color
            cv2.line(extended_image, tuple(map(int, A_ext)), tuple(map(int, C_ext)), (0, 0, 255), 3)
            
            # Display
            plt.figure(figsize=(14, 10))
            plt.imshow(cv2.cvtColor(extended_image, cv2.COLOR_BGR2RGB))
            plt.title('Vanishing Point from Ratio\n(Green: A, B, C; Red: Vanishing Point; Blue line: extended line)', 
                     fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("Error: Distance a is too small")

    else:
        print("Error: Transformed point at infinity has zero q component")
    
    print("\nDone!")

if __name__ == "__main__":
    main()