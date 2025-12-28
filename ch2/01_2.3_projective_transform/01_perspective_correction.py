"""

write a python script to 
1 load the building.jpg image
2 pop out the window to let the user click 4 points
the point will be the corner of a window but projectively imaged near vertically
3 set the destionation points near the original position but no perspective distortion
4 calculate the H matrix
5 transform the image using the H
6 show the two image side by side
you may using opencv and matplotlit in python

"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np

def normalize_points(points):
    """
    Hartley normalization for 2D points.
    points: (N, 2)
    Returns:
        points_norm: (N, 2)
        T: (3, 3) normalization matrix
    """
    centroid = np.mean(points, axis=0)

    shifted = points - centroid
    dist = np.sqrt(np.sum(shifted**2, axis=1))
    mean_dist = np.mean(dist)

    scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0,     -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,     0,      1]
    ])

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_norm_h = (T @ points_h.T).T

    return points_norm_h[:, :2], T


def find_homography_normalized(src_points, dst_points):
    """
    Normalized DLT homography estimation.
    src_points, dst_points: (N, 2), N >= 4
    Returns:
        H: (3, 3) homography matrix
    """
    assert src_points.shape == dst_points.shape
    assert src_points.shape[0] >= 4

    # 1. Normalize points
    src_norm, T_src = normalize_points(src_points)
    dst_norm, T_dst = normalize_points(dst_points)

    # 2. Build DLT system
    N = src_points.shape[0]
    A = np.zeros((2 * N, 9))

    for i in range(N):
        x, y = src_norm[i]
        u, v = dst_norm[i]

        A[2*i]   = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]

    # 3. Solve Ah = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # 4. Denormalize
    H = np.linalg.inv(T_dst) @ H_norm @ T_src

    # 5. Normalize scale (optional)
    H /= H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)

    return H




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
        if len(clicked_points) < 4:
            x, y = event.xdata, event.ydata
            clicked_points.append([x, y])
            
            # Draw a circle at the clicked point
            circle = Circle((x, y), 10, color='lime', fill=True)
            ax.add_patch(circle)
            ax.text(x + 15, y - 15, f"Point {len(clicked_points)}", 
                   color='lime', fontsize=12, weight='bold')
            fig.canvas.draw()
            
            print(f"Point {len(clicked_points)}: ({x:.1f}, {y:.1f})")
            
            if len(clicked_points) == 4:
                print("All 4 points selected! Close the window to continue...")

def select_points(image):
    """Display image and let user click 4 points using matplotlib"""
    global clicked_points, fig, ax, image_rgb
    clicked_points = []
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title('Click 4 points on the window corners\n(in order: top-left, top-right, bottom-right, bottom-left)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("Click 4 points on the window corners (in order: top-left, top-right, bottom-right, bottom-left)")
    plt.show()
    
    if len(clicked_points) != 4:
        print("Warning: Not all 4 points were selected!")
    
    return np.array(clicked_points, dtype=np.float32)

def calculate_destination_points(src_points):
    """Calculate destination points to form a rectangle without perspective distortion"""
    # Calculate the width and height of the bounding rectangle
    x_coords = src_points[:, 0]
    y_coords = src_points[:, 1]
    
    width = max(np.linalg.norm(src_points[1] - src_points[0]),
                np.linalg.norm(src_points[2] - src_points[3]))
    height = max(np.linalg.norm(src_points[3] - src_points[0]),
                 np.linalg.norm(src_points[2] - src_points[1]))
    
    # Create a rectangular destination (assuming points are in order: top-left, top-right, bottom-right, bottom-left)
    # If points are not in perfect order, we'll use the bounding box approach
    x_min, y_min = np.min(src_points, axis=0)
    x_max, y_max = np.max(src_points, axis=0)
    
    # Create rectangular destination points
    dst_points = np.array([
        [x_min, y_min],           # top-left
        [x_max, y_min],           # top-right
        [x_max, y_max],           # bottom-right
        [x_min, y_max]            # bottom-left
    ], dtype=np.float32)
    
    return dst_points

def main():
    # 1. Load the building.jpg image
    image_path = r'ch2\01_2.3_perspetive_correction\building.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # 2. Let user click 4 points
    src_points = select_points(image)
    
    if len(src_points) != 4:
        print("Error: Need exactly 4 points")
        return
    
    # 3. Calculate destination points (rectangular, no perspective distortion)
    dst_points = calculate_destination_points(src_points)
    
    # 4. Calculate the homography matrix H
    # H = findHomography(src_points, dst_points)
    H = find_homography_normalized(src_points, dst_points)
    print(f"\nHomography matrix H:\n{H}")
    
    # 5. Transform the image using H
    h, w = image.shape[:2]
    corrected_image = cv2.warpPerspective(image, H, (w, h))
    
    # 6. Show the two images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image with points marked
    image_with_points = image.copy()
    for i, point in enumerate(src_points):
        pt = tuple(map(int, point))
        cv2.circle(image_with_points, pt, 8, (0, 255, 0), -1)
        cv2.putText(image_with_points, str(i+1), (pt[0]+10, pt[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert BGR to RGB for matplotlib
    axes[0].imshow(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image with Selected Points', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Perspective Corrected Image', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDone! Close the matplotlib window to exit.")

if __name__ == "__main__":
    main()
