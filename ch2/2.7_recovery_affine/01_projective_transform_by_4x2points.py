"""
write a script

1 generate an image that have vertical lines and horizontal lines
2 let the user click 8 points in the image
the first 4 points are the source points and the last 4 are the destination points
3 calculate the projective transform matrix
4 apply the projective transform matrix to the image
5 display the original image and the transformed image

you may copy some logic from the previous exercises

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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


def generate_grid_image(width=800, height=600, grid_spacing=50):
    """Generate an image with vertical and horizontal lines (grid)"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw vertical lines
    for x in range(0, width, grid_spacing):
        cv2.line(img, (x, 0), (x, height), (0, 0, 0), 2)
    
    # Draw horizontal lines
    for y in range(0, height, grid_spacing):
        cv2.line(img, (0, y), (width, y), (0, 0, 0), 2)
    
    return img


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
        if len(clicked_points) < 8:
            x, y = event.xdata, event.ydata
            clicked_points.append([x, y])
            
            # Determine point type and color
            if len(clicked_points) <= 4:
                point_type = "Source"
                color = 'lime'
            else:
                point_type = "Dest"
                color = 'red'
            
            # Draw a circle at the clicked point
            circle = Circle((x, y), 10, color=color, fill=True)
            ax.add_patch(circle)
            ax.text(x + 15, y - 15, f"{point_type} {len(clicked_points) if len(clicked_points) <= 4 else len(clicked_points) - 4}", 
                   color=color, fontsize=12, weight='bold')
            fig.canvas.draw()
            
            print(f"{point_type} Point {len(clicked_points) if len(clicked_points) <= 4 else len(clicked_points) - 4}: ({x:.1f}, {y:.1f})")
            
            if len(clicked_points) == 4:
                print("All 4 source points selected! Now click 4 destination points...")
            elif len(clicked_points) == 8:
                print("All 8 points selected! Close the window to continue...")

def select_points(image):
    """Display image and let user click 8 points using matplotlib"""
    global clicked_points, fig, ax, image_rgb
    clicked_points = []
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title('Click 8 points:\nFirst 4 are SOURCE points (green)\nLast 4 are DESTINATION points (red)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("Click 8 points: First 4 are source points (green), last 4 are destination points (red)")
    plt.show()
    
    if len(clicked_points) != 8:
        print("Warning: Not all 8 points were selected!")
    
    return np.array(clicked_points, dtype=np.float32)

def main():
    # 1. Generate an image with vertical and horizontal lines
    print("Generating grid image with vertical and horizontal lines...")
    image = generate_grid_image(width=800, height=600, grid_spacing=50)
    print(f"Image generated: {image.shape}")
    
    # 2. Let user click 8 points
    all_points = select_points(image)
    
    if len(all_points) != 8:
        print("Error: Need exactly 8 points")
        return
    
    # Split into source and destination points
    src_points = all_points[:4]
    dst_points = all_points[4:]
    
    print(f"\nSource points:\n{src_points}")
    print(f"\nDestination points:\n{dst_points}")
    
    # 3. Calculate the projective transform matrix (homography)
    H = find_homography_normalized(src_points, dst_points)
    print(f"\nHomography matrix H:\n{H}")
    
    # 4. Apply the projective transform matrix to the image
    h, w = image.shape[:2]
    transformed_image = cv2.warpPerspective(image, H, (w, h))
    
    # 5. Display the original image and the transformed image
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image with points marked
    image_with_points = image.copy()
    # Draw source points in green
    for i, point in enumerate(src_points):
        pt = tuple(map(int, point))
        cv2.circle(image_with_points, pt, 8, (0, 255, 0), -1)
        cv2.putText(image_with_points, f"S{i+1}", (pt[0]+10, pt[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Draw destination points in red
    for i, point in enumerate(dst_points):
        pt = tuple(map(int, point))
        cv2.circle(image_with_points, pt, 8, (0, 0, 255), -1)
        cv2.putText(image_with_points, f"D{i+1}", (pt[0]+10, pt[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Convert BGR to RGB for matplotlib
    axes[0].imshow(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image with Selected Points\n(Green=Source, Red=Destination)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Transformed Image', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDone! Close the matplotlib window to exit.")

if __name__ == "__main__":
    main()
