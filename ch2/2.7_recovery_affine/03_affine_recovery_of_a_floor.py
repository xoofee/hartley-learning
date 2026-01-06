"""
1 load the floor.jpg image
2 pop out the window to let the user click 4 points, say, A, B, C, D
3 calculate the cross point of AB and CD, say, E
4 calculate the cross point of BC and AD, say, F
5 calculate the line EF by the two points E and F, using cross product
the line will be [l1, l2, l3]
6 get H by
[1, 0, 0
 0, 1, 0
 l1, l2, l3]
7 transform the image using the H
8 show the two image side by side
draw the line AB, BC, CD, DA, EF in the original image.

NOTE: THE E F will be outside of the image, so you need to extend the image to include the line EF.

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
        if len(clicked_points) < 4:
            x, y = event.xdata, event.ydata
            clicked_points.append([x, y])
            
            # Label points as A, B, C, D
            labels = ['A', 'B', 'C', 'D']
            label = labels[len(clicked_points) - 1]
            
            # Draw a circle at the clicked point
            circle = Circle((x, y), 10, color='lime', fill=True)
            ax.add_patch(circle)
            ax.text(x + 15, y - 15, label, 
                   color='lime', fontsize=12, weight='bold')
            fig.canvas.draw()
            
            print(f"Point {label}: ({x:.1f}, {y:.1f})")
            
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
    ax.set_title('Click 4 points: A, B, C, D\n(in order: A, B, C, D)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("Click 4 points: A, B, C, D")
    plt.show()
    
    if len(clicked_points) != 4:
        print("Warning: Not all 4 points were selected!")
    
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

def line_intersection(line1, line2):
    """Calculate intersection of two lines using cross product"""
    intersection = np.cross(line1, line2)
    if abs(intersection[2]) > 1e-10:
        intersection = intersection / intersection[2]
        return intersection[:2]  # Return as 2D point
    else:
        # Lines are parallel
        return None

def extend_image_to_include_line(image, line, margin=50):
    """Extend image to include a line that may be outside the image"""
    h, w = image.shape[:2]
    
    # Calculate line endpoints at image boundaries
    # Line equation: l1*x + l2*y + l3 = 0
    l1, l2, l3 = line
    
    # Find intersections with image boundaries
    intersections = []
    
    # Top edge (y = 0)
    if abs(l2) > 1e-10:
        x_top = -l3 / l1 if abs(l1) > 1e-10 else None
        if x_top is not None and -margin <= x_top <= w + margin:
            intersections.append((x_top, 0))
    
    # Bottom edge (y = h)
    if abs(l2) > 1e-10:
        x_bottom = (-l2 * h - l3) / l1 if abs(l1) > 1e-10 else None
        if x_bottom is not None and -margin <= x_bottom <= w + margin:
            intersections.append((x_bottom, h))
    
    # Left edge (x = 0)
    if abs(l1) > 1e-10:
        y_left = -l3 / l2 if abs(l2) > 1e-10 else None
        if y_left is not None and -margin <= y_left <= h + margin:
            intersections.append((0, y_left))
    
    # Right edge (x = w)
    if abs(l1) > 1e-10:
        y_right = (-l1 * w - l3) / l2 if abs(l2) > 1e-10 else None
        if y_right is not None and -margin <= y_right <= h + margin:
            intersections.append((w, y_right))
    
    if not intersections:
        return image, 0, 0  # No extension needed
    
    # Find bounding box of intersections
    x_coords = [p[0] for p in intersections]
    y_coords = [p[1] for p in intersections]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    # Calculate padding needed
    pad_left = max(0, margin - min_x)
    pad_right = max(0, max_x - (w - margin))
    pad_top = max(0, margin - min_y)
    pad_bottom = max(0, max_y - (h - margin))
    
    # Extend image
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        extended = cv2.copyMakeBorder(image, 
                                     int(pad_top), int(pad_bottom),
                                     int(pad_left), int(pad_right),
                                     cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return extended, int(pad_left), int(pad_top)
    
    return image, 0, 0

def draw_line_on_image(image, line, color=(0, 0, 255), thickness=2):
    """Draw a line on the image"""
    h, w = image.shape[:2]
    l1, l2, l3 = line
    
    # Find intersections with image boundaries
    points = []
    
    # Top edge (y = 0)
    if abs(l1) > 1e-10:
        x = -l3 / l1
        if 0 <= x <= w:
            points.append((int(x), 0))
    
    # Bottom edge (y = h)
    if abs(l1) > 1e-10:
        x = (-l2 * h - l3) / l1
        if 0 <= x <= w:
            points.append((int(x), h))
    
    # Left edge (x = 0)
    if abs(l2) > 1e-10:
        y = -l3 / l2
        if 0 <= y <= h:
            points.append((0, int(y)))
    
    # Right edge (x = w)
    if abs(l2) > 1e-10:
        y = (-l1 * w - l3) / l2
        if 0 <= y <= h:
            points.append((w, int(y)))
    
    # Draw line if we have two points
    if len(points) >= 2:
        cv2.line(image, points[0], points[1], color, thickness)
    elif len(points) == 1:
        # Line goes through one corner, extend to opposite corner
        if points[0][0] == 0:
            cv2.line(image, points[0], (w, points[0][1]), color, thickness)
        elif points[0][0] == w:
            cv2.line(image, points[0], (0, points[0][1]), color, thickness)
        elif points[0][1] == 0:
            cv2.line(image, points[0], (points[0][0], h), color, thickness)
        elif points[0][1] == h:
            cv2.line(image, points[0], (points[0][0], 0), color, thickness)
    
    return image

def main():
    # 1. Load the floor.jpg image
    image_path = r'ch2\2.7_recovery_affine\floor.jpg'
    
    # Try different possible paths
    if not os.path.exists(image_path):
        # Try relative to current directory
        image_path = 'floor.jpg'
    if not os.path.exists(image_path):
        image_path = os.path.join('ch2', '2.7_recovery_affine', 'floor.jpg')
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please make sure floor.jpg exists in the ch2/2.7_recovery_affine directory")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # 2. Let user click 4 points (A, B, C, D)
    points = select_points(image)
    
    if len(points) != 4:
        print("Error: Need exactly 4 points")
        return
    
    A, B, C, D = points[0], points[1], points[2], points[3]
    
    print(f"\nPoints selected:")
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"C: {C}")
    print(f"D: {D}")
    
    # 3. Calculate intersection E = AB ∩ CD
    line_AB = line_from_points(A, B)
    line_CD = line_from_points(C, D)
    E = line_intersection(line_AB, line_CD)
    
    if E is None:
        print("Error: Lines AB and CD are parallel!")
        return
    
    print(f"\nIntersection E (AB ∩ CD): {E}")
    
    # 4. Calculate intersection F = BC ∩ AD
    line_BC = line_from_points(B, C)
    line_AD = line_from_points(A, D)
    F = line_intersection(line_BC, line_AD)
    
    if F is None:
        print("Error: Lines BC and AD are parallel!")
        return
    
    print(f"Intersection F (BC ∩ AD): {F}")
    
    # 5. Calculate line EF using cross product
    line_EF = line_from_points(E, F)
    print(f"\nLine EF: {line_EF}")
    
    # 6. Get H matrix
    l1, l2, l3 = line_EF
    H = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [l1, l2, l3]
    ], dtype=np.float32)
    
    print(f"\nHomography matrix H:\n{H}")
    
    # Extend image to include line EF if needed
    extended_image, pad_x, pad_y = extend_image_to_include_line(image.copy(), line_EF)
    
    # Adjust points for extended image
    A_ext = A + np.array([pad_x, pad_y])
    B_ext = B + np.array([pad_x, pad_y])
    C_ext = C + np.array([pad_x, pad_y])
    D_ext = D + np.array([pad_x, pad_y])
    E_ext = E + np.array([pad_x, pad_y])
    F_ext = F + np.array([pad_x, pad_y])
    
    # Draw lines on extended image
    image_with_lines = extended_image.copy()
    
    # Draw lines AB, BC, CD, DA
    line_AB_ext = line_from_points(A_ext, B_ext)
    line_BC_ext = line_from_points(B_ext, C_ext)
    line_CD_ext = line_from_points(C_ext, D_ext)
    line_DA_ext = line_from_points(D_ext, A_ext)
    
    image_with_lines = draw_line_on_image(image_with_lines, line_AB_ext, color=(0, 255, 0), thickness=2)
    image_with_lines = draw_line_on_image(image_with_lines, line_BC_ext, color=(0, 255, 0), thickness=2)
    image_with_lines = draw_line_on_image(image_with_lines, line_CD_ext, color=(0, 255, 0), thickness=2)
    image_with_lines = draw_line_on_image(image_with_lines, line_DA_ext, color=(0, 255, 0), thickness=2)
    
    # Draw line EF
    line_EF_ext = line_from_points(E_ext, F_ext)
    image_with_lines = draw_line_on_image(image_with_lines, line_EF_ext, color=(0, 0, 255), thickness=3)
    
    # Draw points
    for pt, label in [(A_ext, 'A'), (B_ext, 'B'), (C_ext, 'C'), (D_ext, 'D')]:
        pt_int = tuple(map(int, pt))
        cv2.circle(image_with_lines, pt_int, 8, (0, 255, 0), -1)
        cv2.putText(image_with_lines, label, (pt_int[0] + 10, pt_int[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw intersection points E and F
    for pt, label in [(E_ext, 'E'), (F_ext, 'F')]:
        pt_int = tuple(map(int, pt))
        cv2.circle(image_with_lines, pt_int, 8, (255, 0, 0), -1)
        cv2.putText(image_with_lines, label, (pt_int[0] + 10, pt_int[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 7. Transform the image using H
    h_ext, w_ext = extended_image.shape[:2]
    transformed_image = cv2.warpPerspective(extended_image, H, (w_ext, h_ext))
    
    # Transform points A, B, C, D to the transformed image
    def transform_point(pt, H):
        """Transform a 2D point using homography matrix H"""
        pt_h = np.array([pt[0], pt[1], 1.0])
        pt_transformed_h = H @ pt_h
        if abs(pt_transformed_h[2]) > 1e-10:
            pt_transformed = pt_transformed_h[:2] / pt_transformed_h[2]
            return pt_transformed
        return None
    
    A_transformed = transform_point(A_ext, H)
    B_transformed = transform_point(B_ext, H)
    C_transformed = transform_point(C_ext, H)
    D_transformed = transform_point(D_ext, H)
    
    # Draw points and labels on transformed image
    transformed_image_with_points = transformed_image.copy()
    for pt, label in [(A_transformed, 'A'), (B_transformed, 'B'), 
                      (C_transformed, 'C'), (D_transformed, 'D')]:
        if pt is not None:
            pt_int = tuple(map(int, pt))
            # Check if point is within image bounds
            if 0 <= pt_int[0] < w_ext and 0 <= pt_int[1] < h_ext:
                cv2.circle(transformed_image_with_points, pt_int, 8, (0, 255, 0), -1)
                cv2.putText(transformed_image_with_points, label, (pt_int[0] + 10, pt_int[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 8. Show the two images side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image with lines
    axes[0].imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image with Lines\n(Green: AB, BC, CD, DA; Red: EF)', fontsize=12)
    axes[0].axis('off')
    
    # Transformed image with points
    axes[1].imshow(cv2.cvtColor(transformed_image_with_points, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Affine Recovered Image with Points', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDone! Close the matplotlib window to exit.")

if __name__ == "__main__":
    main()
