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
        if len(clicked_points) < 5:
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
    
    print("Click 5 points on an conic")
    plt.show()
    
    if len(clicked_points) != 5:
        print("Warning: Not all 5 points were selected!")
    
    return np.array(clicked_points, dtype=np.float32)


def draw_line(image, line, color=(0, 0, 255), thickness=2):
    """Draw a line on the image"""
    
    w, h = image.shape[:2]

    x1 = 0
    y1 = -line[2] / line[1]
    x2 = w
    y2 = (-line[2] - line[0] * x2) / line[1]

    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return image


def draw_conic(image, conic, color=(0, 0, 255), threshold=0.02):
    """Draw a conic on the image"""
    
    w, h = image.shape[:2]

    x = range(h)
    y = range(w)
    X, Y = np.meshgrid(x, y)
    a = conic[0, 0]
    b = conic[0, 1]*2
    c = conic[1, 1]
    d = conic[0, 2]*2
    e = conic[1, 2]*2
    f = conic[2, 2]
    values = a * X**2 + b * X * Y + c * Y**2 + d * X + e * Y + f

    mask = np.bitwise_and(values < threshold, values >= 0.0)
    image[mask] = color

    return image

def get_conic_from_5_point(points):
    """Get a conic from 5 points"""
    A = np.zeros((5, 6))
    for i in range(5):
        x = points[i, 0]
        y = points[i, 1]
        A[i, :] = [x**2, x*y, y**2, x, y, 1]
    
    U, S, Vt = np.linalg.svd(A)
    a,b,c,d,e,f = Vt[-1, :]
    
    return np.array([[a, b/2, d/2],
                     [b/2, c, e/2],
                     [d/2, e/2, f]])

def main():
    # 1. Load the building.jpg image
    image_path = r'ch2\01_2.3_projective_transform\building.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")

    # # let's define two lines in the image
    # h, w= image.shape[:2]

    # r1 = h//3
    # r2 = h//4
    # cx = h/2
    # cy = h/2
    # # （x-cx)^2/r1^2 + (y-cy)^2/r2^2 = 1
    # # ax^2+bxy+cy^2+dx+ey+f = 0
    # a = 1/r1**2
    # b = 0
    # c = 1/r2**2
    # d = -2*cx/r1**2
    # e = -2*cy/r2**2
    # f = cx**2/r1**2 + cy**2/r2**2 - 1

    # conic = np.array([
    #     [a, b/2, d/2],
    #     [b/2, c, e/2],
    #     [d/2, e/2, f]
    # ])

    # print(f"conic: {conic}")

    # image = draw_conic(image, conic, color=(0, 0, 255), threshold=0.05)
    
    # 2. Let user click 5 points
    src_points = select_points(image)
    
    if len(src_points) != 5:
        print("Error: Need exactly 5 points")
        return
    
    conic = get_conic_from_5_point(src_points)
    print(f"conic: {conic}")
    image = draw_conic(image, conic, color=(0, 0, 255), threshold=0.05)

    p0 = np.array([src_points[0, 0], src_points[0, 1], 1])
    line = conic @ p0
    image = draw_line(image, line, color=(0, 255, 0), thickness=2)


    # 6. Show the two images side by side using matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    
    # Original image with points marked
    image_with_points = image.copy()
    for i, point in enumerate(src_points):
        pt = tuple(map(int, point))
        cv2.circle(image_with_points, pt, 8, (0, 255, 0), -1)
        cv2.putText(image_with_points, str(i+1), (pt[0]+10, pt[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert BGR to RGB for matplotlib
    ax.imshow(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))
    ax.set_title('Original Image with Selected Points', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDone! Close the matplotlib window to exit.")

if __name__ == "__main__":
    main()
