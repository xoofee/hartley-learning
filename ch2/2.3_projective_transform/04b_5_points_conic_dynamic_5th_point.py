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
import sys

# Global variables to store clicked points
clicked_points = []
fig = None
ax = None
image_rgb = None
should_exit = False

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
            elif len(clicked_points) == 5:
                print("5th point selected! Close the window to see the result...")

def on_key(event):
    """Callback function for keyboard events"""
    global should_exit
    if event.key == 'q' or event.key == 'Q' or event.key == 'escape':
        print("\nExiting...")
        should_exit = True
        plt.close('all')

def select_4_points(image):
    """Display image and let user click 4 points using matplotlib"""
    global clicked_points, fig, ax, image_rgb, should_exit
    clicked_points = []
    should_exit = False
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title('Click 4 points on the conic\n(Press Q or ESC to exit)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect the click and key events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("Click 4 points on the conic (these will be preserved)")
    plt.show()
    
    if should_exit:
        return None
    
    if len(clicked_points) != 4:
        print("Warning: Not all 4 points were selected!")
        return None
    
    return np.array(clicked_points, dtype=np.float32)

def select_5th_point(image, first_4_points):
    """Display image with first 4 points and let user click the 5th point"""
    global clicked_points, fig, ax, image_rgb, should_exit
    clicked_points = list(first_4_points.tolist())  # Start with the 4 preserved points
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Draw the first 4 points
    for i, point in enumerate(first_4_points):
        circle = Circle((point[0], point[1]), 10, color='lime', fill=True)
        ax.add_patch(circle)
        ax.text(point[0] + 15, point[1] - 15, f"Point {i+1}", 
               color='lime', fontsize=12, weight='bold')
    
    ax.set_title('Click the 5th point on the conic\n(Press Q or ESC to exit)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect the click and key events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("Click the 5th point on the conic")
    plt.show()
    
    if should_exit:
        return None
    
    if len(clicked_points) != 5:
        print("Warning: 5th point was not selected!")
        return None
    
    return np.array(clicked_points, dtype=np.float32)

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

def show_result(image, src_points, conic):
    """Show the result with conic drawn"""
    global should_exit
    
    # Create a copy of the image for drawing
    image_with_conic = draw_conic(image.copy(), conic, color=(0, 0, 255), threshold=0.05)
    
    # Show the result
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    
    # Original image with points marked
    image_with_points = image_with_conic.copy()
    for i, point in enumerate(src_points):
        pt = tuple(map(int, point))
        cv2.circle(image_with_points, pt, 8, (0, 255, 0), -1)
        cv2.putText(image_with_points, str(i+1), (pt[0]+10, pt[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Convert BGR to RGB for matplotlib
    ax.imshow(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))
    ax.set_title('Original Image with Selected Points and Conic\n(Close window to select another 5th point, or press Q/ESC to exit)', 
                fontsize=12)
    ax.axis('off')
    
    # Connect key event for exit
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()
    
    if should_exit:
        return False
    return True

def main():
    # 1. Load the building.jpg image
    image_path = r'ch2\01_2.3_projective_transform\building.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # 2. Let user click 4 points (these will be preserved)
    first_4_points = select_4_points(image)
    
    if first_4_points is None or len(first_4_points) != 4:
        print("Error: Need exactly 4 points to start")
        return
    
    print(f"\n4 points selected and preserved:")
    for i, pt in enumerate(first_4_points):
        print(f"  Point {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # 3. Loop to select the 5th point
    iteration = 1
    while True:
        print(f"\n--- Iteration {iteration} ---")
        
        # Select the 5th point
        all_5_points = select_5th_point(image, first_4_points)
        
        if all_5_points is None:
            print("Exiting...")
            break
        
        # Calculate conic from 5 points
        conic = get_conic_from_5_point(all_5_points)
        print(f"Conic calculated: {conic}")
        
        # Show the result
        continue_loop = show_result(image, all_5_points, conic)
        
        if not continue_loop:
            print("Exiting...")
            break
        
        iteration += 1
    
    print("\nDone!")

if __name__ == "__main__":
    main()
