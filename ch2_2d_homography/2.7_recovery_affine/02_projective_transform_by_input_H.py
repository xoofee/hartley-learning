"""
write a script

1 generate an image that have vertical lines and horizontal lines
2 define the projective transform matrix H directly in code
3 apply the projective transform matrix to the image
4 display the original image and the transformed image

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def draw_line(image, line, color=(0, 0, 255), thickness=2):
    """Draw a line on the image"""
    
    w, h = image.shape[:2]

    x1 = 0
    y1 = -line[2] / line[1]
    x2 = w
    y2 = (-line[2] - line[0] * x2) / line[1]

    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return image

def main():
    # 1. Generate an image with vertical and horizontal lines
    print("Generating grid image with vertical and horizontal lines...")
    image = generate_grid_image(width=800, height=600, grid_spacing=50)
    print(f"Image generated: {image.shape}")
    
    # 2. Define the projective transform matrix H directly in code
    # Example: A perspective transformation matrix
    # You can modify these values to see different transformations
    # H = np.array([
    #     [1.2,  0.1,  50],   # Scale x, shear, translation x
    #     [0.05, 1.1,  30],   # Shear, scale y, translation y
    #     [0.0002, 0.0001, 1.0]  # Perspective parameters
    # ], dtype=np.float32)
    H = np.array([
        [1.0,  0.0,  0.0],   # Scale x, shear, translation x
        [0.0,  1.0,  0.0],   # Shear, scale y, translation y
        [1/600.0, 1/800.0, 1.0]  # Perspective parameters
    ], dtype=np.float32)
    
    print(f"\nHomography matrix H:\n{H}")
    
    # Draw text at the 9 Go board positions (corners, edge centers, and center)
    h, w = image.shape[:2]
    
    # Define the 9 positions (like Go board star points)
    positions = [
        (20, 20, "TL"),      # Top-left corner
        (w//2, 20, "TC"),    # Top center
        (w-20, 20, "TR"),    # Top-right corner
        (20, h//2, "LC"),    # Left center
        (w//2, h//2, "C"),   # Center
        (w-20, h//2, "RC"),  # Right center
        (20, h-20, "BL"),    # Bottom-left corner
        (w//2, h-20, "BC"),  # Bottom center
        (w-20, h-20, "BR"),  # Bottom-right corner
    ]
    
    # Draw text at each position
    for x, y, label in positions:
        cv2.putText(image, label, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    # 3. Apply the projective transform matrix to the image
    transformed_image = cv2.warpPerspective(image, H, (w, h))
    
    l_inf = np.array([0, 0, 1])
    l_inf_transformed = np.linalg.inv(H).T @ l_inf
    print(f"l_inf_transformed: {l_inf_transformed}")
    transformed_image = draw_line(transformed_image, l_inf_transformed, color=(0, 0, 255), thickness=2)

    # 4. Display the original image and the transformed image
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Transformed image
    axes[1].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Transformed Image', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDone! Close the matplotlib window to exit.")

if __name__ == "__main__":
    main()
