"""
Illustration of conic transformation under projective transformation.

According to section 2.3.1:
- Under point transformation x' = Hx, a conic C transforms as C' = H^(-T) * C * H^(-1)
- Conics transform covariantly (presence of H^(-1))
- Dual conic C* transforms as C*' = H * C* * H^T

This script demonstrates:
1. Create conics (circles and ellipses) in the original space
2. Apply a projective transformation H
3. Show how conics transform using C' = H^(-T) * C * H^(-1)
4. Visualize both original and transformed conics
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Ellipse

def create_homography_matrix():
    """Create a projective transformation matrix H"""
    # Create a homography that includes perspective distortion
    H = np.array([
        [1.3, 0.2, 60],
        [0.15, 1.2, 40],
        [0.0015, 0.001, 1.0]
    ], dtype=np.float32)
    return H

def transform_conic(C, H):
    """
    Transform a conic using C' = H^(-T) * C * H^(-1)
    
    Args:
        C: Conic matrix (3x3 symmetric matrix)
        H: Homography matrix (3x3)
    
    Returns:
        C': Transformed conic matrix
    """
    H_inv = np.linalg.inv(H)
    H_inv_T = H_inv.T
    C_prime = H_inv_T @ C @ H_inv
    return C_prime

def transform_dual_conic(C_star, H):
    """
    Transform a dual conic using C*' = H * C* * H^T
    
    Args:
        C_star: Dual conic matrix (3x3)
        H: Homography matrix (3x3)
    
    Returns:
        C_star_prime: Transformed dual conic matrix
    """
    C_star_prime = H @ C_star @ H.T
    return C_star_prime

def conic_to_ellipse_params(C):
    """
    Extract ellipse parameters from a conic matrix
    
    For a conic C, the equation is x^T * C * x = 0
    We extract center, axes, and rotation angle
    """
    # Normalize conic
    if abs(C[2, 2]) > 1e-10:
        C = C / C[2, 2]
    
    # Extract quadratic form parameters
    a = C[0, 0]
    b = C[0, 1] * 2  # Note: conic has 2b, so we multiply by 2
    c = C[1, 1]
    d = C[0, 2] * 2
    e = C[1, 2] * 2
    f = C[2, 2]
    
    # Calculate center
    det = a * c - (b/2)**2
    if abs(det) < 1e-10:
        return None  # Degenerate conic
    
    x0 = (b * e - 2 * c * d) / (4 * det)
    y0 = (b * d - 2 * a * e) / (4 * det)
    
    # Translate conic to origin
    C_translated = C.copy()
    C_translated[0, 2] = 0
    C_translated[2, 0] = 0
    C_translated[1, 2] = 0
    C_translated[2, 1] = 0
    C_translated[2, 2] = -(a * x0**2 + b * x0 * y0 + c * y0**2 + d * x0 + e * y0)
    
    # Eigenvalue decomposition to get axes and rotation
    Q = np.array([[C_translated[0, 0], C_translated[0, 1]],
                  [C_translated[0, 1], C_translated[1, 1]]])
    
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    
    if eigenvals[0] * eigenvals[1] > 0:
        # Both same sign - ellipse
        eigenvals = np.abs(eigenvals)
        if abs(C_translated[2, 2]) > 1e-10:
            eigenvals = eigenvals / abs(C_translated[2, 2])
            axes = 2 * np.sqrt(1.0 / eigenvals)
        else:
            return None
    else:
        return None  # Hyperbola or parabola
    
    # Rotation angle
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    return {'center': (x0, y0), 'width': axes[0], 'height': axes[1], 'angle': angle}

def draw_conic_points(C, x_range, y_range, num_points=1000):
    """
    Draw conic by sampling points that satisfy x^T * C * x = 0
    
    For visualization, we'll sample points and check if they're close to the conic
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate conic equation: [x, y, 1] * C * [x, y, 1]^T = 0
    points = np.stack([X.flatten(), Y.flatten(), np.ones(X.size)])
    values = np.diag(points.T @ C @ points).reshape(X.shape)
    
    # Find contour at value 0
    return X, Y, values

def create_circle_conic(center, radius):
    """
    Create a conic matrix for a circle
    
    For circle: (x - cx)^2 + (y - cy)^2 = r^2
    Expanding: x^2 + y^2 - 2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) = 0
    
    Conic matrix C:
    [1   0   -cx  ]
    [0   1   -cy  ]
    [-cx -cy  cx^2+cy^2-r^2]
    """
    cx, cy = center
    r = radius
    
    C = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [-cx, -cy, cx**2 + cy**2 - r**2]
    ], dtype=np.float32)
    
    return C

def main():
    # Create a projective transformation matrix
    H = create_homography_matrix()
    print("Homography matrix H:")
    print(H)
    print("\nH^(-T) (used in conic transformation):")
    print(np.linalg.inv(H).T)
    
    # Create original conics (circles)
    original_conics = [
        create_circle_conic((100, 100), 40),
        create_circle_conic((150, 100), 30),
        create_circle_conic((100, 150), 35),
        create_circle_conic((200, 150), 25),
    ]
    
    # Define visualization range
    x_range = (0, 300)
    y_range = (0, 250)
    
    # Transform conics
    transformed_conics = [transform_conic(C, H) for C in original_conics]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original conics
    ax1 = axes[0]
    colors = ['blue', 'green', 'orange', 'purple']
    
    for i, C in enumerate(original_conics):
        # Extract ellipse parameters
        params = conic_to_ellipse_params(C)
        if params:
            ellipse = Ellipse(params['center'], params['width'], params['height'],
                            angle=params['angle'], fill=False, 
                            edgecolor=colors[i % len(colors)], linewidth=2, alpha=0.8)
            ax1.add_patch(ellipse)
            ax1.text(params['center'][0], params['center'][1], f'C{i+1}',
                    fontsize=10, color=colors[i % len(colors)], 
                    ha='center', va='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original Conics (Circles)\nC: x^T * C * x = 0', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot transformed conics
    ax2 = axes[1]
    
    # Calculate transformed range by transforming corners
    corners = np.array([
        [x_range[0], y_range[0], 1],
        [x_range[1], y_range[0], 1],
        [x_range[1], y_range[1], 1],
        [x_range[0], y_range[1], 1]
    ])
    transformed_corners = (H @ corners.T).T
    transformed_corners_2d = transformed_corners[:, :2] / transformed_corners[:, 2:3]
    x_range_trans = (transformed_corners_2d[:, 0].min() - 30, transformed_corners_2d[:, 0].max() + 30)
    y_range_trans = (transformed_corners_2d[:, 1].min() - 30, transformed_corners_2d[:, 1].max() + 30)
    
    for i, C_prime in enumerate(transformed_conics):
        # Extract ellipse parameters
        params = conic_to_ellipse_params(C_prime)
        if params:
            # Check if ellipse is within reasonable bounds
            if (x_range_trans[0] <= params['center'][0] <= x_range_trans[1] and
                y_range_trans[0] <= params['center'][1] <= y_range_trans[1]):
                ellipse = Ellipse(params['center'], params['width'], params['height'],
                                angle=params['angle'], fill=False, 
                                edgecolor=colors[i % len(colors)], linewidth=2, 
                                linestyle='--', alpha=0.8)
                ax2.add_patch(ellipse)
                ax2.text(params['center'][0], params['center'][1], f"C'{i+1}",
                        fontsize=10, color=colors[i % len(colors)], 
                        ha='center', va='center', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_xlim(x_range_trans)
    ax2.set_ylim(y_range_trans)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Transformed Conics (Ellipses)\nC\' = H^(-T) * C * H^(-1)\n(Circles become ellipses under projective transformation)', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('x\'')
    ax2.set_ylabel('y\'')
    
    plt.tight_layout()
    
    # Add text box with transformation formula
    fig.text(0.5, 0.02, 
            'Conic Transformation: C\' = H^(-T) * C * H^(-1)  |  Dual Conic: C*\' = H * C* * H^T  |  Conics transform covariantly', 
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()
    
    # Print transformation details
    print("\n" + "="*60)
    print("Conic Transformation Details:")
    print("="*60)
    for i, (C, C_prime) in enumerate(zip(original_conics, transformed_conics)):
        print(f"\nConic {i+1}:")
        print("  Original C:")
        print(f"    {C}")
        print("  Transformed C':")
        print(f"    {C_prime}")
        print("  Verification: C' = H^(-T) * C * H^(-1)")
        H_inv = np.linalg.inv(H)
        computed = H_inv.T @ C @ H_inv
        print(f"  Match: {np.allclose(computed, C_prime, atol=1e-5)}")
        
        # Show that circles become ellipses
        orig_params = conic_to_ellipse_params(C)
        trans_params = conic_to_ellipse_params(C_prime)
        if orig_params and trans_params:
            print(f"  Original: Circle with radius ≈ {orig_params['width']/2:.2f}")
            print(f"  Transformed: Ellipse with axes ({trans_params['width']:.2f}, {trans_params['height']:.2f})")

if __name__ == "__main__":
    main()

