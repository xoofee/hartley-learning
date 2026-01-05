"""
Illustration of line transformation under projective transformation.

According to section 2.3.1:
- Under point transformation x' = Hx, a line transforms as l' = H^(-T) * l
- Points transform contravariantly (according to H)
- Lines transform covariantly (according to H^(-1))

This script demonstrates:
1. Create some lines in the original space
2. Apply a projective transformation H
3. Show how lines transform using l' = H^(-T) * l
4. Visualize both original and transformed lines
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def create_homography_matrix():
    """Create a projective transformation matrix H"""
    # Create a homography that includes perspective distortion
    H = np.array([
        [1.2, 0.3, 50],
        [0.1, 1.1, 30],
        [0.001, 0.002, 1.0]
    ], dtype=np.float32)
    return H

def transform_line(l, H):
    """
    Transform a line using l' = H^(-T) * l
    
    Args:
        l: Line in homogeneous coordinates [a, b, c] where ax + by + c = 0
        H: Homography matrix (3x3)
    
    Returns:
        l': Transformed line in homogeneous coordinates
    """
    H_inv_T = np.linalg.inv(H).T
    l_prime = H_inv_T @ l
    return l_prime

def line_to_points(l, x_range, y_range):
    """
    Convert a line equation to points for visualization
    
    Args:
        l: Line in homogeneous coordinates [a, b, c]
        x_range: Tuple (x_min, x_max)
        y_range: Tuple (y_min, y_max)
    
    Returns:
        x, y: Arrays of points on the line
    """
    a, b, c = l[0], l[1], l[2]
    
    # Handle vertical lines (b ≈ 0)
    if abs(b) < 1e-6:
        x = np.full(100, -c / a)
        y = np.linspace(y_range[0], y_range[1], 100)
    else:
        x = np.linspace(x_range[0], x_range[1], 100)
        y = -(a * x + c) / b
    
    return x, y

def transform_point(x, H):
    """Transform a point using x' = Hx"""
    x_homogeneous = np.array([x[0], x[1], 1.0])
    x_prime_homogeneous = H @ x_homogeneous
    x_prime = x_prime_homogeneous[:2] / x_prime_homogeneous[2]
    return x_prime

def main():
    # Create a projective transformation matrix
    H = create_homography_matrix()
    print("Homography matrix H:")
    print(H)
    print("\nH^(-T) (used for line transformation):")
    print(np.linalg.inv(H).T)
    
    # Define original lines in homogeneous coordinates [a, b, c] where ax + by + c = 0
    # Create a grid of parallel and perpendicular lines
    original_lines = [
        np.array([1, 0, -50]),   # Vertical line: x = 50
        np.array([1, 0, -100]),  # Vertical line: x = 100
        np.array([1, 0, -150]),  # Vertical line: x = 150
        np.array([0, 1, -50]),   # Horizontal line: y = 50
        np.array([0, 1, -100]),  # Horizontal line: y = 100
        np.array([0, 1, -150]),  # Horizontal line: y = 150
        np.array([1, -1, 0]),    # Diagonal line: x - y = 0
        np.array([1, 1, -200]),  # Diagonal line: x + y = 200
    ]
    
    # Define visualization range
    x_range = (0, 250)
    y_range = (0, 250)
    
    # Transform lines
    transformed_lines = [transform_line(l, H) for l in original_lines]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original lines
    ax1 = axes[0]
    for i, l in enumerate(original_lines):
        x, y = line_to_points(l, x_range, y_range)
        # Filter points within range
        mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
        ax1.plot(x[mask], y[mask], 'b-', linewidth=2, alpha=0.7)
        # Add label
        mid_idx = len(x[mask]) // 2
        if mid_idx > 0:
            ax1.text(x[mask][mid_idx], y[mask][mid_idx], f'L{i+1}', 
                    fontsize=9, color='blue', bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', alpha=0.7))
    
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Original Lines\n(Parallel lines remain parallel in Euclidean space)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot transformed lines
    ax2 = axes[1]
    # Calculate transformed range by transforming corners
    corners = np.array([
        [x_range[0], y_range[0]],
        [x_range[1], y_range[0]],
        [x_range[1], y_range[1]],
        [x_range[0], y_range[1]]
    ])
    transformed_corners = np.array([transform_point(c, H) for c in corners])
    x_range_trans = (transformed_corners[:, 0].min() - 20, transformed_corners[:, 0].max() + 20)
    y_range_trans = (transformed_corners[:, 1].min() - 20, transformed_corners[:, 1].max() + 20)
    
    for i, l_prime in enumerate(transformed_lines):
        x, y = line_to_points(l_prime, x_range_trans, y_range_trans)
        # Filter points within reasonable range
        mask = (x >= x_range_trans[0]) & (x <= x_range_trans[1]) & \
               (y >= y_range_trans[0]) & (y <= y_range_trans[1])
        if np.any(mask):
            ax2.plot(x[mask], y[mask], 'r-', linewidth=2, alpha=0.7)
            # Add label
            valid_indices = np.where(mask)[0]
            if len(valid_indices) > 0:
                mid_idx = len(valid_indices) // 2
                ax2.text(x[valid_indices[mid_idx]], y[valid_indices[mid_idx]], f"L'{i+1}", 
                        fontsize=9, color='red', bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.7))
    
    ax2.set_xlim(x_range_trans)
    ax2.set_ylim(y_range_trans)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Transformed Lines\n(l\' = H^(-T) * l)\n(Parallel lines converge under projective transformation)', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('x\'')
    ax2.set_ylabel('y\'')
    
    plt.tight_layout()
    
    # Add text box with transformation formula
    fig.text(0.5, 0.02, 
            'Line Transformation: l\' = H^(-T) * l  |  Points transform as: x\' = Hx  |  Lines transform covariantly', 
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()
    
    # Print transformation details
    print("\n" + "="*60)
    print("Line Transformation Details:")
    print("="*60)
    for i, (l, l_prime) in enumerate(zip(original_lines, transformed_lines)):
        print(f"\nLine {i+1}:")
        print(f"  Original:   {l[0]:.2f}x + {l[1]:.2f}y + {l[2]:.2f} = 0")
        print(f"  Transformed: {l_prime[0]:.3f}x' + {l_prime[1]:.3f}y' + {l_prime[2]:.3f} = 0")
        print(f"  Verification: l'^T = l^T * H^(-1)")
        l_T_H_inv = (l.T @ np.linalg.inv(H)).T
        print(f"  Computed:    [{l_T_H_inv[0]:.3f}, {l_T_H_inv[1]:.3f}, {l_T_H_inv[2]:.3f}]")
        print(f"  Direct:      [{l_prime[0]:.3f}, {l_prime[1]:.3f}, {l_prime[2]:.3f}]")
        print(f"  Match: {np.allclose(l_T_H_inv, l_prime)}")

if __name__ == "__main__":
    main()

