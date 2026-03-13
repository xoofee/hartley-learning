"""
a single script to run a gui, with PyQt5 if necessary

let user select 8 points to calculate the homography matrix H, just like 01_projective_transform_by_4x2points.py
but:
  - do not close and then redraw.
  - let user dynamically move the eight points (by dragging) after he create it.
  - dynamically (as the user drag any of the eight points)
    - recalculate the homography matrix H and the invariant points.
    - show the H in gui
    - show the original and transformed image in the same axis. The transformed image is tranparent over the original image.
    - show the invariant points in the same axis.
    - decompose H to HsHaHp where Hs is the similarity transform, Ha is the affine transform, Hp is the perspective transform. the scale factor s in Hs is positive
        Hs=[sR t;
            0 1]
    - dynamically show the Hs Ha Hp in the gui

  - interaction
    - let the user edit the H. when H is changed, the transformed image and the target 4 points and the invariant points and Hs Ha Hp should be updated.
    - let user edit the Hs Ha Hp. when Hs Ha Hp is changed, the transformed image and the target 4 points and the invariant points and H should be updated.

    do not remove this comment.

"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QPushButton, QMessageBox,
                             QScrollArea, QGroupBox, QGridLayout, QLineEdit, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont


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
    scale = np.sqrt(2) / mean_dist if mean_dist > 1e-10 else 1.0
    
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
    if src_points.shape[0] < 4:
        return np.eye(3)
    
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
    
    # 5. Normalize scale
    H /= H[2, 2] if abs(H[2, 2]) > 1e-12 else np.linalg.norm(H)
    
    return H


def find_invariant_points(H):
    """
    Find invariant points (fixed points) of homography H using eigenvector analysis.
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eig(H)
        invariant_points = []
        valid_eigenvalues = []
        
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            eigenvec = eigenvec.real if np.allclose(eigenvec.imag, 0) else eigenvec
            if abs(eigenvec[2]) > 1e-10:
                x = eigenvec[0] / eigenvec[2]
                y = eigenvec[1] / eigenvec[2]
                invariant_points.append((x, y))
                valid_eigenvalues.append(eigenval)
        
        return invariant_points, valid_eigenvalues
    except:
        return [], []



def decompose_homography_my2(H):
    """
    Decompose H into H = Hs * Ha * Hp
    where:
    - Hs: similarity transform [sR t; 0 1] with s > 0
    - Ha: affine transform [K 0; 0 1] where K is upper triangular, det(K) = 1
    - Hp: perspective transform [I 0; v^T w]
    
    Mathematical formulation:
    H = Hs * Ha * Hp
      = [sR t; 0 1] * [K 0; 0 1] * [I 0; v^T w]
      = [sRK + tv^T  tu; v^T w]
    
    Returns: (Hs, Ha, Hp)
    """
    try:
        # Normalize H to avoid numerical issues
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        
        # Extract perspective part: Hp = [I 0; v^T w]
        v = H[2, :2].copy()
        w = H[2, 2]
        
        # Construct Hp with proper normalization
        Hp = np.eye(3)
        Hp[2, :2] = v
        Hp[2, 2] = w
        
        # Remove perspective: H_affine = H * Hp^-1 = Hs * Ha
        Hp_inv = np.linalg.inv(Hp)
        H_affine = H @ Hp_inv
        
        # Normalize H_affine
        if abs(H_affine[2, 2]) > 1e-12:
            H_affine = H_affine / H_affine[2, 2]
        
        # Extract translation and affine part
        t = H_affine[:2, 2]
        A = H_affine[:2, :2]
        
        # Decompose A = sRK where we want to separate into (sR) and K
        # First, extract scale from determinant
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-12:
            # Degenerate case: use identity
            s = 1.0
            R = np.eye(2)
            K = np.eye(2)
        else:
            # Ensure positive determinant for scale extraction
            s = np.sqrt(abs(det_A))
            if s < 1e-10:
                s = 1.0
            
            # Normalize A to have unit determinant: RK = A / s
            RK = A / s
            
            # QR decomposition: RK = R * K where R is orthogonal, K is upper triangular
            R, K = np.linalg.qr(RK)
            
            # Ensure positive diagonal for K (upper triangular should have positive diagonal)
            # for i in range(2):
            #     if K[i, i] < 0:
            #         K[i, :] *= -1
            #         R[:, i] *= -1
            
            # Ensure R is a proper rotation (det = 1), not a reflection
            # if np.linalg.det(R) < 0:
            #     R[:, -1] *= -1
            #     K[-1, :] *= -1
            
            # Normalize K to have det(K) = 1
            # det_K = np.linalg.det(K)
            # if abs(det_K) > 1e-12:
            #     scale_K = np.sqrt(abs(det_K))
            #     K = K / scale_K
            #     # Adjust scale: s_new = s * scale_K
            #     s = s * scale_K
        
        # Construct Ha: [K 0; 0 1] with no translation
        Ha = np.eye(3)
        Ha[:2, :2] = K
        Ha[:2, 2] = 0.0  # Explicitly set translation to zero
        
        # Construct Hs: [sR t; 0 1]
        Hs = np.eye(3)
        Hs[:2, :2] = s * R
        Hs[:2, 2] = t
        
        return Hs, Ha, Hp
    
    except Exception as e:
        print(f"Error in decompose_homography_my2: {e}")
        import traceback
        traceback.print_exc()
        # Return identity matrices on error
        return np.eye(3), np.eye(3), np.eye(3)

def decompose_homography_my(H):

    """
    Hs = [sR t;
         0 1]
    Ha = [K 0;
         0 1]
    Hp = [I 0;
          v^T u]
    H = Hs * Ha * Hp
      = [sR t;
         0 1] * 
         [K 0;
         v^T  u]
       = [sRK + tv^T  tu;
           v^T  u]   

    """

    # H = H/H[2,2] if abs(H[2,2]) > 1e-12 else H
    # already normalized before calling this function

    v = H[2,0:2].T
    t = H[:2, 2]
    sRK = H[:2, :2] - t@v.T
    s = np.sqrt(np.linalg.det(sRK))
    RK = sRK/s
    R, K = np.linalg.qr(RK)

    # Signs of the diagonal of R
    sign = np.sign(np.diag(K))
    sign[sign == 0] = 1   # avoid zero

    D = np.diag(sign)

    K = K @ D
    R = D @ R

    Hs = np.eye(3)
    Hs[:2, :2] = s*R
    Hs[:2, 2] = t

    Ha = np.eye(3)
    Ha[:2, :2] = K

    Hp = np.eye(3)
    Hp[2, :2] = v
    
    return Hs, Ha, Hp


def decompose_homography(H):

    # print(f"=========== decompose_homography ===========")
    # print(f"Decomposing homography: {H}")
    # print(f"Decomposing homography my: {decompose_homography_my(H)}")
    # print(f"Decomposing homography cursor: {decompose_homography_cursor(H)}")

    # return decompose_homography_my(H)
    # return decompose_homography_cursor(H)
    return decompose_homography_my2(H)

def decompose_homography_cursor(H):
    """
    Decompose H into H = Hs * Ha * Hp
    where:
    - Hs: similarity transform [sR t; 0 1] with s > 0
    - Ha: affine transform [K 0; 0 1] where K is upper triangular, det(K) = 1, no translation
    - Hp: perspective transform [I 0; v^T w]
    
    Returns: (Hs, Ha, Hp)
    """
    try:
        # Store original H for verification
        H_original = H.copy()
        
        # Normalize H
        H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
        
        # Extract perspective part: Hp = [I 0; v^T w]
        v = H[2, :2].copy()
        w = H[2, 2]
        
        # Construct Hp
        Hp = np.eye(3)
        Hp[2, :2] = v
        Hp[2, 2] = w
        
        # Remove perspective: H' = H * Hp^-1 = Hs * Ha
        Hp_inv = np.linalg.inv(Hp)
        H_affine = H @ Hp_inv
        
        # Normalize H_affine
        H_affine = H_affine / H_affine[2, 2] if abs(H_affine[2, 2]) > 1e-12 else H_affine
        
        # Extract affine part: Ha = [K 0; 0 1] where K is upper triangular, det(K) = 1
        # We want: H_affine[:2, :2] = (s * R_rot) * K where K is upper triangular, det(K) = 1
        A = H_affine[:2, :2]
        Q, R = np.linalg.qr(A)
        
        # Ensure positive diagonal for R (affine part)
        for i in range(2):
            if R[i, i] < 0:
                R[i, :] *= -1
                Q[:, i] *= -1
        
        # Ensure Q is a proper rotation (det = 1), not a reflection
        if np.linalg.det(Q) < 0:
            Q[:, -1] *= -1
            R[-1, :] *= -1
        
        # Normalize K so that det(K) = 1
        # We have A = Q * R, and we want A = (s * R_rot) * K where det(K) = 1
        # So: A = Q * R = (Q * sqrt(det(R))) * (R / sqrt(det(R)))
        # So: A = (Q * sqrt(det(R))) * K where K = R / sqrt(det(R)) has det(K) = 1
        
        det_R = np.linalg.det(R)
        if abs(det_R) > 1e-12:
            # Scale factor: we want det(K) = 1, so K = R / sqrt(det(R))
            scale_R = np.sqrt(abs(det_R))
            # Normalize R: K = R / scale_R (so det(K) = 1)
            K = R / scale_R
            # Now A = Q * R = Q * scale_R * K = (Q * scale_R) * K
            # Q is a rotation, so Q_scaled = Q * scale_R is a scaled rotation
            Q_scaled = Q * scale_R
        else:
            K = R
            Q_scaled = Q
        
        # Construct Ha with no translation part
        Ha = np.eye(3)
        Ha[:2, :2] = K
        Ha[:2, 2] = 0.0  # No translation part
        
        # Now we have: H_affine[:2, :2] = Q_scaled * K
        # We need to decompose Q_scaled into s * R_rot (similarity part)
        # Use SVD to extract scale and rotation from Q_scaled
        U, S, Vt = np.linalg.svd(Q_scaled)
        
        # Scale is the geometric mean of singular values
        s = np.sqrt(S[0] * S[1])
        if s < 1e-10:
            s = 1.0
        
        # Rotation matrix
        R_rot = U @ Vt
        if np.linalg.det(R_rot) < 0:
            U[:, -1] *= -1
            R_rot = U @ Vt
        
        # Construct Hs with proper rotation, scale, and translation
        Hs = np.eye(3)
        Hs[:2, :2] = s * R_rot
        Hs[:2, 2] = H_affine[:2, 2]  # Translation from H_affine goes to Hs
        
        # Verify decomposition: H_reconstructed = Hs * Ha * Hp should equal H_original
        H_reconstructed = Hs @ Ha @ Hp
        # Normalize both matrices for comparison
        H_reconstructed = H_reconstructed / H_reconstructed[2, 2] if abs(H_reconstructed[2, 2]) > 1e-12 else H_reconstructed
        H_original_normalized = H_original / H_original[2, 2] if abs(H_original[2, 2]) > 1e-12 else H_original
        
        # Compute difference
        diff = np.abs(H_reconstructed - H_original_normalized)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Print warning only if difference is significant
        if max_diff > 1e-6:
            print(f"Warning: Decomposition verification failed!")
            print(f"  Max difference: {max_diff:.2e}")
            print(f"  Mean difference: {mean_diff:.2e}")
            print(f"  H original:\n{H_original_normalized}")
            print(f"  H reconstructed:\n{H_reconstructed}")
            print(f"  Difference:\n{diff}")
        
        return Hs, Ha, Hp
    except Exception as e:
        print(f"Decomposition error: {e}")
        import traceback
        traceback.print_exc()
        return np.eye(3), np.eye(3), np.eye(3)


def compose_homography(Hs, Ha, Hp):
    """
    Compose H from Hs, Ha, Hp: H = Hs * Ha * Hp
    """
    try:
        H = Hs @ Ha @ Hp
        H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
        return H
    except:
        return np.eye(3)


def calculate_decomposition_error(H, Hs, Ha, Hp):
    """
    Calculate the maximum absolute error between H and Hs*Ha*Hp.
    
    Args:
        H: Original homography matrix (3x3)
        Hs, Ha, Hp: Decomposed matrices (3x3 each)
    
    Returns:
        max_error: Maximum absolute error
    """
    try:
        # Compose H_reconstructed = Hs * Ha * Hp
        H_reconstructed = compose_homography(Hs, Ha, Hp)
        
        # Normalize both matrices for comparison
        H_normalized = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
        H_reconstructed_normalized = H_reconstructed / H_reconstructed[2, 2] if abs(H_reconstructed[2, 2]) > 1e-12 else H_reconstructed
        
        # Calculate absolute difference
        diff = np.abs(H_normalized - H_reconstructed_normalized)
        max_error = np.max(diff)
        
        return max_error
    except:
        return float('inf')


def generate_grid_image(width=800, height=600, grid_spacing=50):
    """Generate an image with vertical and horizontal lines (grid)"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    for x in range(0, width, grid_spacing):
        cv2.line(img, (x, 0), (x, height), (0, 0, 0), 2)
    for y in range(0, height, grid_spacing):
        cv2.line(img, (0, y), (width, y), (0, 0, 0), 2)
    return img


class DraggablePoint:
    """A draggable point on the image"""
    def __init__(self, x, y, color, label, radius=8):
        self.x = x
        self.y = y
        self.color = color
        self.label = label
        self.radius = radius
        self.dragging = False
    
    def contains(self, x, y):
        """Check if point contains the given coordinates"""
        dx = x - self.x
        dy = y - self.y
        return dx*dx + dy*dy <= self.radius * self.radius
    
    def set_pos(self, x, y):
        """Set position"""
        self.x = x
        self.y = y


class ImageWidget(QWidget):
    """Widget for displaying image with draggable points"""
    point_moved = pyqtSignal(int, float, float)  # point_index, x, y
    point_clicked = pyqtSignal(float, float)  # x, y for new point selection
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.points = []  # List of DraggablePoint
        self.invariant_points = []
        self.transformed_image = None
        self.alpha = 0.5  # Transparency for transformed image
        self.dragged_point_idx = None
        self.selection_mode = False  # True: selection mode, False: dragging mode
        self.setMinimumSize(800, 600)
    
    def set_image(self, image):
        """Set the original image"""
        self.image = image.copy()
        self.update()
    
    def set_transformed_image(self, transformed_image):
        """Set the transformed image"""
        self.transformed_image = transformed_image
        self.update()
    
    def set_points(self, points, colors, labels):
        """Set the draggable points"""
        self.points = []
        for (x, y), color, label in zip(points, colors, labels):
            self.points.append(DraggablePoint(x, y, color, label))
        self.update()
    
    def set_invariant_points(self, points):
        """Set invariant points to display"""
        self.invariant_points = points
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            # Convert to image coordinates (matching paintEvent coordinate system)
            if self.image is not None:
                pixmap_size = self.size()
                img_h, img_w = self.image.shape[:2]
                scale_x = pixmap_size.width() / img_w
                scale_y = pixmap_size.height() / img_h
                scale = min(scale_x, scale_y)
                
                scaled_w = int(img_w * scale)
                scaled_h = int(img_h * scale)
                x_offset = (pixmap_size.width() - scaled_w) // 2
                y_offset = (pixmap_size.height() - scaled_h) // 2
                
                # Convert widget coordinates to image coordinates
                img_x = (x - x_offset) / scale
                img_y = (y - y_offset) / scale
                
                if self.selection_mode:
                    # In selection mode, emit click signal
                    self.point_clicked.emit(img_x, img_y)
                else:
                    # In dragging mode, check if clicking on a point
                    for i, point in enumerate(self.points):
                        if point.contains(img_x, img_y):
                            self.dragged_point_idx = i
                            point.dragging = True
                            break
    
    def mouseMoveEvent(self, event):
        """Handle mouse move (dragging)"""
        if self.dragged_point_idx is not None and self.image is not None:
            x = event.x()
            y = event.y()
            # Convert to image coordinates (matching paintEvent coordinate system)
            pixmap_size = self.size()
            img_h, img_w = self.image.shape[:2]
            scale_x = pixmap_size.width() / img_w
            scale_y = pixmap_size.height() / img_h
            scale = min(scale_x, scale_y)
            
            scaled_w = int(img_w * scale)
            scaled_h = int(img_h * scale)
            x_offset = (pixmap_size.width() - scaled_w) // 2
            y_offset = (pixmap_size.height() - scaled_h) // 2
            
            # Convert widget coordinates to image coordinates
            img_x = (x - x_offset) / scale
            img_y = (y - y_offset) / scale
            
            # Clamp to image bounds
            img_x = max(0, min(img_w - 1, img_x))
            img_y = max(0, min(img_h - 1, img_y))
            
            point = self.points[self.dragged_point_idx]
            point.set_pos(img_x, img_y)
            self.point_moved.emit(self.dragged_point_idx, img_x, img_y)
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            if self.dragged_point_idx is not None:
                self.points[self.dragged_point_idx].dragging = False
                self.dragged_point_idx = None
    
    def paintEvent(self, event):
        """Paint the image and points"""
        if self.image is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Scale to fit widget
        pixmap_size = self.size()
        img_h, img_w = self.image.shape[:2]
        scale_x = pixmap_size.width() / img_w
        scale_y = pixmap_size.height() / img_h
        scale = min(scale_x, scale_y)
        
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        x_offset = (pixmap_size.width() - scaled_w) // 2
        y_offset = (pixmap_size.height() - scaled_h) // 2
        
        # Draw original image
        qimg = QImage(self.image.data, img_w, img_h, img_w * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg).scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(x_offset, y_offset, pixmap)
        
        # Draw transformed image with transparency
        if self.transformed_image is not None:
            qimg_trans = QImage(self.transformed_image.data, img_w, img_h, img_w * 3, QImage.Format_RGB888).rgbSwapped()
            pixmap_trans = QPixmap.fromImage(qimg_trans).scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.setOpacity(self.alpha)
            painter.drawPixmap(x_offset, y_offset, pixmap_trans)
            painter.setOpacity(1.0)
        
        # Draw points
        for point in self.points:
            x = int(x_offset + point.x * scale)
            y = int(y_offset + point.y * scale)
            pen = QPen(QColor(*point.color), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(*point.color))
            painter.drawEllipse(x - point.radius, y - point.radius, 
                              point.radius * 2, point.radius * 2)
            # Draw label
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(x + point.radius + 5, y - point.radius, point.label)
        
        # Draw invariant points
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for i, (x, y) in enumerate(self.invariant_points):
            if 0 <= x < img_w and 0 <= y < img_h:
                px = int(x_offset + x * scale)
                py = int(y_offset + y * scale)
                painter.drawEllipse(px - 10, py - 10, 20, 20)
                painter.setPen(QColor(255, 0, 0))
                painter.setFont(QFont("Arial", 9))
                painter.drawText(px + 12, py - 12, f"F{i+1}")
                pen = QPen(QColor(255, 0, 0), 2)
                painter.setPen(pen)
        
        self.scale_factor = scale
        self.x_offset = x_offset
        self.y_offset = y_offset


class MatrixEditWidget(QWidget):
    """Widget for editing a 3x3 matrix"""
    matrix_changed = pyqtSignal(np.ndarray)
    
    def __init__(self, label):
        super().__init__()
        self.label = label
        self.matrix = np.eye(3)
        self.text_edits = []
        self.updating = False
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        title = QLabel(self.label)
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        grid = QGridLayout()
        self.text_edits = []
        for i in range(3):
            row_edits = []
            for j in range(3):
                edit = QLineEdit()
                edit.setMaximumWidth(80)
                edit.textChanged.connect(self.on_text_changed)
                grid.addWidget(edit, i, j)
                row_edits.append(edit)
            self.text_edits.append(row_edits)
        
        layout.addLayout(grid)
        self.setLayout(layout)
    
    def set_matrix(self, matrix):
        """Set the matrix and update text fields"""
        self.updating = True
        self.matrix = matrix.copy()
        for i in range(3):
            for j in range(3):
                self.text_edits[i][j].setText(f"{matrix[i, j]:.6f}")
        self.updating = False
    
    def on_text_changed(self):
        """Handle text change"""
        if self.updating:
            return
        
        try:
            new_matrix = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    new_matrix[i, j] = float(self.text_edits[i][j].text())
            self.matrix = new_matrix
            self.matrix_changed.emit(new_matrix)
        except ValueError:
            pass  # Invalid input, ignore


class ScaleRotationEditWidget(QWidget):
    """Widget for editing scale s and rotation angle theta (in degrees)"""
    scale_rotation_changed = pyqtSignal(float, float)  # s, theta_degrees
    
    def __init__(self):
        super().__init__()
        self.s = 1.0
        self.theta_degrees = 0.0
        self.text_edits = []  # For s (QLineEdit)
        self.theta_spin = None  # For theta (QDoubleSpinBox)
        self.updating = False
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        title = QLabel("Similarity Parameters (s, θ)")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        grid = QGridLayout()
        
        # Scale s
        label_s = QLabel("Scale s:")
        edit_s = QLineEdit()
        edit_s.setMaximumWidth(100)
        edit_s.textChanged.connect(self.on_text_changed)
        grid.addWidget(label_s, 0, 0)
        grid.addWidget(edit_s, 0, 1)
        self.text_edits.append(edit_s)
        
        # Rotation angle theta (degrees) - use spin box
        label_theta = QLabel("Angle θ (deg):")
        spin_theta = QDoubleSpinBox()
        spin_theta.setMaximumWidth(100)
        spin_theta.setRange(-360.0, 360.0)  # Allow full rotation range
        spin_theta.setSingleStep(1.0)  # Increment/decrement by 1 degree
        spin_theta.setDecimals(1)  # Show 1 decimal place
        spin_theta.valueChanged.connect(self.on_theta_changed)
        grid.addWidget(label_theta, 1, 0)
        grid.addWidget(spin_theta, 1, 1)
        self.theta_spin = spin_theta
        
        layout.addLayout(grid)
        self.setLayout(layout)
    
    def set_scale_rotation(self, s, theta_degrees):
        """Set the scale and rotation angle (called from external source)"""
        self.updating = True
        self.s = s
        self.theta_degrees = theta_degrees
        
        # Update s (QLineEdit) - only if value changed
        current_s_text = self.text_edits[0].text()
        try:
            current_s = float(current_s_text) if current_s_text else None
            if current_s is None or abs(current_s - s) > 1e-6:
                # Value changed or invalid, update with formatting
                self.text_edits[0].setText(f"{s:.6f}")
        except ValueError:
            # Invalid text, update with formatting
            self.text_edits[0].setText(f"{s:.6f}")
        
        # Update theta (QDoubleSpinBox) - only if value changed
        if abs(self.theta_spin.value() - theta_degrees) > 1e-6:
            self.theta_spin.setValue(theta_degrees)
        
        self.updating = False
    
    def on_text_changed(self):
        """Handle text change for s (real-time updates as user types)"""
        if self.updating:
            return
        
        try:
            new_s = float(self.text_edits[0].text())
            # Only emit if value actually changed (to avoid infinite loops)
            if abs(new_s - self.s) > 1e-6:
                self.s = new_s
                theta = self.theta_spin.value()
                self.scale_rotation_changed.emit(new_s, theta)
        except ValueError:
            # Invalid input while typing - don't do anything, let user continue typing
            pass
    
    def on_theta_changed(self, value):
        """Handle theta spin box value change"""
        if self.updating:
            return
        
        # Only emit if value actually changed
        if abs(value - self.theta_degrees) > 1e-6:
            self.theta_degrees = value
            try:
                s = float(self.text_edits[0].text())
                self.scale_rotation_changed.emit(s, value)
            except ValueError:
                # If s is invalid, still emit with current s value
                self.scale_rotation_changed.emit(self.s, value)


def extract_scale_rotation_from_Hs(Hs):
    """
    Extract scale s and rotation angle theta (in degrees) from Hs.
    Hs[:2, :2] = s * R where R is a rotation matrix.
    
    Returns: (s, theta_degrees)
    """
    try:
        A = Hs[:2, :2]
        # Use SVD to extract scale and rotation
        U, S, Vt = np.linalg.svd(A)
        
        # Scale is the geometric mean of singular values
        s = np.sqrt(S[0] * S[1])
        if s < 1e-10:
            s = 1.0
        
        # Rotation matrix
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        
        # Extract angle from rotation matrix
        # R = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        theta_radians = np.arctan2(R[1, 0], R[0, 0])
        theta_degrees = np.degrees(theta_radians)
        
        return s, theta_degrees
    except:
        return 1.0, 0.0


def construct_Hs_from_scale_rotation(s, theta_degrees, translation):
    """
    Construct Hs from scale s, rotation angle theta (in degrees), and translation.
    Hs = [s*R t; 0 1] where R is a rotation matrix.
    
    Args:
        s: scale factor
        theta_degrees: rotation angle in degrees
        translation: (tx, ty) translation vector
    
    Returns:
        Hs: (3, 3) similarity transformation matrix
    """
    theta_radians = np.radians(theta_degrees)
    cos_theta = np.cos(theta_radians)
    sin_theta = np.sin(theta_radians)
    
    # Rotation matrix
    R = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Construct Hs
    Hs = np.eye(3)
    Hs[:2, :2] = s * R
    Hs[:2, 2] = translation
    
    return Hs


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.src_points = []
        self.dst_points = []
        self.H = np.eye(3)
        self.Hs = np.eye(3)
        self.Ha = np.eye(3)
        self.Hp = np.eye(3)
        self.invariant_points = []
        self.point_selection_mode = True  # True: selecting points, False: dragging mode
        
        # Store initial state for reset
        self.initial_src_points = []
        self.initial_dst_points = []
        self.initial_point_selection_mode = True
        
        self.init_ui()
        self.load_image()
        self.image_widget.selection_mode = True
    
    def init_ui(self):
        self.setWindowTitle("Homography Research GUI")
        self.setGeometry(100, 100, 1600, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left side: Image display
        left_layout = QVBoxLayout()
        self.image_widget = ImageWidget()
        self.image_widget.point_moved.connect(self.on_point_moved)
        self.image_widget.point_clicked.connect(self.on_image_click)
        left_layout.addWidget(self.image_widget)
        
        # Reset button
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Initial State")
        self.reset_button.clicked.connect(self.reset_to_initial_state)
        button_layout.addWidget(self.reset_button)
        left_layout.addLayout(button_layout)
        
        main_layout.addLayout(left_layout, 2)
        
        # Right side: Controls and matrices
        right_layout = QVBoxLayout()
        
        # Decomposition error display
        error_group = QGroupBox("Decomposition Error (H - Hs * Ha * Hp)")
        error_layout = QVBoxLayout()
        self.error_label = QLabel("Max Error: N/A")
        error_layout.addWidget(self.error_label)
        error_group.setLayout(error_layout)
        right_layout.addWidget(error_group)
        
        # Matrix editors
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        self.H_editor = MatrixEditWidget("Homography H")
        self.H_editor.matrix_changed.connect(self.on_H_changed)
        scroll_layout.addWidget(self.H_editor)
        
        # Scale and rotation editor for Hs
        self.scale_rotation_editor = ScaleRotationEditWidget()
        self.scale_rotation_editor.scale_rotation_changed.connect(self.on_scale_rotation_changed)
        scroll_layout.addWidget(self.scale_rotation_editor)
        
        self.Hs_editor = MatrixEditWidget("Similarity Hs")
        self.Hs_editor.matrix_changed.connect(self.on_Hs_changed)
        scroll_layout.addWidget(self.Hs_editor)
        
        self.Ha_editor = MatrixEditWidget("Affine Ha")
        self.Ha_editor.matrix_changed.connect(self.on_Ha_changed)
        scroll_layout.addWidget(self.Ha_editor)
        
        self.Hp_editor = MatrixEditWidget("Perspective Hp")
        self.Hp_editor.matrix_changed.connect(self.on_Hp_changed)
        scroll_layout.addWidget(self.Hp_editor)
        
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)
        
        main_layout.addLayout(right_layout, 1)
        
        central_widget.setLayout(main_layout)
    
    def load_image(self):
        """Load or generate the image"""
        self.image = generate_grid_image(width=800, height=600, grid_spacing=50)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_widget.set_image(self.image_rgb)
    
    def on_image_click(self, img_x, img_y):
        """Handle image click for point selection"""
        if not self.point_selection_mode:
            return
        
        # Clamp to image bounds
        img_h, img_w = self.image.shape[:2]
        img_x = max(0, min(img_w - 1, img_x))
        img_y = max(0, min(img_h - 1, img_y))
        
        if len(self.src_points) < 4:
            self.src_points.append([img_x, img_y])
            print(f"Source point {len(self.src_points)}: ({img_x:.1f}, {img_y:.1f})")
            # Update display
            all_points = self.src_points + self.dst_points
            colors = [(0, 255, 0)] * len(self.src_points) + [(255, 0, 0)] * len(self.dst_points)
            labels = [f"S{i+1}" for i in range(len(self.src_points))] + [f"D{i+1}" for i in range(len(self.dst_points))]
            self.image_widget.set_points(all_points, colors, labels)
        elif len(self.dst_points) < 4:
            self.dst_points.append([img_x, img_y])
            print(f"Destination point {len(self.dst_points)}: ({img_x:.1f}, {img_y:.1f})")
            # Update display
            all_points = self.src_points + self.dst_points
            colors = [(0, 255, 0)] * 4 + [(255, 0, 0)] * len(self.dst_points)
            labels = [f"S{i+1}" for i in range(4)] + [f"D{i+1}" for i in range(len(self.dst_points))]
            self.image_widget.set_points(all_points, colors, labels)
        
        if len(self.src_points) == 4 and len(self.dst_points) == 4:
            # Auto-switch to drag mode after 8 points are selected
            self.point_selection_mode = False
            self.image_widget.selection_mode = False
            self.update_all()
    
    def on_point_moved(self, point_idx, x, y):
        """Handle point drag"""
        if point_idx < 4:
            self.src_points[point_idx] = [x, y]
        else:
            self.dst_points[point_idx - 4] = [x, y]
        self.update_all()
    
    def update_all(self):
        """Update everything based on current points"""
        if len(self.src_points) == 4 and len(self.dst_points) == 4:
            # Update H from points
            src_pts = np.array(self.src_points, dtype=np.float32)
            dst_pts = np.array(self.dst_points, dtype=np.float32)
            self.H = find_homography_normalized(src_pts, dst_pts)
            
            # Update decomposition
            self.Hs, self.Ha, self.Hp = decompose_homography(self.H)
            
            # Calculate and display decomposition error
            max_error = calculate_decomposition_error(self.H, self.Hs, self.Ha, self.Hp)
            self.error_label.setText(f"Max Error: {max_error:.2e}")
            
            # Update invariant points
            self.invariant_points, _ = find_invariant_points(self.H)
            
            # Update transformed image
            h, w = self.image.shape[:2]
            self.transformed_image = cv2.warpPerspective(self.image, self.H, (w, h))
            self.transformed_image_rgb = cv2.cvtColor(self.transformed_image, cv2.COLOR_BGR2RGB)
            self.image_widget.set_transformed_image(self.transformed_image_rgb)
            
            # Update points display
            all_points = self.src_points + self.dst_points
            colors = [(0, 255, 0)] * 4 + [(255, 0, 0)] * 4  # Green for source, red for dest
            labels = [f"S{i+1}" for i in range(4)] + [f"D{i+1}" for i in range(4)]
            self.image_widget.set_points(all_points, colors, labels)
            
            # Update invariant points
            self.image_widget.set_invariant_points(self.invariant_points)
            
            # Update matrix editors (without triggering callbacks)
            self.H_editor.updating = True
            self.H_editor.set_matrix(self.H)
            self.H_editor.updating = False
            
            # Update Hs editor and scale/rotation widget
            s, theta_degrees = extract_scale_rotation_from_Hs(self.Hs)
            self.scale_rotation_editor.updating = True
            self.scale_rotation_editor.set_scale_rotation(s, theta_degrees)
            self.scale_rotation_editor.updating = False
            
            self.Hs_editor.updating = True
            self.Hs_editor.set_matrix(self.Hs)
            self.Hs_editor.updating = False
            
            self.Ha_editor.updating = True
            self.Ha_editor.set_matrix(self.Ha)
            self.Ha_editor.updating = False
            
            self.Hp_editor.updating = True
            self.Hp_editor.set_matrix(self.Hp)
            self.Hp_editor.updating = False
    
    def on_H_changed(self, H):
        """Handle H matrix edit"""
        if self.H_editor.updating:
            return
        self.H = H
        # Update decomposition
        self.Hs, self.Ha, self.Hp = decompose_homography(self.H)
        
        # Calculate and display decomposition error
        max_error = calculate_decomposition_error(self.H, self.Hs, self.Ha, self.Hp)
        self.error_label.setText(f"Max Error: {max_error:.2e}")
        
        # Update invariant points
        self.invariant_points, _ = find_invariant_points(self.H)
        # Update transformed image
        h, w = self.image.shape[:2]
        self.transformed_image = cv2.warpPerspective(self.image, self.H, (w, h))
        self.transformed_image_rgb = cv2.cvtColor(self.transformed_image, cv2.COLOR_BGR2RGB)
        self.image_widget.set_transformed_image(self.transformed_image_rgb)
        self.image_widget.set_invariant_points(self.invariant_points)
        # Update destination points (transform source points)
        if len(self.src_points) == 4:
            src_pts_h = np.hstack([np.array(self.src_points), np.ones((4, 1))])
            dst_pts_h = (self.H @ src_pts_h.T).T
            self.dst_points = (dst_pts_h[:, :2] / dst_pts_h[:, 2:3]).tolist()
            all_points = self.src_points + self.dst_points
            colors = [(0, 255, 0)] * 4 + [(255, 0, 0)] * 4
            labels = [f"S{i+1}" for i in range(4)] + [f"D{i+1}" for i in range(4)]
            self.image_widget.set_points(all_points, colors, labels)
        # Update other matrix editors
        s, theta_degrees = extract_scale_rotation_from_Hs(self.Hs)
        self.scale_rotation_editor.updating = True
        self.scale_rotation_editor.set_scale_rotation(s, theta_degrees)
        self.scale_rotation_editor.updating = False
        
        self.Hs_editor.updating = True
        self.Hs_editor.set_matrix(self.Hs)
        self.Hs_editor.updating = False
        self.Ha_editor.updating = True
        self.Ha_editor.set_matrix(self.Ha)
        self.Ha_editor.updating = False
        self.Hp_editor.updating = True
        self.Hp_editor.set_matrix(self.Hp)
        self.Hp_editor.updating = False
    
    def on_scale_rotation_changed(self, s, theta_degrees):
        """Handle scale and rotation change"""
        if self.scale_rotation_editor.updating:
            return
        # Get current translation from Hs
        translation = self.Hs[:2, 2].copy()
        # Construct new Hs from s, theta, and translation
        self.Hs = construct_Hs_from_scale_rotation(s, theta_degrees, translation)
        # Update Hs editor without triggering callback
        self.Hs_editor.updating = True
        self.Hs_editor.set_matrix(self.Hs)
        self.Hs_editor.updating = False
        # Update everything from decomposition
        self.update_from_decomposition()
    
    def on_Hs_changed(self, Hs):
        """Handle Hs matrix edit"""
        if self.Hs_editor.updating:
            return
        self.Hs = Hs
        # Update scale/rotation widget
        s, theta_degrees = extract_scale_rotation_from_Hs(self.Hs)
        self.scale_rotation_editor.updating = True
        self.scale_rotation_editor.set_scale_rotation(s, theta_degrees)
        self.scale_rotation_editor.updating = False
        # Update everything from decomposition
        self.update_from_decomposition()
    
    def on_Ha_changed(self, Ha):
        """Handle Ha matrix edit"""
        if self.Ha_editor.updating:
            return
        self.Ha = Ha
        self.update_from_decomposition()
    
    def on_Hp_changed(self, Hp):
        """Handle Hp matrix edit"""
        if self.Hp_editor.updating:
            return
        self.Hp = Hp
        self.update_from_decomposition()
    
    def update_from_decomposition(self):
        """Update H and everything from Hs, Ha, Hp"""
        self.H = compose_homography(self.Hs, self.Ha, self.Hp)
        
        # Calculate and display decomposition error
        max_error = calculate_decomposition_error(self.H, self.Hs, self.Ha, self.Hp)
        self.error_label.setText(f"Max Error: {max_error:.2e}")
        
        # Update invariant points
        self.invariant_points, _ = find_invariant_points(self.H)
        # Update transformed image
        h, w = self.image.shape[:2]
        self.transformed_image = cv2.warpPerspective(self.image, self.H, (w, h))
        self.transformed_image_rgb = cv2.cvtColor(self.transformed_image, cv2.COLOR_BGR2RGB)
        self.image_widget.set_transformed_image(self.transformed_image_rgb)
        self.image_widget.set_invariant_points(self.invariant_points)
        # Update destination points
        if len(self.src_points) == 4:
            src_pts_h = np.hstack([np.array(self.src_points), np.ones((4, 1))])
            dst_pts_h = (self.H @ src_pts_h.T).T
            self.dst_points = (dst_pts_h[:, :2] / dst_pts_h[:, 2:3]).tolist()
            all_points = self.src_points + self.dst_points
            colors = [(0, 255, 0)] * 4 + [(255, 0, 0)] * 4
            labels = [f"S{i+1}" for i in range(4)] + [f"D{i+1}" for i in range(4)]
            self.image_widget.set_points(all_points, colors, labels)
        # Update H editor
        self.H_editor.updating = True
        self.H_editor.set_matrix(self.H)
        self.H_editor.updating = False
        
        # Update scale/rotation widget (Hs may have changed)
        s, theta_degrees = extract_scale_rotation_from_Hs(self.Hs)
        self.scale_rotation_editor.updating = True
        self.scale_rotation_editor.set_scale_rotation(s, theta_degrees)
        self.scale_rotation_editor.updating = False
    
    def reset_to_initial_state(self):
        """Reset the widget to its initial state"""
        # Reset points
        self.src_points = []
        self.dst_points = []
        
        # Reset point selection mode
        self.point_selection_mode = self.initial_point_selection_mode
        self.image_widget.selection_mode = self.initial_point_selection_mode
        
        # Reset matrices to identity
        self.H = np.eye(3)
        self.Hs = np.eye(3)
        self.Ha = np.eye(3)
        self.Hp = np.eye(3)
        self.invariant_points = []
        
        # Clear points display
        self.image_widget.set_points([], [], [])
        self.image_widget.set_invariant_points([])
        
        # Clear transformed image
        self.image_widget.set_transformed_image(None)
        
        # Reset error display
        self.error_label.setText("Max Error: N/A")
        
        # Update matrix editors
        self.H_editor.updating = True
        self.H_editor.set_matrix(self.H)
        self.H_editor.updating = False
        
        self.Hs_editor.updating = True
        self.Hs_editor.set_matrix(self.Hs)
        self.Hs_editor.updating = False
        
        self.Ha_editor.updating = True
        self.Ha_editor.set_matrix(self.Ha)
        self.Ha_editor.updating = False
        
        self.Hp_editor.updating = True
        self.Hp_editor.set_matrix(self.Hp)
        self.Hp_editor.updating = False
        
        # Update scale/rotation widget
        s, theta_degrees = extract_scale_rotation_from_Hs(self.Hs)
        self.scale_rotation_editor.updating = True
        self.scale_rotation_editor.set_scale_rotation(s, theta_degrees)
        self.scale_rotation_editor.updating = False


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
