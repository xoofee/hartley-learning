"""
A7.3 Elation on P.631 of the book 

write a python script to 
1 load the building.jpg image (source)
2 pop out the window to let the user click 4 points (image points)
the point will be the corner of a window but projectively imaged near vertically
3 set the destionation(world) points near the original position but no perspective distortion
4 calculate the H matrix (image to world)
5 let the user dynamically drag the window center point to any position
6 calculate the translation vector in the world coordinate system
7 calculate the transform matrix P by
    P = H_world_to_image * T * inv(H_world_to_image)
    where the T is the translation matrix in the world coordinate system
8 transform the window using the P
9 imprint the window on the source
10 show the source image with the window imprinted on it
   the imprint should be dynamically updated as the window is dragged

you may use opencv or matplotlib or PyQt5 that you think is the best for the task

make the image zoomable (by mouse wheel) and draggable (by mouse right button)

do not remove this comment.

[for youtube]
Title:
Interactive Elation Transformation: Drag Windows in Projective Space | Computer Vision Tutorial

Introduction:
Welcome to this computer vision tutorial on elation transformations! In this video, I'll demonstrate an interactive PyQt5 application that lets you drag windows in projective space using homography matrices.

Here's what we'll cover:
- How to calculate homography matrices from point correspondences
- Understanding the elation transformation formula: P = H × T × inv(H)
- Implementing real-time window translation in world coordinates
- Creating an interactive GUI with zoom, pan, and drag functionality

The application allows you to:
1. Click 4 points to mark a window with perspective distortion
2. Automatically calculate the homography from image to world coordinates
3. Drag the window center point to translate it in world space
4. See the window transform and imprint on the source image in real-time

This is based on section A7.3 (Elation) from Hartley & Zisserman's "Multiple View Geometry" book, demonstrating how translations in world coordinates appear as projective transformations in image space.

Let's dive in and see how projective geometry makes this possible!


"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor


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
        self.display_image = None  # Image with window imprinted
        self.points = []  # List of DraggablePoint
        self.window_center = None  # DraggablePoint for window center
        self.dragged_point_idx = None
        self.selection_mode = True  # True: selection mode, False: dragging mode
        self.setMinimumSize(800, 600)
        self.update_cursor()
        
        # Zoom and pan state
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.panning = False
        self.last_pan_pos = None
    
    def set_image(self, image):
        """Set the original image"""
        self.image = image.copy()
        self.display_image = image.copy()
        self.update_cursor()
        self.update()
    
    def reset_zoom_pan(self):
        """Reset zoom and pan to default values"""
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()
    
    def update_cursor(self):
        """Update cursor based on selection mode"""
        if self.selection_mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
    
    def set_selection_mode(self, mode):
        """Set selection mode and update cursor"""
        self.selection_mode = mode
        self.update_cursor()
    
    def set_points(self, points, colors, labels):
        """Set the draggable points"""
        self.points = []
        for (x, y), color, label in zip(points, colors, labels):
            self.points.append(DraggablePoint(x, y, color, label))
        self.update()
    
    def set_window_center(self, x, y):
        """Set the window center point"""
        if self.window_center is None:
            self.window_center = DraggablePoint(x, y, (255, 255, 0), "Center", radius=10)
        else:
            self.window_center.set_pos(x, y)
        self.update()
    
    def set_display_image(self, image):
        """Set the display image with window imprinted"""
        self.display_image = image.copy()
        self.update()
    
    def widget_to_image_coords(self, widget_x, widget_y):
        """Convert widget coordinates to image coordinates, accounting for scaling, padding, zoom and pan"""
        if self.image is None:
            return widget_x, widget_y
        
        pixmap_size = self.size()
        img_h, img_w = self.image.shape[:2]
        
        # Calculate base scale to fit (same as in paintEvent)
        scale_x = pixmap_size.width() / img_w
        scale_y = pixmap_size.height() / img_h
        base_scale = min(scale_x, scale_y)
        
        # Apply zoom
        scale = base_scale * self.zoom_factor
        
        scaled_w = img_w * scale
        scaled_h = img_h * scale
        
        # Calculate offsets with pan
        x_offset = (pixmap_size.width() - scaled_w) / 2 + self.pan_x
        y_offset = (pixmap_size.height() - scaled_h) / 2 + self.pan_y
        
        # Convert widget coordinates to image coordinates
        img_x = (widget_x - x_offset) / scale
        img_y = (widget_y - y_offset) / scale
        
        return img_x, img_y
    
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            # Convert to image coordinates
            if self.image is not None:
                img_x, img_y = self.widget_to_image_coords(x, y)
                
                # Clamp to image bounds
                img_h, img_w = self.image.shape[:2]
                img_x = max(0, min(img_w - 1, img_x))
                img_y = max(0, min(img_h - 1, img_y))
                
                if self.selection_mode:
                    # In selection mode, emit click signal
                    self.point_clicked.emit(img_x, img_y)
                else:
                    # In dragging mode, only allow dragging window center
                    if self.window_center is not None and self.window_center.contains(img_x, img_y):
                        self.dragged_point_idx = -1  # Special index for window center
                        self.window_center.dragging = True
        elif event.button() == Qt.RightButton:
            # Start panning
            self.panning = True
            self.last_pan_pos = (event.x(), event.y())
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move (dragging)"""
        if self.panning and self.last_pan_pos is not None:
            # Panning with right mouse button
            dx = event.x() - self.last_pan_pos[0]
            dy = event.y() - self.last_pan_pos[1]
            self.pan_x += dx
            self.pan_y += dy
            self.last_pan_pos = (event.x(), event.y())
            self.update()
        elif self.dragged_point_idx is not None and self.image is not None:
            x = event.x()
            y = event.y()
            # Convert to image coordinates
            img_x, img_y = self.widget_to_image_coords(x, y)
            
            # Clamp to image bounds
            img_h, img_w = self.image.shape[:2]
            img_x = max(0, min(img_w - 1, img_x))
            img_y = max(0, min(img_h - 1, img_y))
            
            if self.dragged_point_idx == -1:
                # Dragging window center
                self.window_center.set_pos(img_x, img_y)
                self.point_moved.emit(-1, img_x, img_y)
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            if self.dragged_point_idx is not None:
                if self.dragged_point_idx == -1:
                    self.window_center.dragging = False
                self.dragged_point_idx = None
        elif event.button() == Qt.RightButton:
            # Stop panning
            self.panning = False
            self.last_pan_pos = None
            self.update_cursor()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.image is None:
            return
        
        # Get mouse position in widget coordinates
        mouse_x = event.x()
        mouse_y = event.y()
        
        # Get image coordinates before zoom (point under cursor)
        img_x, img_y = self.widget_to_image_coords(mouse_x, mouse_y)
        
        # Zoom factor change
        zoom_delta = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_zoom = self.zoom_factor * zoom_delta
        
        # Limit zoom range
        new_zoom = max(0.1, min(10.0, new_zoom))
        
        if new_zoom != self.zoom_factor:
            # Calculate new pan to keep the point under cursor fixed
            pixmap_size = self.size()
            img_h, img_w = self.image.shape[:2]
            scale_x = pixmap_size.width() / img_w
            scale_y = pixmap_size.height() / img_h
            base_scale = min(scale_x, scale_y)
            
            old_scale = base_scale * self.zoom_factor
            new_scale = base_scale * new_zoom
            
            # Calculate where the image point should be after zoom
            # Old position: mouse_x = (img_x * old_scale) + old_x_offset
            # New position: mouse_x = (img_x * new_scale) + new_x_offset
            # We want: old_x_offset + img_x * old_scale = new_x_offset + img_x * new_scale
            old_x_offset = (pixmap_size.width() - img_w * old_scale) / 2 + self.pan_x
            old_y_offset = (pixmap_size.height() - img_h * old_scale) / 2 + self.pan_y
            
            # Calculate what the new offset should be to keep the point under cursor
            new_x_offset = mouse_x - img_x * new_scale
            new_y_offset = mouse_y - img_y * new_scale
            
            # Calculate new pan
            base_x_offset = (pixmap_size.width() - img_w * new_scale) / 2
            base_y_offset = (pixmap_size.height() - img_h * new_scale) / 2
            self.pan_x = new_x_offset - base_x_offset
            self.pan_y = new_y_offset - base_y_offset
            
            self.zoom_factor = new_zoom
            self.update()
    
    def paintEvent(self, event):
        """Paint the image and points"""
        if self.display_image is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Scale to fit widget
        pixmap_size = self.size()
        img_h, img_w = self.display_image.shape[:2]
        scale_x = pixmap_size.width() / img_w
        scale_y = pixmap_size.height() / img_h
        base_scale = min(scale_x, scale_y)
        
        # Apply zoom
        scale = base_scale * self.zoom_factor
        
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        
        # Apply pan offset
        x_offset = (pixmap_size.width() - scaled_w) / 2 + self.pan_x
        y_offset = (pixmap_size.height() - scaled_h) / 2 + self.pan_y
        
        # Draw display image (already in RGB format)
        qimg = QImage(self.display_image.data, img_w, img_h, img_w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(int(x_offset), int(y_offset), pixmap)
        
        # Draw corner points as crosses
        for point in self.points:
            x = int(x_offset + point.x * scale)
            y = int(y_offset + point.y * scale)
            cross_size = int(point.radius * 2 * self.zoom_factor)  # Scale cross size with zoom
            pen = QPen(QColor(*point.color), max(1, int(2 * self.zoom_factor)))
            painter.setPen(pen)
            # Draw cross
            painter.drawLine(x - cross_size, y, x + cross_size, y)
            painter.drawLine(x, y - cross_size, x, y + cross_size)
            # Draw label
            painter.setPen(QColor(0, 0, 0))
            font_size = max(8, int(10 * self.zoom_factor))
            painter.setFont(QFont("Arial", font_size, QFont.Bold))
            painter.drawText(x + cross_size + 5, y - cross_size, point.label)
        
        # Draw window center point
        if self.window_center is not None:
            x = int(x_offset + self.window_center.x * scale)
            y = int(y_offset + self.window_center.y * scale)
            radius = int(self.window_center.radius * self.zoom_factor)  # Scale radius with zoom
            pen = QPen(QColor(*self.window_center.color), max(1, int(3 * self.zoom_factor)))
            painter.setPen(pen)
            painter.setBrush(QColor(*self.window_center.color))
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
            # Draw label
            painter.setPen(QColor(0, 0, 0))
            font_size = max(8, int(12 * self.zoom_factor))
            painter.setFont(QFont("Arial", font_size, QFont.Bold))
            painter.drawText(x + radius + 5, y - radius, self.window_center.label)
        
        self.scale_factor = scale
        self.x_offset = x_offset
        self.y_offset = y_offset


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.image_rgb = None
        self.src_points = []  # Image points (4 corners of window)
        self.dst_points = []  # World points (rectangular window)
        self.H = None  # Homography from image to world
        self.original_window_center_world = None  # Window center in world coordinates
        self.original_window_center_image = None  # Window center in image coordinates
        self.window_mask = None  # Mask for the window region
        self.window_region = None  # Window region extracted from image
        self.world_rect = None  # Rectangular window in world coordinates
        self.world_rect_corners = None  # Corners of world rectangle
        self.H_world_to_image = None  # Homography from world rect to image
        self.init_ui()
        self.load_image()
    
    def init_ui(self):
        self.setWindowTitle("Elation Window Translation")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Image display
        self.image_widget = ImageWidget()
        self.image_widget.point_moved.connect(self.on_point_moved)
        self.image_widget.point_clicked.connect(self.on_image_click)
        layout.addWidget(self.image_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        central_widget.setLayout(layout)
    
    def load_image(self):
        """Load the building.jpg image"""
        try:
            # Try to load from current directory
            # img_path = r'ch2\2.3_projective_transform\building.jpg'
            img_path = r'ch2\2.3_projective_transform\church.jpg'
            self.image = cv2.imread(img_path)

            if self.image is None:
                QMessageBox.warning(self, "Image Not Found", 
                                  "building.jpg not found. Please place it in the script directory or specify the path.")
                # Create a placeholder image
                self.image = np.ones((600, 800, 3), dtype=np.uint8) * 200
                cv2.putText(self.image, "building.jpg not found", (200, 300), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_widget.set_image(self.image_rgb)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")
    
    def reset(self):
        """Reset all points and state"""
        self.src_points = []
        self.dst_points = []
        self.H = None
        self.original_window_center_world = None
        self.original_window_center_image = None
        self.window_mask = None
        self.window_region = None
        self.world_rect = None
        self.world_rect_corners = None
        self.H_world_to_image = None
        self.image_widget.points = []
        self.image_widget.window_center = None
        self.image_widget.set_selection_mode(True)
        self.image_widget.reset_zoom_pan()
        self.image_widget.set_image(self.image_rgb)
        self.update()
    
    def on_image_click(self, img_x, img_y):
        """Handle image click for point selection"""
        if not self.image_widget.selection_mode:
            return
        
        # Clamp to image bounds
        img_h, img_w = self.image.shape[:2]
        img_x = max(0, min(img_w - 1, img_x))
        img_y = max(0, min(img_h - 1, img_y))
        
        if len(self.src_points) < 4:
            # Selecting image points (window corners with perspective)
            self.src_points.append([img_x, img_y])
            print(f"Image point {len(self.src_points)}: ({img_x:.1f}, {img_y:.1f})")
            
            # Update display
            all_points = self.src_points + self.dst_points
            colors = [(0, 255, 0)] * len(self.src_points) + [(255, 0, 0)] * len(self.dst_points)
            labels = [f"I{i+1}" for i in range(len(self.src_points))] + [f"W{i+1}" for i in range(len(self.dst_points))]
            self.image_widget.set_points(all_points, colors, labels)
            
            # After 4 image points, automatically set world points
            if len(self.src_points) == 4:
                self.set_world_points()
        elif len(self.dst_points) < 4:
            # This shouldn't happen as world points are set automatically
            pass
    
    def set_world_points(self):
        """Set world points (rectangular window) based on image points"""
        # Transform image points to approximate world coordinates
        # We'll create a rectangle that roughly matches the image points
        src_pts = np.array(self.src_points, dtype=np.float32)
        
        # Calculate bounding box
        min_x, min_y = np.min(src_pts, axis=0)
        max_x, max_y = np.max(src_pts, axis=0)
        
        # Create a rectangular window in world coordinates
        # Use the center and average dimensions
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = (max_x - min_x) * 0.8  # Slightly smaller
        height = (max_y - min_y) * 0.8
        
        # Create rectangle corners (clockwise from top-left)
        self.dst_points = [
            [center_x - width/2, center_y - height/2],  # Top-left
            [center_x + width/2, center_y - height/2],  # Top-right
            [center_x + width/2, center_y + height/2],  # Bottom-right
            [center_x - width/2, center_y + height/2]   # Bottom-left
        ]
        
        print("World points set (rectangular):")
        for i, pt in enumerate(self.dst_points):
            print(f"  W{i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
        
        # Update display
        all_points = self.src_points + self.dst_points
        colors = [(0, 255, 0)] * 4 + [(255, 0, 0)] * 4
        labels = [f"I{i+1}" for i in range(4)] + [f"W{i+1}" for i in range(4)]
        self.image_widget.set_points(all_points, colors, labels)
        
        # Automatically switch to drag mode and set up window
        self.image_widget.set_selection_mode(False)
        self.setup_window()
    
    def setup_window(self):
        """Calculate H and extract window region"""
        if len(self.src_points) != 4 or len(self.dst_points) != 4:
            return
        
        # Calculate homography from image to world
        src_pts = np.array(self.src_points, dtype=np.float32)
        dst_pts = np.array(self.dst_points, dtype=np.float32)
        self.H = find_homography_normalized(src_pts, dst_pts)
        
        print(f"\nHomography H (image to world):\n{self.H}")
        
        # Calculate inverse homography (world to image)
        H_inv = np.linalg.inv(self.H)
        
        # Calculate window center in world coordinates
        world_center = np.mean(dst_pts, axis=0)
        self.original_window_center_world = world_center
        
        # Transform center to image coordinates
        center_h = np.array([world_center[0], world_center[1], 1.0])
        center_img_h = H_inv @ center_h
        self.original_window_center_image = (center_img_h[:2] / center_img_h[2])
        
        print(f"Window center (world): ({world_center[0]:.1f}, {world_center[1]:.1f})")
        print(f"Window center (image): ({self.original_window_center_image[0]:.1f}, {self.original_window_center_image[1]:.1f})")
        
        # Extract window region from image
        self.extract_window_region()
        
        # Set window center point for dragging
        self.image_widget.set_window_center(self.original_window_center_image[0], self.original_window_center_image[1])
        
        # Update display
        self.update_display()
    
    def extract_window_region(self):
        """Extract the window region from the image"""
        if self.H is None:
            return
        
        h, w = self.image.shape[:2]
        
        # Use the 4 image points (clicked window corners) to create mask
        image_corners = np.array(self.src_points, dtype=np.float32)
        
        # Create mask for the window region
        self.window_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self.window_mask, [image_corners.astype(np.int32)], 255)
        
        # Extract window region from original image
        self.window_region = cv2.bitwise_and(self.image, self.image, mask=self.window_mask)
        
        # Also store the world rectangle for transformation
        # Get bounding box of world points
        dst_pts = np.array(self.dst_points, dtype=np.float32)
        min_x, min_y = np.min(dst_pts, axis=0)
        max_x, max_y = np.max(dst_pts, axis=0)
        
        # Create rectangle in world space
        world_w = int(max_x - min_x)
        world_h = int(max_y - min_y)
        self.world_rect = np.ones((world_h, world_w, 3), dtype=np.uint8) * 255
        
        # Store world rectangle corners
        self.world_rect_corners = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ], dtype=np.float32)
        
        # Calculate homography from world rectangle to image corners
        H_inv = np.linalg.inv(self.H)
        world_to_image_corners = []
        for pt in self.world_rect_corners:
            pt_h = np.array([pt[0], pt[1], 1.0])
            img_pt_h = H_inv @ pt_h
            img_pt = img_pt_h[:2] / img_pt_h[2]
            world_to_image_corners.append(img_pt)
        world_to_image_corners = np.array(world_to_image_corners, dtype=np.float32)
        
        # Homography from world rectangle to image
        self.H_world_to_image = cv2.getPerspectiveTransform(
            self.world_rect_corners, world_to_image_corners
        )
    
    def on_point_moved(self, point_idx, x, y):
        """Handle point drag"""
        if point_idx == -1:
            # Window center dragged
            self.original_window_center_image = np.array([x, y])
            self.update_transformation()
    
    def update_transformation(self):
        """Update transformation when window center is dragged"""
        if self.H is None or self.original_window_center_image is None:
            return
        
        # Calculate new window center in world coordinates
        center_img_h = np.array([self.original_window_center_image[0], self.original_window_center_image[1], 1.0])
        center_world_h = self.H @ center_img_h
        new_center_world = center_world_h[:2] / center_world_h[2]
        
        # Calculate translation in world coordinates
        translation_world = new_center_world - self.original_window_center_world
        
        # Create translation matrix in world coordinates
        T_world = np.eye(3)
        T_world[0, 2] = translation_world[0]
        T_world[1, 2] = translation_world[1]
        
        # elation
        # Calculate P = H_world_to_image * T_world * inv(H_world_to_image)
        H_world_to_image_inv = np.linalg.inv(self.H_world_to_image)
        P = self.H_world_to_image @ T_world @ H_world_to_image_inv
        P /= P[2, 2] if abs(P[2, 2]) > 1e-12 else 1.0
        
        print(f"\nTranslation (world): ({translation_world[0]:.2f}, {translation_world[1]:.2f})")
        print(f"Transform P:\n{P}")
        
        # Transform window region using P
        self.transform_and_imprint_window(P)
    
    def transform_and_imprint_window(self, P):
        """Transform window region using P and imprint on source image"""
        if self.window_region is None or self.window_mask is None:
            return
        
        h, w = self.image.shape[:2]
        
        # Transform the window region using warpPerspective
        # This handles interpolation and is much more efficient
        transformed_window = cv2.warpPerspective(
            self.window_region, P, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # Transform the mask as well
        transformed_mask = cv2.warpPerspective(
            self.window_mask, P, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Imprint window on source image
        display_img = self.image_rgb.copy()
        
        # Use mask to blend (only where mask is non-zero)
        mask_3d = (transformed_mask[:, :, np.newaxis] > 0).astype(np.float32)
        window_rgb = cv2.cvtColor(transformed_window, cv2.COLOR_BGR2RGB)
        
        # Blend: show transformed window where mask is active
        display_img = (1 - mask_3d) * display_img + mask_3d * window_rgb
        display_img = display_img.astype(np.uint8)
        
        self.image_widget.set_display_image(display_img)
    
    def update_display(self):
        """Update the display image"""
        if self.window_region is None:
            self.image_widget.set_display_image(self.image_rgb)
        else:
            # Initially, just show the window region imprinted
            display_img = self.image_rgb.copy()
            mask_3d = self.window_mask[:, :, np.newaxis] / 255.0
            window_rgb = cv2.cvtColor(self.window_region, cv2.COLOR_BGR2RGB)
            display_img = (1 - mask_3d * 0.5) * display_img + mask_3d * 0.5 * window_rgb
            display_img = display_img.astype(np.uint8)
            self.image_widget.set_display_image(display_img)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
