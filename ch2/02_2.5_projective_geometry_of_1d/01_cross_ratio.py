"""
1 randomly generate 4 coded marker in a white background image
but the keypoint of the 4 markers should be collinear

redo. randomly generate, do not let the user click. do not load any image. just start with a white image of  1920x1080


2 display on screen. when user maximize the window, aspect ratio should be maintained

(i will screenshot the screen and print it)

3 write a live video stream (video capture) to detect the 4 markers and calculate the cross ratio
4 display the cross ratio on screen


pip install opencv-contrib-python

"""

import cv2
import numpy as np
import random

def generate_marker_image():
    """Generate a white image with 4 ArUco markers with collinear keypoints"""
    # Create white background image 1920x1080
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    
    # Initialize ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Generate 4 different marker IDs
    marker_ids = [0, 1, 2, 3]
    
    # Use the diagonal line of the image (from top-left to bottom-right)
    # Image dimensions: 1920x1080
    margin = 20
    width, height = 1920-2*margin, 1080-2*margin
    line_start = (margin, margin)  # Top-left corner
    line_end = (width - margin, height - margin)  # Bottom-right corner
    
    # Marker size
    marker_size = 150
    
    # Calculate diagonal length
    diagonal_length = np.sqrt(width**2 + height**2)
    
    # Minimum spacing between marker centers (marker size + margin)
    min_spacing = marker_size + 20  # 20 pixel margin
    min_spacing_ratio = min_spacing / diagonal_length
    
    # Randomly generate 4 positions along the diagonal with minimum spacing
    # Keep markers away from edges (at least marker_size/2 from edges)
    edge_margin = (marker_size / 2) / diagonal_length
    min_t = edge_margin
    max_t = 1.0 - edge_margin
    
    # Generate t values with minimum spacing using iterative approach
    max_attempts = 1000
    t_values = []
    
    for attempt in range(max_attempts):
        t_values = []
        for i in range(4):
            if i == 0:
                # First marker: random position in available range
                t = random.uniform(min_t, max_t - 3 * min_spacing_ratio)
            else:
                # Subsequent markers: ensure minimum spacing from previous
                min_next_t = t_values[-1] + min_spacing_ratio
                if min_next_t > max_t:
                    break  # Not enough space, restart
                # Calculate max position considering remaining markers
                remaining_markers = 4 - i
                max_next_t = max_t - (remaining_markers - 1) * min_spacing_ratio
                if min_next_t > max_next_t:
                    break  # Not enough space, restart
                t = random.uniform(min_next_t, max_next_t)
            t_values.append(t)
        
        if len(t_values) == 4:
            break  # Successfully generated 4 positions with spacing
    
    if len(t_values) != 4:
        # Fallback: evenly space markers if random generation fails
        available_range = max_t - min_t - 3 * min_spacing_ratio
        t_values = [min_t + i * (available_range / 3 + min_spacing_ratio) for i in range(4)]
    
    marker_centers = []
    for t in t_values:
        x = int(line_start[0] + t * (line_end[0] - line_start[0]))
        y = int(line_start[1] + t * (line_end[1] - line_start[1]))
        marker_centers.append((x, y))
    
    # Generate and place markers
    for i, (marker_id, center) in enumerate(zip(marker_ids, marker_centers)):
        # Generate marker (compatible with both old and new OpenCV versions)
        try:
            # Try new API (OpenCV 4.7+)
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        except AttributeError:
            # Fall back to old API (OpenCV 4.0-4.6)
            marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
        
        # Calculate top-left corner position
        top_left_x = center[0] - marker_size // 2
        top_left_y = center[1] - marker_size // 2
        
        # Place marker on white background
        # Convert marker to 3-channel
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        
        # Place marker on image
        y1 = max(0, top_left_y)
        y2 = min(1080, top_left_y + marker_size)
        x1 = max(0, top_left_x)
        x2 = min(1920, top_left_x + marker_size)
        
        marker_y1 = max(0, -top_left_y)
        marker_y2 = marker_y1 + (y2 - y1)
        marker_x1 = max(0, -top_left_x)
        marker_x2 = marker_x1 + (x2 - x1)
        
        img[y1:y2, x1:x2] = marker_bgr[marker_y1:marker_y2, marker_x1:marker_x2]
    
    return img, marker_centers

def calculate_cross_ratio(points):
    """
    Calculate cross ratio for 4 collinear points A, B, C, D
    CR = (AC/BC) / (AD/BD) = (AC * BD) / (BC * AD)
    where AC, BC, AD, BD are signed distances along the line
    """
    if len(points) != 4:
        return None
    
    # Convert to numpy array
    points = np.array(points)
    
    # Calculate distances along the line
    # First, find the line direction vector
    # Use first and last point to define line direction
    line_vec = points[3] - points[0]
    line_vec = line_vec / np.linalg.norm(line_vec)
    
    # Project all points onto the line
    # Use first point as origin
    origin = points[0]
    projections = []
    for p in points:
        vec = p - origin
        proj = np.dot(vec, line_vec)
        projections.append(proj)
    
    # Sort projections to get order along line
    sorted_indices = np.argsort(projections)
    sorted_projections = [projections[i] for i in sorted_indices]
    
    # Calculate signed distances
    A, B, C, D = sorted_projections
    
    AC = C - A
    BC = C - B
    AD = D - A
    BD = D - B
    
    # Avoid division by zero
    if abs(BC) < 1e-6 or abs(AD) < 1e-6:
        return None
    
    cross_ratio = (AC * BD) / (BC * AD)
    return cross_ratio

def detect_markers_and_calculate_cr(frame):
    """Detect ArUco markers in frame and calculate cross ratio"""
    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Detect markers (compatible with both old and new OpenCV versions)
    try:
        # Try new API (OpenCV 4.7+)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(frame)
    except AttributeError:
        # Fall back to old API (OpenCV 4.0-4.6)
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    
    if ids is None or len(ids) < 4:
        return frame, None, None
    
    # Get center points of detected markers
    marker_centers = []
    marker_id_to_center = {}
    
    for i, marker_id in enumerate(ids.flatten()):
        corner = corners[i][0]
        # Calculate center point
        center = np.mean(corner, axis=0)
        marker_centers.append(center)
        marker_id_to_center[marker_id] = center
    
    # Check if we have exactly 4 markers
    if len(marker_centers) != 4:
        return frame, None, None
    
    # Draw markers and centers
    frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    # Draw center points
    for center in marker_centers:
        cv2.circle(frame, tuple(center.astype(int)), 5, (0, 255, 0), -1)
    
    # Calculate cross ratio
    cross_ratio = calculate_cross_ratio(marker_centers)
    
    return frame, cross_ratio, marker_centers

def display_image_with_aspect_ratio(img, window_name="Cross Ratio Markers"):
    """Display image maintaining aspect ratio when window is resized"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(window_name, img)
    print(f"Image displayed. Press any key to continue to video stream...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 1. Generate marker image
    print("Generating 4 ArUco markers with collinear keypoints...")
    marker_img, marker_centers = generate_marker_image()
    
    print(f"Markers generated at centers: {marker_centers}")
    
    # 2. Display image maintaining aspect ratio
    print("\nDisplaying marker image (maintains aspect ratio when maximized)...")
    display_image_with_aspect_ratio(marker_img)
    
    # 3. Live video stream to detect markers and calculate cross ratio
    print("\nStarting video stream...")
    print("Press 'q' to quit")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect markers and calculate cross ratio
        frame, cross_ratio, centers = detect_markers_and_calculate_cr(frame)
        
        # Display cross ratio on screen
        if cross_ratio is not None:
            text = f"Cross Ratio: {cross_ratio:.4f}"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw line connecting the centers
            if centers is not None and len(centers) == 4:
                # Draw line through all centers
                for i in range(len(centers) - 1):
                    pt1 = tuple(centers[i].astype(int))
                    pt2 = tuple(centers[i+1].astype(int))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        else:
            text = "Cross Ratio: Not enough markers detected (need 4)"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Cross Ratio Detection', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo stream ended.")

if __name__ == "__main__":
    main()
