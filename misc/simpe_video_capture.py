"""
Simple video capture: camera id 1, 1280x800, cv2.imshow, ESC to exit.
No PyQt5. Use to check if camera open/preview is slow with OpenCV only.
"""
import cv2

CAM_ID = 1
WIDTH = 1280
HEIGHT = 800

def main():
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("Failed to open camera", CAM_ID)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opened camera {CAM_ID}, resolution {got_w}x{got_h}")

    win = "Video (ESC to exit)"
    cv2.namedWindow(win)

    # while True:
    for i in range(10000000):
        print(f"{i}")
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(win, frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
