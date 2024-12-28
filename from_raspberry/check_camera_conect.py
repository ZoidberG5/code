import cv2

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            print(f"Camera index {index} is available.")
            cap.release()  # Release the camera
        else:
            cap.release()  # Ensure the camera is released even if not opened
    if not available_cameras:
        print("No cameras are available.")
    return available_cameras

if __name__ == "__main__":
    available_cameras = list_available_cameras()
    print(f"Available cameras: {available_cameras}")
