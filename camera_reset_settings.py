import cv2

def reset_camera_settings(camera_index):
    """
    Resets camera settings to default values for the camera at the given index.
    """
    print(f"Initializing camera {camera_index}...")
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        print(f"Failed to open camera {camera_index}.")
        return False

    print(f"Resetting settings for camera {camera_index}...")
    # Reset basic camera properties (default values might vary per camera)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Default brightness
    camera.set(cv2.CAP_PROP_CONTRAST, 0.5)    # Default contrast
    camera.set(cv2.CAP_PROP_SATURATION, 0.5)  # Default saturation
    camera.set(cv2.CAP_PROP_GAIN, 1.0)        # Default gain
    camera.set(cv2.CAP_PROP_EXPOSURE, -1)     # Default auto-exposure

    # Capture a test frame to verify the camera is working
    ret, frame = camera.read()
    if not ret:
        print(f"Failed to capture frame from camera {camera_index}.")
        camera.release()
        return False

    # Display the captured frame
    cv2.imshow(f"Camera {camera_index} Frame", frame)
    cv2.waitKey(1000)  # Display the frame for 1 second
    cv2.destroyWindow(f"Camera {camera_index} Frame")

    # Release the camera
    camera.release()
    print(f"Settings reset for camera {camera_index}.")
    return True

def reset_all_cameras():
    """
    Resets settings for two cameras connected to the computer.
    """
    for index in range(2):  # Assuming two cameras are connected
        success = reset_camera_settings(index)
        if not success:
            print(f"Skipping further operations for camera {index} due to an error.")

if __name__ == "__main__":
    reset_all_cameras()
    print("All cameras processed.")
    cv2.destroyAllWindows()
