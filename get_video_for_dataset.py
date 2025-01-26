import cv2
import datetime
import os
import time
from OOP_webcam import CameraSystem

def init_cameras(camera_indices, width=640, height=480):
    """
    Initialize cameras and check which are connected.
    Returns a dictionary with connected cameras and their indices.
    """
    cameras = {
        "right": cv2.VideoCapture(camera_indices[1]),
        "left": cv2.VideoCapture(camera_indices[0])
    }
    connected_cameras = {}
    for name, cap in cameras.items():
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            connected_cameras[name] = cap
        else:
            print(f"Warning: {name.capitalize()} camera is not connected.")
    return connected_cameras

def release_cameras(connected_cameras):
    """
    Release all connected cameras and close windows.
    """
    for name, cap in connected_cameras.items():
        cap.release()
    cv2.destroyAllWindows()

def record_video(connected_cameras, save_path, width=640, height=480, FPS=15, duration=10):
    """
    Record video from connected cameras for a fixed duration.
    """
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Create VideoWriter objects for each connected camera
    writers = {}
    for name, cap in connected_cameras.items():
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"{save_path}/{current_time}_{name.upper()}_video.mp4"  # Save as MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        writers[name] = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
        print(f"Recording {name} camera to: {video_path}")

    print(f"Recording started for {duration} seconds.")
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= duration:
            print("Recording finished.")
            break

        # Capture and write frames from all connected cameras
        for name, cap in connected_cameras.items():
            ret, frame = cap.read()
            if ret:
                writers[name].write(frame)
                cv2.imshow(f"{name.capitalize()} Camera", frame)
            else:
                print(f"Error: Could not read frame from {name} camera.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manual exit detected (not required for duration-based recording).")
            break

    # Release VideoWriter objects
    for writer in writers.values():
        writer.release()

if __name__ == "__main__":
    # Initialize cameras
    camera_indices = (2, 0)  # [left index, right index]
    image_resolution = (1920, 1080)
    #connected_cameras = init_cameras(camera_indices, width=1920, height=1080)

    usb_camera = CameraSystem(cameras_distance=0.235, max_cameras=10)
    cap_left, cap_right = usb_camera.init_two_cameras(camera_indices, width=image_resolution[0], height=image_resolution[1])

    # Set exposure value
    cap_left.set(cv2.CAP_PROP_ZOOM, 0)
    cap_right.set(cv2.CAP_PROP_ZOOM, 0)


    # Print exposure value
    print(f"The exposure value is:{cap_left.get(cv2.CAP_PROP_EXPOSURE)}")

    # Store connected cameras in a dictionary
    connected_cameras = {}
    connected_cameras['left'] = cap_left
    connected_cameras['right'] = cap_right

    # Check if any camera is connected
    if not connected_cameras:
        print("Error: No cameras are connected.")
    else:
        # Record video for 10 seconds
        record_video(connected_cameras, "code/videos", width=1920, height=1080, FPS=5, duration=20)

    # Release cameras
    #release_cameras(connected_cameras)
    usb_camera.release_two_cameras(cap_left, cap_right)
