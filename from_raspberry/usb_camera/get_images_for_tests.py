import cv2
import datetime
import time
import os

def get_external_camera_indexes():
    start_time = time.perf_counter()
    
    # Detect all connected cameras and ignore the built-in one (usually index 0)
    index = 0
    external_cameras = []
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            # Assume the built-in camera has a typical resolution of 640x480 (common for built-ins)
            is_built_in = cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 640 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480

            if not is_built_in or index != 0:  # Ignore index 0 if it seems like a built-in camera
                external_cameras.append(index)

            cap.release()
        index += 1

    end_time = time.perf_counter()
    print(f"get_external_camera_indexes() runtime: {end_time - start_time:.4f} seconds")
    
    return external_cameras

import cv2
import time

def capture_images(cameras, width=640, height=480):
    start_time = time.perf_counter()
    
    # Check if two cameras are detected
    if len(cameras) < 2:
        print("Error: Less than 2 external cameras detected.")
        return None, None

    # Open the two cameras
    cap_left = cv2.VideoCapture(cameras[0])
    cap_right = cv2.VideoCapture(cameras[1])

    # Set resolution for both cameras
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read frames from both cameras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    # Release the camera resources
    cap_left.release()
    cap_right.release()

    end_time = time.perf_counter()
    print(f"capture_images() runtime: {end_time - start_time:.4f} seconds")

    if ret_left and ret_right:
        print("Images captured successfully.")
        return frame_left, frame_right
    else:
        print("Error: Unable to read frames from cameras.")
        return None, None


def save_images(frame_left, frame_right):
    start_time = time.perf_counter()
    
    output_dir="/home/admin/GIT/naggles/static/testing_images"
    #output_filename = os.path.join(output_dir, "detected_image0.jpg")

    # Ensure the directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get current time for the filenames
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the file names for saving the images
    left_image_filename = os.path.join(output_dir, f"{current_time}_left.jpg")
    right_image_filename = os.path.join(output_dir, f"{current_time}_right.jpg")

    # Save the images if frames are valid
    if frame_left is not None and frame_right is not None:
        cv2.imwrite(left_image_filename, frame_left)
        cv2.imwrite(right_image_filename, frame_right)
        print(f"Images saved: {left_image_filename}, {right_image_filename}")
    else:
        print("Error: Invalid frames, images not saved.")

    end_time = time.perf_counter()
    print(f"save_images() runtime: {end_time - start_time:.4f} seconds")

# Measure the total runtime of the script
start_time_total = time.perf_counter()

# Detect external cameras
#external_camera_indexes = get_external_camera_indexes()
external_camera_indexes = [0,2]

# Capture images from the detected cameras
frame_left, frame_right = capture_images(external_camera_indexes, width=640, height=480)

# Save the captured images
save_images(frame_left, frame_right)

# End the total runtime measurement
end_time_total = time.perf_counter()
print(f"Total runtime: {end_time_total - start_time_total:.4f} seconds")
