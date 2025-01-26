import cv2
import time
import datetime
from OOP_webcam import CameraSystem


def init_two_cameras(width=640, height=480):
    # Open the USB cameras (index depending on your setup)
    cap_right = cv2.VideoCapture(1)
    cap_left = cv2.VideoCapture(0)
    
    # Set the resolution (width and height)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap_right, cap_left

def release_two_cameras(cap_right, cap_left):
    # Release the camera
    cap_right.release()
    cap_left.release()
    cv2.destroyAllWindows()
    return

def capture_images(n, save_path, width=640, height=480, FPS = 15):
    # Open the USB camera (index 0 or 1 depending on your setup)
    cap = cv2.VideoCapture(0)
    
    # Set the resolution (width and height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    for i in range(n):
        # Capture a frame
        ret, frame = cap.read()

        if ret:
            # Save each image with a unique filename
            image_path = f"{save_path}/captured_image_{i+1}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image {i+1} saved at {image_path}")
        else:
            print(f"Error: Could not read frame {i+1} from camera.")
        time.sleep(1/FPS)
        
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

def capture_images_from_two_cameras(cap_right, cap_left, n, save_path, width=640, height=480, FPS = 15):
    # # Open the USB cameras (index depending on your setup)
    # cap_right = cv2.VideoCapture(1)
    # cap_left = cv2.VideoCapture(0)
    
    # # Set the resolution (width and height)
    # cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap_right.isOpened():
        print("Error: Could not open RIGHT camera.")
        return
    if not cap_left.isOpened():
        print("Error: Could not open LEFT camera.")
        return

    for i in range(n):
        # Enable auto-exposure (property ID 21 is for exposure, and setting it to -1 or 1 often means 'auto')
        exposure_supported_left = cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # The value depends on the camera; 0.75 often means 'auto mode'        
        exposure_supported_right = cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # The value depends on the camera; 0.75 often means 'auto mode'
        if exposure_supported_left and exposure_supported_right:
            print("Auto-exposure enabled (which may adjust shutter speed).")
        else:
            print("Auto-exposure not supported by this camera.")

        # exposure_value = cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        # exposure_value = cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        # # Enable auto-focus (property ID 28 is for focus, but the value for auto-focus varies by camera)
        # # Setting it to 1 typically enables auto-focus; 0 disables it
        # focus_supported = cap_left.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # Get current time for the filenames
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        # Capture a frame
        ret_right, frame_right = cap_right.read()
        ret_left, frame_left = cap_left.read()

        if ret_right:
            # Save right image with a unique filename
            image_path = f"{save_path}/{current_time}_RIGHT_image_{i+1}.jpg"
            cv2.imwrite(image_path, frame_right)
            print(f"Image {i+1} from right camera saved at {image_path}")
        else:
            print(f"Error: Could not read frame {i+1} from right camera.")
        
        if ret_left:
            # Save left image with a unique filename
            image_path = f"{save_path}/{current_time}_LEFT_image_{i+1}.jpg"
            cv2.imwrite(image_path, frame_left)
            print(f"Image {i+1} from left camera saved at {image_path}")
        else:
            print(f"Error: Could not read frame {i+1} from left camera.")
        time.sleep(1/FPS)

    # # Release the camera
    # cap_right.release()
    # cap_left.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_indices = (2, 0)  # [left index, right index]
    image_resolution = (1920, 1080) # [width, height]
    # Usage example: capture n images
    #capture_images(5, "code\images", width=1920, height=1080, FPS = 15)
    usb_camera = CameraSystem(cameras_distance=0.235, max_cameras=10)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cap_left, cap_right = usb_camera.init_two_cameras(camera_indices, width=image_resolution[0], height=image_resolution[1])
    #usb_camera.capture_images_from_cameras(cap_right, cap_left, camera_indices, "code\images", width=1920, height=1080)
    capture_images_from_two_cameras(cap_right, cap_left, 20, "code\images", width=image_resolution[0], height=image_resolution[1], FPS = 5)
    release_two_cameras(cap_right, cap_left)