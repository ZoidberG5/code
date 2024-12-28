#from mpu6050_sensor.mpu6050_OOP import MPU6050Sensor
from yolov5_obj_detection.yolov5_detector import YOLOv5Detector
import time
import cv2
import os

def check_exposure(exposure_value = -1, output_dir = "/home/admin/GIT/naggles/static/images"):
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the ID of the camera, use 1 or 2 if you have multiple cameras

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        # Set the exposure time
        # Note: The exposure value might need to be adjusted based on your specific camera and environment.
        # Positive values: Exposure time in milliseconds (sometimes), 
        # Negative values: Camera will auto-expose (this depends on the camera/driver)
        
        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)  # Example value, adjust as needed
        
        # Capture a single frame to test
        ret, frame = cap.read()
        if ret:
            output_filename = os.path.join(output_dir, f"exposure_image,{exposure_value}.jpg")
            cv2.imwrite(output_filename, frame)

        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    exposure_value = -4
    output_dir="/home/admin/GIT/naggles/static/images"
    check_exposure(exposure_value, output_dir)
