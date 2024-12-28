#this script will take image and rotate it acording to the reading from the mpu6050
import sys
import os

# Add the path to the directory where image_handler.py is located
sys.path.append(os.path.abspath("/home/admin/GIT/naggles/usb_camera"))
# Print sys.path to check the module search paths
#print(sys.path)
from usb_camera.image_handler import capture_and_rotate_images

def start_capture():
    print("Starting image capture...")
    capture_and_rotate_images(image_count=15, interval=1)
    print("Image capture finished.")

if __name__ == "__main__":
    start_capture()
