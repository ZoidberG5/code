# usb_webcam_OOP
import cv2
import numpy as np

class CameraSystem:
    def __init__(self, cameras_distance=0.28, max_cameras=10):
        """
        Initializes the CameraSystem with the specified cameras distance and max cameras to scan.
        
        :param cameras_distance: The distance between the cameras in meters.
        :param max_cameras: The maximum number of cameras to scan for availability.
        """
        self.cameras_distance = cameras_distance  # Distance between the cameras
        self.max_cameras = max_cameras  # Maximum number of cameras to scan
        #self.available_cameras = self.list_available_cameras()  # List of available camera indices

    def list_available_cameras(self):
        """
        Lists all available cameras by checking each index up to max_cameras.
        
        :return: A list of indices of available cameras.
        """
        available_cameras = []
        for index in range(self.max_cameras):
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

    def calculate_distance(self, detection_r, detection_l):
        """
        Calculates the distance to an object based on the angle of view from two cameras.

        :param detection_r: The detection data from the right camera.
        :param detection_l: The detection data from the left camera.
        :return: The calculated distance to the object in meters.
        """
        if detection_r and detection_l:  # Ensure both detections are valid
            # Directly access the 'x' angle of view from both right and left detections
            out_angle_r = abs(detection_r['x'])  # Using 'x' key from angle_of_view
            out_angle_l = abs(detection_l['x'])  # Using 'x' key from angle_of_view
            y_angle_r = abs(detection_r['y'])
            y_angle_l = abs(detection_l['y'])

            avg_y_angle = ((y_angle_r + y_angle_l)/2)*(np.pi / 180)

            print(f"Right camera angle of view (x): {out_angle_r:.2f}")
            print(f"Left camera angle of view (x): {out_angle_l:.2f}")

            # Convert the angles to radians for trigonometry calculations
            inner_angle_right = (90 - out_angle_r) * np.pi / 180
            inner_angle_left = (90 - out_angle_l) * np.pi / 180

            # Use trigonometry to calculate the distance to the object
            distance_to_object = self.cameras_distance * (
                (np.sin(inner_angle_right) * np.sin(inner_angle_left)) / 
                np.sin(inner_angle_right + inner_angle_left)
            )

            real_distance_to_object = distance_to_object/np.cos(avg_y_angle)
            return real_distance_to_object
        else:
            return 0