# usb_webcam_OOP
import cv2
import numpy as np

class CameraSystem:
    def __init__(self, cameras_distance=0.235, max_cameras=10):
        """
        Initializes the CameraSystem with the specified cameras distance and max cameras to scan.
        
        :param cameras_distance: The distance between the cameras in meters.
        :param max_cameras: The maximum number of cameras to scan for availability.
        """
        self.cameras_distance = cameras_distance  # Distance between the cameras
        self.max_cameras = max_cameras  # Maximum number of cameras to scan
        self.available_cameras = self.list_available_cameras()  # List of available camera indices

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

    def init_two_cameras(self, camera_indices, width=640, height=480):
        # Open the USB cameras (index depending on your setup)
        cap_left = cv2.VideoCapture(camera_indices[0])
        cap_right = cv2.VideoCapture(camera_indices[1])
        
        # Set the resolution (width and height)
        cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # # Enable auto-exposure
        # cap_left.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        # cap_right.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) 

        # # Enable auto-focus try this first!!!
        # cap_left.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        # cap_right.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        return cap_right, cap_left

    def release_two_cameras(self, cap_right, cap_left):
        # Release the camera
        cap_right.release()
        cap_left.release()
        cv2.destroyAllWindows()
        return

    def capture_images_from_cameras(self, cap_right, cap_left, camera_indices, save_path="code\static\images", width=640, height=480):
        """
        Captures images from the specified camera indices simultaneously.

        :param camera_indices: List of camera indices (e.g., [0, 1] for two cameras).
        :param width: Width of the capture resolution.
        :param height: Height of the capture resolution.
        :return: A dictionary with camera index as keys and captured frames as values.
        """
        # if len(camera_indices) != 2:
        #     print("This function is designed to capture images from exactly two cameras.")
        #     return {}

        # # Open both cameras
        # cap_right = cv2.VideoCapture(camera_indices[0])
        # cap_left = cv2.VideoCapture(camera_indices[1])

        # # Set the resolution for each camera
        # cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Check if both cameras opened successfully
        if not cap_left.isOpened():
            print(f"Error: Could not open left camera (index {camera_indices[0]}).")
            cap_right.release()  # Release right camera if left camera fails
            return {}
        if not cap_right.isOpened():
            print(f"Error: Could not open right camera (index {camera_indices[1]}).")
            cap_left.release()  # Release left camera if right camera fails
            return {}

        # Capture frames from both cameras simultaneously
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        frames = {}

        if ret_left:
            frames[0] = frame_left
            # Save each image with a unique filename
            image_path = f"{save_path}/captured_image_LEFT.jpg"
            cv2.imwrite(image_path, frame_left)
        else:
            print(f"Error: Could not read frame from left camera (index {camera_indices[0]}).")

        if ret_right:
            frames[1] = frame_right
            # Save each image with a unique filename
            image_path = f"{save_path}/captured_image_RIGHT.jpg"
            cv2.imwrite(image_path, frame_right)
        else:
            print(f"Error: Could not read frame from right camera (index {camera_indices[1]}).")

        # # Release both cameras
        # cap_left.release()
        # cap_right.release()

        return frames

    def calculate_distance(self, detection_r, detection_l):
        """
        Calculates the distance to an object based on the angle of view from two cameras.

        :param detection_r: The detection data from the right camera.
        :param detection_l: The detection data from the left camera.
        :return: The calculated distance to the object in meters.
        """
        if detection_r and detection_l:  # Ensure both detections are valid
            # Directly access the 'x' angle of view from both right and left detections
            out_angle_r = detection_r['x']  # Using 'x' key from angle_of_view
            out_angle_l = detection_l['x']  # Using 'x' key from angle_of_view
            y_angle_r = detection_r['y']
            y_angle_l = detection_l['y']

            avg_y_angle = ((abs(y_angle_r) + abs(y_angle_l))/2)*(np.pi / 180)

            print(f"Right camera angle of view (x): {out_angle_r:.2f}")
            print(f"Left camera angle of view (x): {out_angle_l:.2f}")

            if out_angle_l > 0 and out_angle_r < 0: #between the cameras
                inner_angle_right = (90 - abs(out_angle_r))
                inner_angle_left = (90 - abs(out_angle_l))
            elif out_angle_l < 0 and out_angle_r < 0: #left from the left camera
                if abs(out_angle_l) < abs(out_angle_r):
                    inner_angle_right = (90 - abs(out_angle_r))
                    inner_angle_left = (90 + abs(out_angle_l))
                else:    
                    inner_angle_right = (90 + abs(out_angle_r))
                    inner_angle_left = (90 - abs(out_angle_l))
            elif out_angle_l > 0 and out_angle_r > 0: #right from the right camera
                if abs(out_angle_l) < abs(out_angle_r):
                    inner_angle_right = (90 - abs(out_angle_r))
                    inner_angle_left = (90 + abs(out_angle_l))
                else:    
                    inner_angle_right = (90 + abs(out_angle_r))
                    inner_angle_left = (90 - abs(out_angle_l))
            else:
                return 0
            
            # Calculate internal angle of the object
            internal_angle_obj = 180 - (inner_angle_left + inner_angle_right)
            if internal_angle_obj <= 0: # need to be check! dosen't look good in tests (left from both cameras)
                return -4

            # Convert the angles to radians for trigonometry calculations
            internal_angle_obj = np.radians(internal_angle_obj)
            inner_angle_left = np.radians(inner_angle_left)
            inner_angle_right = np.radians(inner_angle_right)

            # Calculate distance using trigonometric relations
            denominator = (np.sin(internal_angle_obj) ** 2)
            if denominator == 0:
                return -3
            
            numerator = (np.sin(inner_angle_left) ** 2) + (np.sin(inner_angle_right) ** 2)
            # Ensure that the value inside sqrt is positive
            value_inside_sqrt = 2 * (numerator / denominator) - 1
            if value_inside_sqrt < 0:
                return -2
            
            # Use trigonometry to calculate the distance to the object
            distance_to_object_x = (self.cameras_distance / 2) * np.sqrt(value_inside_sqrt)
            #return distance_to_object_x
            real_distance_to_object = distance_to_object_x/np.cos(avg_y_angle)
            return real_distance_to_object
        else:
            return -1
    
    def calculate_distance_niconielsen32(self, center_object_right, center_object_left, frame_right, frame_left, baseline, f, alpha):
        # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
        height_right, width_right, depth_right = frame_right.shape
        height_left, width_left, depth_left = frame_left.shape

        if width_right == width_left:
            f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

        else:
            print('Left and right camera frames do not have the same pixel width')
                                
        x_right = center_object_right['x']
        x_left = center_object_left['x']

        # CALCULATE THE DISPARITY:
        disparity = x_left-x_right      #Displacement between left and right frames [pixels]

        # CALCULATE DEPTH z:
        zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]

        return abs(zDepth)

    def increase_image_quality(self, image, blur_kernel=(15, 15), alpha=2.0, beta=-1.0):
        """
        Enhances the quality of an image by increasing local contrast.

        :param image: Input image (numpy.ndarray or similar).
        :param blur_kernel: Kernel size for Gaussian blur.
        :param alpha: Weight of the original image.
        :param beta: Weight of the blurred image.
        :return: Enhanced image.
        """
        if image is None:
            raise ValueError("Input image is None. Please provide a valid image.")

        # Convert to numpy.ndarray if it's not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, blur_kernel, sigmaX=10, sigmaY=10)

        # Enhance local contrast
        enhanced_image = cv2.addWeighted(image, alpha, blurred, beta, 0)

        return enhanced_image

if __name__ == "__main__":
    # Create an instance of the CameraSystem class
    camera_system = CameraSystem(cameras_distance=0.235, max_cameras=10)
    
    # Call the `list_available_cameras` method
    available_cameras = camera_system.list_available_cameras()
    
    # Print the available cameras
    print(f"Available cameras: {available_cameras}")