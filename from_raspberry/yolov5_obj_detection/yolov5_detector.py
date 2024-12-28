# yolov5_detector
import torch
import cv2
import time
#from datetime import datetime
import os




class YOLOv5Detector:
    def __init__(self, model_name='yolov5n', img_size=320, conf_threshold=0.25, width=640, height=480, output_dir="/home/admin/GIT/naggles/static/images", horizontal_fov = 60, vertical_fov = 47):
        """
        Initializes the YOLOv5 detector.

        :param model_name: YOLOv5 model to use (e.g., 'yolov5n' or 'yolov5s').
        :param img_size: Image size to be used for detection.
        :param conf_threshold: Confidence threshold for object detection.
        :param width: Camera resolution width.
        :param height: Camera resolution height.
        :param output_dir: Directory to save images with detections.
        """
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

        # Ensure the directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def clean_up(self):
        # Cleanup
        cv2.destroyAllWindows()

    def detect_and_describe(self, source=0, save_images = True):
        """
        Performs object detection and returns detected objects with details.

        :param source: Video source (0 for webcam, path for video file or image directory).
        :return: A list of dictionaries containing details about each detected object.
        """
        #current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Format: YYYYMMDD_HHMMSS_milliseconds
        # Open the video source (0 for webcam)
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return []

        # Set the resolution of the camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            cap.release()
            return []

        # Perform object detection
        results = self.model(frame, size=self.img_size)

        # Save the frame with detections (rendered with bounding boxes)
        if results:
            results.render()  # Draw bounding boxes and labels on the frame
            if source == 0:
                output_filename = os.path.join(self.output_dir, "detected_image0.jpg")
                # if i want to patch the live time to the name of the image it's:
                #output_filename = os.path.join(self.output_dir, "detected_image0_{current_time}.jpg")
            else:
                output_filename = os.path.join(self.output_dir, "detected_image2.jpg")
            # save the image with the right index for each of the camera
            if save_images == True:
                cv2.imwrite(output_filename, frame)

        # Release the video source
        cap.release()

        return results, frame

    def process_detections(self, results, frame, res_width=640, res_height=480, horizontal_fov=60, vertical_fov=47):
        """
        Processes the detections and returns a list of detected objects with details.
        Crops the detected objects from the frame and stores them in the detection.

        :param results: YOLOv5 detection results.
        :param frame: The image frame in which detections were made.
        :return: A list of dictionaries containing details about each detected object, including cropped images.
        """
        df = results.pandas().xyxy[0]  # Bounding boxes with labels and confidence scores

        # Apply confidence threshold by filtering the results
        df = df[df['confidence'] >= self.conf_threshold]  # Filter by confidence threshold

        detections = []

        # Get the actual frame dimensions (width and height)
        frame_height, frame_width = frame.shape[:2]

        for index, row in df.iterrows():
            object_id = index
            object_name = row['name']
            confidence = row['confidence'] * 100  # Convert to percentage

            # Scale the bounding box coordinates to match the actual frame resolution
            bounding_box = {
                'xmin': int(row['xmin'] ),
                'ymin': int(row['ymin'] ),
                'xmax': int(row['xmax'] ),
                'ymax': int(row['ymax'] )
            }

            # Crop the object from the image using the bounding box coordinates
            cropped_image = frame[bounding_box['ymin']:bounding_box['ymax'], bounding_box['xmin']:bounding_box['xmax']]

            # Calculate the center pixel of the object in the scaled resolution
            object_center_x = int((bounding_box['xmin'] + bounding_box['xmax']) / 2)
            object_center_y = int((bounding_box['ymin'] + bounding_box['ymax']) / 2)
            image_center_x = frame_width // 2
            image_center_y = frame_height // 2
            obj_center_pixel = {
                'x': object_center_x,
                'y': object_center_y
            }
            img_center_pixels = {
                'x': image_center_x,
                'y': image_center_y
            }

            # Calculate the angle of view
            angle_of_view_x = round(((object_center_x - image_center_x) / image_center_x) * (horizontal_fov / 2), 3)
            angle_of_view_y = round(((object_center_y - image_center_y) / image_center_y) * (vertical_fov / 2), 3)

            angle_of_view = {
                'x': angle_of_view_x,
                'y': angle_of_view_y
            }

            # Store the cropped image in the detection dictionary instead of saving to disk
            detection = {
                'object_id': object_id,
                'object': object_name,
                'confidence': confidence,
                'bounding_box': bounding_box,
                'object_center_pixel': obj_center_pixel,
                'image_center_pixel': img_center_pixels,
                'angle_of_view': angle_of_view,
                'cropped_image': cropped_image  # Add cropped image directly here
            }

            detections.append(detection)

        return detections


    def print_detections(self, detections, width = 640, hight = 480):
        if detections is None:
            print("Detections are empty.")
        else:
            # Print the detected objects
            for detection in detections:
                print(f"Object ID: {detection['object_id']}, Object: {detection['object']}, "
                      f"Confidence: {detection['confidence']:.2f}%, "
                      f"Bounding Box: {detection['bounding_box']}, Center Of Object: {detection['object_center_pixel']}, "
                      f"Center Of Image: {detection['image_center_pixel']}, AOV: {detection['angle_of_view']}")
