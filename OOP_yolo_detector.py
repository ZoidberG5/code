# OOP_yolo_detector.py
import torch
import cv2
import time

class YOLOv5Detector:
    def __init__(
        self, 
        weights_path='yolov5s', 
        img_size=320, 
        conf_threshold=0.25, 
        width=640, 
        height=480, 
        output_dir="static/images", 
        horizontal_fov=60, 
        vertical_fov=47,
        device=None
    ):
        """
        Initializes the YOLOv5 detector.

        :param weights_path: Path to the YOLOv5 weights file.
        :param device: Torch device ('cuda:0' or 'cpu'). If None, automatically selects 'cuda:0' if available.
        """
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        try:
            if weights_path != 'yolov5s':
                print(f"Loading custom weights from: {weights_path}")
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 
                    'custom', 
                    path=weights_path, 
                    device=self.device,
                    force_reload=True  # Force reload to ensure the latest version
                )
            else:
                print("Loading default YOLOv5 model.")
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 
                    weights_path, 
                    device=self.device,
                    force_reload=True  # Force reload to ensure the latest version
                )
            self.model.eval()  # Set model to evaluation mode
            print(f"Model loaded successfully on {self.device.type}.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")

        # try:
        #     print(f"Loading YOLOv5 model from local file: {weights_path}")
        #     model_data = torch.load(weights_path, map_location=self.device)
        #     self.model = model_data['model'].float()
        #     self.model.to(self.device).eval()
        #     print(f"Model loaded successfully from {weights_path} on {self.device.type}.")
        # except Exception as e:
        #     print(f"Error loading YOLOv5 model from local file: {e}")
            raise e  # Re-raise exception after logging

        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

    def clean_up(self):
        # Cleanup
        cv2.destroyAllWindows()

    def detect_and_describe(self, frame, save_images=True, camera_index=0):
        """
        Performs object detection on a provided frame and returns detection results.

        :param frame: Image frame to perform detection on.
        :param save_images: Whether to save the detection results as an image.
        :param camera_index: Camera index to label the saved image correctly.
        :return: The detection results and the frame with detections.
        """
        print("Frame received for detection.")

        # Ensure frame is in RGB
        if frame is None:
            print("Received empty frame.")
            return None, frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        with torch.no_grad():
            results = self.model(frame_rgb, size=self.img_size)

        if results:
            results.render()  # This draws boxes and labels (optional if we draw manually)

            # Access detections
            detections = results.xyxy[0].cpu().numpy()  # Convert to numpy array

            # Iterate over detections to add object IDs
            for idx, detection in enumerate(detections):
                x1, y1, x2, y2, conf, cls = detection[:6]

                # Format the ID to display
                object_id = f"ID: {idx}"

                # Convert coordinates to integers for OpenCV
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Set the position to place the text (above the bounding box)
                text_position = (x1, y1 - 10 if y1 > 20 else y1 + 30)

                # Draw the text (ID + class label)
                label = f"ID: {idx} {self.model.names[int(cls)]} {conf:.2f}"
                cv2.putText(
                    frame, label, text_position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255), #BGR
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

            # Save the frame with detections
            output_filename = f"code/static/detected_image_{camera_index}.jpg"
            if save_images:
                saved = cv2.imwrite(output_filename, frame)
                if saved:
                    print(f"Image saved successfully at {output_filename}")
                else:
                    print(f"Failed to save image at {output_filename}")

        return results, frame

    def process_detections(self, results, frame, res_width=640, res_height=480, horizontal_fov=60, vertical_fov=47):
        """
        Processes the detections and returns a list of detected objects with details.
        Crops the detected objects from the frame and stores them in the detection.

        :param results: YOLOv5 detection results.
        :param frame: The image frame in which detections were made.
        :return: A list of dictionaries containing details about each detected object, including cropped images.
        """
        if results is None:
            return []

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
                'xmin': int(row['xmin']),
                'ymin': int(row['ymin']),
                'xmax': int(row['xmax']),
                'ymax': int(row['ymax'])
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

    def print_detections(self, detections):
        if not detections:
            print("Detections are empty.")
        else:
            # Print the detected objects
            for detection in detections:
                print(f"Object ID: {detection['object_id']}, Object: {detection['object']}, "
                      f"Confidence: {detection['confidence']:.2f}%, "
                      f"Bounding Box: {detection['bounding_box']}, Center Of Object: {detection['object_center_pixel']}, "
                      f"Center Of Image: {detection['image_center_pixel']}, AOV: {detection['angle_of_view']}")
