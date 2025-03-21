# main_with_gui.py
import logging
import time
import cv2
import numpy as np
import os
import sys
import threading
from datetime import datetime
from OOP_webcam import CameraSystem
from OOP_yolo_detector import YOLOv5Detector
from OOP_GUI_radar import RadarDisplay
import torch
import json

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%d.%m.%Y")
current_time = now.strftime("%d.%m.%Y_%H.%M.%S")

# Create the directory for the current date if it doesn't exist
log_dir = f"code/static/logs/logs_{current_date}"
os.makedirs(log_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(log_dir, f"log_{current_time}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Change to DEBUG or ERROR depending on verbosity needs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def compare_orb(image1, image2, min_matches=10):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    if des1 is None or des2 is None:
        print("No descriptors found in one or both images.")
        return False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) >= min_matches:
        return True, len(matches)
    else:
        return False, len(matches)

def resize_images_to_same_size(image1, image2):
    if image1 is None or image2 is None:
        raise ValueError("One of the images is None.")
    
    if image1.size == 0 or image2.size == 0:
        raise ValueError("One of the images has zero size.")
    
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    if height1 == 0 or width1 == 0 or height2 == 0 or width2 == 0:
        raise ValueError("One of the images has zero height or width.")

    new_width, new_height = min(width1, width2), min(height1, height2)

    resized_image1 = cv2.resize(image1, (new_width, new_height))
    resized_image2 = cv2.resize(image2, (new_width, new_height))

    return resized_image1, resized_image2

def find_max_in_orb_shai(matrix):
    pairs = []
    max_pairs = []
    used_indices_r = []
    used_indices_l = []

    # Collect all (r, l, orb_result) pairs
    for r in range(len(matrix)):
        for l in range(len(matrix[r])):
            is_2_similar, orb_result, details = matrix[r][l]
            if (details['object_name_right'] == details['object_name_left']) and abs(abs(details['right_image_aov']['x']) - abs(details['left_image_aov']['x'])) < config['correlation']['ANGLE_PAIR_THRESHOLD']:
                pairs.append(((r, l), orb_result))  # Append as a tuple ((r, l), orb_result)
            else:
                continue

    # Sort pairs by orb_result in descending order
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    for pos in range(len(sorted_pairs)):
        (r, l) = sorted_pairs[pos][0]  # Extract (r, l) from the sorted pair
        orb_value = sorted_pairs[pos][1]  # Extract orb_result from the sorted pair
        
        # Check if current (r, l) is not yet used
        if (r not in used_indices_r) and (l not in used_indices_l):
            used_indices_r.append(r)  # Mark as used
            used_indices_l.append(l)  # Mark as used
            max_pairs.append(((r, l), orb_value))  # Append to max_pairs
        else:
            continue
    return max_pairs

def find_maximum_orb_result_pairs(matrix):
    max_pairs = []
    used_indices = set()  # Set to track used (row, col) pairs

    # Loop through each row in the matrix
    for i in range(len(matrix)):
        max_value = None
        max_index = None

        # Loop through each element in the row
        for j in range(len(matrix[i])):
            is_2_similar, _, details = matrix[i][j]
            if is_2_similar != False:  # in case of error handling one of the two objects OR the objects are not alike
                orb_result = details['orb_result']

                # Check if current orb_result is the max and not yet used
                if (i, j) not in used_indices and (max_value is None or orb_result > max_value):
                    max_value = orb_result
                    max_index = (i, j)
            else:
                break
        # If a max was found, add to results and mark row and column as used
        if max_index:
            max_pairs.append((max_index, max_value))
            # Mark the row and column as used to prevent overlap
            row, col = max_index
            for k in range(len(matrix[i])):
                used_indices.add((row, k))  # Mark entire row
            for k in range(len(matrix)):
                used_indices.add((k, col))  # Mark entire column

    return max_pairs

def filter_out_object_types(radar_objects, types_to_exclude):
    """
    Filters out objects whose 'name' is in the types_to_exclude list.

    Args:
        radar_objects (list): List of object dictionaries.
        types_to_exclude (list): List of object names to exclude (e.g., ['car', 'truck']).

    Returns:
        list: Filtered list of radar objects.
    """
    return [obj for obj in radar_objects if obj['name'] in types_to_exclude]


class TrackedObject:
    def __init__(self, object_id, name, distance, angle):
        self.id = object_id
        self.name = name

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Initialize both pre and post states
        initial_state = np.array([[distance], [angle], [0], [0]], dtype=np.float32)
        self.kf.statePre = initial_state.copy()
        self.kf.statePost = initial_state.copy()

        self.has_been_updated = False

    def update(self, distance, angle):
        measurement = np.array([[np.float32(distance)], [np.float32(angle)]])
        self.kf.correct(measurement)
        self.has_been_updated = True

    def predict(self):
        if not self.has_been_updated:
            return self.kf.statePre[0][0], self.kf.statePre[1][0]
        prediction = self.kf.predict()
        return prediction[0][0], prediction[1][0]

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0

    def update(self, detections):
        matched_ids = set()
        updated_tracked = {}

        for det in detections:
            name = det['name']
            distance = det['distance']
            angle = det['angle']

            # Skip invalid readings
            if distance == 0.0 and angle == 0.0:
                continue

            matched = False
            for obj_id, obj in self.tracked_objects.items():
                pred_distance, pred_angle = obj.predict()
                if obj.name == name and abs(pred_distance - distance) < 1.5 and abs(pred_angle - angle) < 10:
                    obj.update(distance, angle)
                    updated_tracked[obj_id] = obj
                    matched_ids.add(obj_id)
                    matched = True
                    break

            if not matched:
                new_obj = TrackedObject(self.next_id, name, distance, angle)
                updated_tracked[self.next_id] = new_obj
                self.next_id += 1

        self.tracked_objects = updated_tracked

        # Return list of predictions
        filtered_results = []
        for obj in self.tracked_objects.values():
            dist, ang = obj.predict()
            filtered_results.append({
                'id': obj.id,
                'name': obj.name,
                'distance': round(float(dist), 2),
                'angle': round(float(ang), 2)
            })

        return filtered_results


# Camera and resolution configuration
#config['camera']['image_resolution'] = [1920, 1080]
#config['camera']['camera_fov'] = [62.2, 48.8]
#config['camera']['f'] = 3.04  # focal length in [mm]

# Configuration settings
with open('code/config.json', 'r') as f:
    config = json.load(f)

#config['CONF_THRESHOLD']  # Confidence threshold for object detection
#config['correlation']['ANGLE_PAIR_THRESHOLD'] = 5  # Maximum difference in angle of view for a pair of objects to be considered a match

monitoring_rounds_of_detections = []

# YOLOv5 detector configuration
weights_path_chosen = 'yolov5s'  # or your custom weights path

# The main processing loop that captures images, performs object detection, and updates the radar display
def processing_loop(detector, usb_camera, radar_display, camera_indices, device):
    """
    The processing loop that captures images, performs object detection, and updates the radar display.
    Runs in a separate thread to keep Tkinter responsive in the main thread.
    """
    logging.info("----------GPU details----------")
    logging.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("No CUDA device detected. The program will run on CPU.")
        print("Running on CPU")
    logging.info("----------Processing loop started----------")
    try:
        if config['image_capture']['TAKE_NEW_IMAGES']:
            cap_left, cap_right = usb_camera.init_two_cameras(camera_indices, width=config['camera']['image_resolution'][0], height=config['camera']['image_resolution'][1])
        elif config['runtime']['USE_SAVED_VIDEO']:
            # Initialize video files instead of cameras
            VIDEO_DIR = "code/videos"
            video_path_left = os.path.join(VIDEO_DIR, "20250321_121636_LEFT_video.mp4")
            video_path_right = os.path.join(VIDEO_DIR, "20250321_121636_RIGHT_video.mp4")
            cap_left = cv2.VideoCapture(video_path_left)
            cap_right = cv2.VideoCapture(video_path_right)
        round_of_detection = 0

        # print("Checking if video files exist:")
        # print(f"Left exists? {os.path.exists(video_path_left)} -> {video_path_left}")
        # print(f"Right exists? {os.path.exists(video_path_right)} -> {video_path_right}")

        while True:
            try:
                logging.info("----New round of detections----")
                print("Processing frames...")
                # Capture images
                if config['image_capture']['TAKE_NEW_IMAGES']:
                    frames = usb_camera.capture_images_from_cameras(
                        cap_left, cap_right, camera_indices, save_path="code/static/images", 
                        width=config['camera']['image_resolution'][0], height=config['camera']['image_resolution'][1])
                elif config['runtime']['USE_SAVED_VIDEO']:
                    # Read frames from video files
                    ret_left, frame_left = cap_left.read()
                    ret_right, frame_right = cap_right.read()
                    if not ret_left or not ret_right:
                        logging.info("Failed to read frames from video. Exiting loop.")
                        print("Failed to read frames from video. Exiting loop.")
                        break
                    frames = {0: frame_left, 1: frame_right}
                else:
                    left_image_path = "code/images/LEFT_image_1_20241202_161508.jpg"
                    right_image_path = "code/images/RIGHT_image_1_20241202_161508.jpg"
                    frame_left = cv2.imread(left_image_path)
                    frame_right = cv2.imread(right_image_path)
                    frames = {0: frame_left, 1: frame_right}
                
                # Extract the frames from the dictionary
                frame_left = frames[0]
                frame_right = frames[1]

                # Process frames for quality enhancement
                frames = {0: usb_camera.increase_image_quality(frame_left), 1: usb_camera.increase_image_quality(frame_right)}

                # Perform detection and get matched objects
                detections_r, detections_l = [], []
                # For right camera
                results_r, frame_r = detector.detect_and_describe(frame=frames[1], save_images=config['image_capture']['SAVE_IMAGES'], camera_index=camera_indices[1])
                if results_r is not None:
                    detections_r = detector.process_detections(results_r, frame_r, config['camera']['image_resolution'][0], config['camera']['image_resolution'][1], config['camera']['camera_fov'][0], config['camera']['camera_fov'][1])
                    detector.print_detections(detections_r)
                # For left camera
                results_l, frame_l = detector.detect_and_describe(frame=frames[0], save_images=config['image_capture']['SAVE_IMAGES'], camera_index=camera_indices[0])
                if results_l is not None:
                    detections_l = detector.process_detections(results_l, frame_l, config['camera']['image_resolution'][0], config['camera']['image_resolution'][1], config['camera']['camera_fov'][0], config['camera']['camera_fov'][1])
                    detector.print_detections(detections_l)

                # This section saves the detailed images taken during the detection
                images_dir = f"{log_dir}/images_{current_time}"
                os.makedirs(images_dir, exist_ok=True)  # Create the directories if they don't exist
                current_time_images = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
                output_filename_r = f"{images_dir}/{current_time_images}_Right.jpg"
                output_filename_l = f"{images_dir}/{current_time_images}_Left.jpg"
                flag_save_left_image_to_logs = False
                flag_save_right_image_to_logs = False

                if detections_r and detections_l:  # Check if there are any detections to be worked with
                    for conf_arr_r in detections_r:  # Check all the detections to collect the confidence for each object
                        if (conf_arr_r['confidence'] > config['object_detection']['CONF_THRESHOLD']):
                            flag_save_left_image_to_logs = True

                    message = "Right Images saved successfully" if flag_save_left_image_to_logs else "Failed to save the right images"
                    if flag_save_left_image_to_logs:
                        cv2.imwrite(output_filename_r, frame_r)
                    print(message)

                    for conf_arr_l in detections_l:  # Check all the detections to collect the confidence for each object
                        if (conf_arr_l['confidence'] > config['object_detection']['CONF_THRESHOLD']):
                            flag_save_right_image_to_logs = True
                    if flag_save_right_image_to_logs:
                        cv2.imwrite(output_filename_l, frame_l)
                        print(f"Left Images saved successfully")
                    else:
                        print("Failed to save the left images")

                round_of_objects = []
                if detections_r and detections_l: 
                    matched_objects = []
                    correlation_results_matrix = np.empty((len(detections_l), len(detections_r)), dtype=object)

                    # Initialize the correlation results matrix
                    for l in range(len(detections_l)):
                        for r in range(len(detections_r)):
                            correlation_results_matrix[l, r] = (False, 0, {})

                    # Process the ORB matching for object detection
                    for r, obj_r in enumerate(detections_r):
                        for l, obj_l in enumerate(detections_l):
                            try:
                                cropped_img_r = obj_r['cropped_image']
                                cropped_img_l = obj_l['cropped_image']
                                if cropped_img_r is None or cropped_img_l is None or cropped_img_r.size == 0 or cropped_img_l.size == 0:
                                    print(f"Error: Invalid cropped images for object {r}. Skipping comparison.")
                                    continue

                                cropped_img_r_resized, cropped_img_l_resized = resize_images_to_same_size(cropped_img_r, cropped_img_l)
                                is_similar, orb_result = compare_orb(cropped_img_r_resized, cropped_img_l_resized)

                                matched_object_data = {
                                    'object_id_r': r,
                                    'object_id_l': l,
                                    'confidence_r': round(float(obj_r['confidence']), 2),
                                    'confidence_l': round(float(obj_l['confidence']), 2),
                                    'object_name_right': obj_r['object'],
                                    'object_name_left': obj_l['object'],
                                    'object_center_right': obj_r['object_center_pixel'],
                                    'object_center_left': obj_l['object_center_pixel'],
                                    'right_image_aov': obj_r['angle_of_view'],
                                    'left_image_aov': obj_l['angle_of_view'],
                                    'orb_result': orb_result
                                }
                                matched_objects.append(matched_object_data)
                                correlation_results_matrix[l, r] = (is_similar, orb_result, matched_object_data)
                            except Exception as e:
                                logging.error("Error during ORB comparison", exc_info=True)

                    logging.info("-----------------------------------------------------")                    
                    logging.info(f"\nThe correlation matrix: {correlation_results_matrix}")

                    # Find the best matches and calculate distances
                    result = find_max_in_orb_shai(correlation_results_matrix)
                    logging.info(f"The results of Max orb pairs are: {result}")

                    radar_objects = []
                    round_of_objects = []
                    for i in range(len(result)):
                        matched_object = correlation_results_matrix[result[i][0][0], result[i][0][1]][2]
                        if config['algorithm']['USING_Niconielsen32_ALGORITHEM']:  # Calculate the distance using Niconielsen32 algorithm
                            right_center_object = matched_object['object_center_right']
                            left_center_object = matched_object['object_center_left']
                            distance_to_object = usb_camera.calculate_distance_niconielsen32(
                                right_center_object, left_center_object, frames[0], frames[1], usb_camera.cameras_distance, f, config['camera']['camera_fov'][0]
                            )
                        else: 
                            # Calculate the distance using our algorithm
                            right_aov = matched_object['right_image_aov']
                            left_aov = matched_object['left_image_aov']
                            distance_to_object = usb_camera.calculate_distance(right_aov, left_aov)
                        
                        angle_of_view_x = (matched_object['right_image_aov']['x'] + matched_object['left_image_aov']['x']) / 2
                        angle_of_view_y = (matched_object['right_image_aov']['y'] + matched_object['left_image_aov']['y']) / 2

                        if distance_to_object != -4.0:
                            error_correction = (90 / (config['camera']['camera_fov'][0] / 2))
                            distance_to_object = abs(distance_to_object * np.cos(np.deg2rad(angle_of_view_x * error_correction)))  # Calculate the distance in the same line of the object
                        
                        # Append the object data to the radar_objects list
                        radar_objects.append({
                            'name': matched_object['object_name_right'],
                            'distance': round(float(distance_to_object), 2),  # Convert to float
                            'angle': round(angle_of_view_x, 2)
                        })

                        # Append the object data to the round_of_objects list
                        round_of_objects.append({
                            'name': matched_object['object_name_right'],
                            'distance': round(float(distance_to_object), 2),  # Convert to float
                            'angle_x': round(angle_of_view_x, 2),
                            'angle_y': round(angle_of_view_y, 2)
                        })

                elif (len(detections_l) == 0) or (len(detections_r) == 0):  # In case there is no detection from one of the images, reset radar_objects
                    radar_objects = []
                
                # filtered_objects = filter_out_object_types(radar_objects, ['person'])

                # # Update radar display with the new object list
                # radar_display.update_display(filtered_objects)
                # logging.info(f"The data of objects: {filtered_objects}")


                # This section is for the object tracking
                filtered_objects = filter_out_object_types(radar_objects, ['person'])
                smoothed_objects = object_tracker.update(filtered_objects)
                radar_display.update_display(smoothed_objects)
                logging.info(f"The data of objects (smoothed): {smoothed_objects}")

                # Save the round of objects to the monitoring list
                if round_of_objects != []:
                    monitoring_rounds_of_detections.append((round_of_detection, filtered_objects)) #monitoring_rounds_of_detections is a list of tuples that save all the objects detected in each round all together
                round_of_detection += 1
                print(monitoring_rounds_of_detections)

                # Optional: Adjust sleep time as needed
                time.sleep(0.2)  # Delay for the next loop iteration
            except Exception as e:
                logging.critical("Critical error in while loop", exc_info=True)
                logging.info("The video ended and now the App will be closed! Bye Bye")
                break

            if config['runtime']['RUN_ONE_TIME_ONLY'] is True:
                break
    except Exception as e:
        logging.critical("Critical error in processing loop", exc_info=True)

if __name__ == "__main__":
    logging.info("Application started")
    try:
        # Determine device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device.type}")
        logging.info(f"Using device: {device.type}")

        # Initialize the camera, YOLO detector, and radar display
        usb_camera = CameraSystem(cameras_distance=0.235, max_cameras=10)
        camera_indices = (2, 1)  # [left index, right index]
        detector = YOLOv5Detector(
            weights_path=weights_path_chosen,  # e.g., 'yolov5s'
            img_size=320,
            conf_threshold=config['object_detection']['CONF_THRESHOLD'],
            width=config['camera']['image_resolution'][0],
            height=config['camera']['image_resolution'][1],
            output_dir="code/static",
            horizontal_fov=config['camera']['camera_fov'][0],
            vertical_fov=config['camera']['camera_fov'][1],
            device=device  # Pass the device here
        )
        object_tracker = ObjectTracker()

        # Create RadarDisplay and start the processing loop in a separate thread
        radar_display = RadarDisplay([])

        # Start the processing loop in a separate thread
        processing_thread = threading.Thread(
            target=processing_loop, 
            args=(detector, usb_camera, radar_display, camera_indices, device)
        )
        processing_thread.daemon = True  # Daemonize the thread so it exits when the main program exits
        processing_thread.start()

        # Run the Tkinter mainloop in the main thread
        radar_display.run()
    except Exception as e:
        logging.critical("Critical error in application", exc_info=True)
