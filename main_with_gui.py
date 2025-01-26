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

    #print(f"Keypoints in image1: {len(kp1)}, Keypoints in image2: {len(kp2)}")

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

    # # save the new cropped images
    # output_filename = os.path.join("/home/admin/GIT/naggles/static/images", "cropped_image0.jpg")
    # cv2.imwrite(output_filename, image1)
    # output_filename = os.path.join("/home/admin/GIT/naggles/static/images", "cropped_image2.jpg")
    # cv2.imwrite(output_filename, image2)

    # # save the new resized images
    # output_filename = os.path.join("/home/admin/GIT/naggles/static/images", "resized_image0.jpg")
    # cv2.imwrite(output_filename, resized_image1)
    # output_filename = os.path.join("/home/admin/GIT/naggles/static/images", "resized_image2.jpg")
    # cv2.imwrite(output_filename, resized_image2)

    return resized_image1, resized_image2

def find_max_in_orb_shai(matrix):
    pairs = []
    max_pairs = []
    used_indices_r = []
    used_indices_l = []

    # Collect all (r, l, orb_result) pairs
    for r in range(len(matrix)):
        for l in range(len(matrix[r])):
            is_2_similar, _, details = matrix[r][l]
            orb_result = details['orb_result']
            if (details['object_name_right'] == details['object_name_left']) and abs(abs(details['right_image_aov']['x']) - abs(details['left_image_aov']['x']))<ANGLE_PAIR_THRESHOLD:
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
            if is_2_similar != False: # in case of error handeling one of the two objects OR the objects are not a like
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

# Camera and resolution configuration
image_resolution = [1920, 1080]
camera_fov = [62.2, 48.8]
f = 3.04 # focal length in [mm]

# Configuration settings
config = {
    'RUN_ONE_TIME_ONLY': False, # run the code one time only
    'TAKE_NEW_IMAGES': False,   # run the code on saved images or use the camera to take live images
    'USE_SAVED_VIDEO': True,    # run the code on saved video or use the camera to take live images
    'SAVE_IMAGES': True,        # save the images taken during the detection
    'USING_Niconielsen32_ALGORITHEM' : False #Calculate the distance according to Nico Algorithem or our Algorithem
}

CONF_THRESHOLD = 0.6 # Confidence threshold for object detection
ANGLE_PAIR_THRESHOLD = 5  # Maximum difference in angle of view for a pair of objects to be considered a match

monitoring_rounds_of_detections = []

# YOLOv5 detector configuration
# weights_path_chosen = 'yolov5/runs/train/exp/weights/best.pt'
weights_path_chosen = 'yolov5s'

# The main processing loop that captures images, performs object detection, and updates the radar display
def processing_loop(detector, usb_camera, radar_display, camera_indices):
    """
    The processing loop that captures images, performs object detection, and updates the radar display.
    Runs in a separate thread to keep Tkinter responsive in the main thread.
    """
    logging.info("----------GPU details----------")
    logging.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("No CUDA device detected. The program will run on CPU.")

    logging.info("----------Processing loop started----------")
    try:
        if config['TAKE_NEW_IMAGES']:
            cap_left, cap_right = usb_camera.init_two_cameras(camera_indices, width=image_resolution[0], height=image_resolution[1])
        elif config['USE_SAVED_VIDEO']:
            # Initialize video files instead of cameras
            VIDEO_DIR = "code/videos"
            video_path_left = os.path.join(VIDEO_DIR, "20250110_095814_LEFT_video.mp4")
            video_path_right = os.path.join(VIDEO_DIR, "20250110_095814_RIGHT_video.mp4")
            cap_left = cv2.VideoCapture(video_path_left)
            cap_right = cv2.VideoCapture(video_path_right)
        round_of_detection = 0

        while True:
            try:
                logging.info("----New round of detections----")
                print("Processing frames...")
                # Capture images
                if config['TAKE_NEW_IMAGES']:
                    frames = usb_camera.capture_images_from_cameras(
                        cap_left, cap_right, camera_indices, save_path="code/static/images", 
                        width=image_resolution[0], height=image_resolution[1])

                elif config['USE_SAVED_VIDEO']:
                    # Read frames from video files
                    ret_left, frame_left = cap_left.read()
                    ret_right, frame_right = cap_right.read()
                    frames = {0: frame_left, 1: frame_right}

                else:
                    left_image_path = "code\images\LEFT_image_1_20241202_161508.jpg"
                    right_image_path = "code\images\RIGHT_image_1_20241202_161508.jpg"
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
                # for _, frame in frames.items():
                    #print(f"Running detection on camera {camera_index}")
                    # if camera_index == 1:
                results_r, frame_r = detector.detect_and_describe(frame=frames[1], save_images=config['SAVE_IMAGES'], camera_index=camera_indices[1])
                detections_r = detector.process_detections(results_r, frame_r, image_resolution[0], image_resolution[1], camera_fov[0], camera_fov[1])
                detector.print_detections(detections_r)
                # else:
                results_l, frame_l = detector.detect_and_describe(frame=frames[0], save_images=config['SAVE_IMAGES'], camera_index=camera_indices[0])
                detections_l = detector.process_detections(results_l, frame_l, image_resolution[0], image_resolution[1], camera_fov[0], camera_fov[1])
                detector.print_detections(detections_l)

                #This section is saved the detailed images taken during the detection
                images_dir = f"{log_dir}/images_{current_time}"
                os.makedirs(images_dir, exist_ok=True)  # Create the directories if they don't exist
                current_time_images = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
                output_filename_r = f"{images_dir}/{current_time_images}_Right.jpg"
                output_filename_l = f"{images_dir}/{current_time_images}_Left.jpg"
                flag_save_left_image_to_logs = False
                flag_save_right_image_to_logs = False

                if detections_r and detections_l: #check if there are any detections to be worked with
                        for conf_arr_r in detections_r: # check all the detections to colect the confidence for each objects
                            if (conf_arr_r['confidence'] > CONF_THRESHOLD):
                                flag_save_left_image_to_logs = True

                        message = "Right Images saved successfully" if flag_save_left_image_to_logs else "Failed to save the right images"
                        if flag_save_left_image_to_logs:
                            cv2.imwrite(output_filename_r, frame_r)
                        print(message)

                        for conf_arr_l in detections_l: # check all the detections to colect the confidence for each objects
                            if (conf_arr_l['confidence'] > CONF_THRESHOLD):
                                flag_save_right_image_to_logs = True
                        if flag_save_right_image_to_logs:
                            cv2.imwrite(output_filename_l, frame_l)
                            print(f"Left Images saved successfully")
                        else:
                            print("Failed to save the left images")
                    # else:
                    #     print("Error: Frame is None.")

                if detections_r and detections_l: 
                    matched_objects = []
                    correlation_results_matrix = np.zeros((len(detections_l), len(detections_r)), dtype=object)

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

                                # if is_similar:
                                matched_object_data = {
                                    'object_id_r': r,
                                    'object_id_l': l,
                                    'confidence_r': round(float(obj_r['confidence']),2),
                                    'confidence_l': round(float(obj_l['confidence']),2),
                                    'object_name_right': obj_r['object'],
                                    'object_name_left': obj_l['object'],
                                    'object_center_right': obj_r['object_center_pixel'],
                                    'object_center_left': obj_l['object_center_pixel'],
                                    'right_image_aov': obj_r['angle_of_view'],
                                    'left_image_aov': obj_l['angle_of_view'],
                                    'orb_result': orb_result
                                }
                                matched_objects.append(matched_object_data)
                                # else:
                                #     # in case of 2 objects are not similar we dont use their data
                                #     matched_object_data = None
                                correlation_results_matrix[l, r] = (is_similar, orb_result, matched_object_data)
                            except Exception as e:
                                logging.error("Error during ORB comparison", exc_info=True)
                    #logging.info(f"the objects are: {matched_objects}")
                    logging.info("-----------------------------------------------------")                    
                    logging.info(f"\nthe correlation matrixe: {correlation_results_matrix}")

                    # Find the best matches and calculate distances
                    #result = find_maximum_orb_result_pairs(correlation_results_matrix)
                    result = find_max_in_orb_shai(correlation_results_matrix)
                    logging.info(f"The results of Max orb pairs are: {result}")

                    radar_objects = []
                    round_of_objects = []
                    for i in range(len(result)):
                        matched_object = correlation_results_matrix[result[i][0][0], result[i][0][1]][2]
                        if config['USING_Niconielsen32_ALGORITHEM']: #in this section we calculate the distanc according to the algorithem
                            # in this section we calculate the distance using Niconielsen32 algorithem
                            right_center_object = matched_object['object_center_right']
                            left_center_object = matched_object['object_center_left']
                            distance_to_object = usb_camera.calculate_distance_niconielsen32(
                                right_center_object, left_center_object, frames[0], frames[1], usb_camera.cameras_distance, f, camera_fov[0]
                            )
                        else: 
                            # in this section we calculate the distance using our algorithem
                            right_aov = matched_object['right_image_aov']
                            left_aov = matched_object['left_image_aov']
                            distance_to_object = usb_camera.calculate_distance(right_aov, left_aov)
                        
                        angle_of_view_x = (matched_object['right_image_aov']['x'] + matched_object['left_image_aov']['x'])/2
                        angle_of_view_y = (matched_object['right_image_aov']['y'] + matched_object['left_image_aov']['y'])/2

                        if distance_to_object != -4.0:
                            error_correction = (90/(camera_fov[0]/2))
                            distance_to_object = abs(distance_to_object*np.cos(np.deg2rad(angle_of_view_x*error_correction))) # calculate the distance in the same line of the object
                        
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

                elif (len(detections_l) == 0) or (len(detections_r) == 0): # in case there is no detection from one of the images, reset radar_objects
                    radar_objects = []
                # Update radar display with the new object list
                radar_display.update_display(radar_objects)
                logging.info(f"The data of objects: {radar_objects}")

                # In this part i'll handle the the comperison between the detections now and the detections before
                distance_error_ratio = 0.1
                angle_x_error_ratio = 0.1
                angle_y_error_ratio = 0.1

                monitoring_rounds_of_detections.append((round_of_detection, round_of_objects))
                round_of_detection += 1

                print(monitoring_rounds_of_detections)

                #time.sleep(0.1)  # Delay for the next loop iteration
            except Exception as e:
                logging.critical("Critical error in while loop", exc_info=True)
                logging.info(f"The video ended and now the App will be close! Bey Bey")
                break

            if config['RUN_ONE_TIME_ONLY'] is True:
                break
    except Exception as e:
        logging.critical("Critical error in processing loop", exc_info=True)

if __name__ == "__main__":
    logging.info("Application started")
    try:
        # Initialize the camera, YOLO detector, and radar display
        usb_camera = CameraSystem(cameras_distance=0.235, max_cameras=10)
        camera_indices = (2, 0)  # [left index, right index]
        detector = YOLOv5Detector(
            weights_path=weights_path_chosen, # original is: weights_path='yolo5s'
            img_size=320,
            conf_threshold=CONF_THRESHOLD,
            width=image_resolution[0],
            height=image_resolution[1],
            output_dir="code/static",
            horizontal_fov=camera_fov[0],
            vertical_fov=camera_fov[1]
        )

        # Create RadarDisplay and start the processing loop in a separate thread
        radar_display = RadarDisplay([])

        # Start the processing loop in a separate thread
        processing_thread = threading.Thread(target=processing_loop, args=(detector, usb_camera, radar_display, camera_indices))
        processing_thread.daemon = True  # Daemonize the thread so it exits when the main program exits
        processing_thread.start()

        # Run the Tkinter mainloop in the main thread
        radar_display.run()
    except Exception as e:
        logging.critical("Critical error in application", exc_info=True)