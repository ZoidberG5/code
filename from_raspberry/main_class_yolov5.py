import logging
import time
import cv2
import numpy as np
import os
from datetime import datetime
from usb_camera.usb_webcam_OOP import CameraSystem
from yolov5_obj_detection.yolov5_detector import YOLOv5Detector

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

    # save the new cropped images
    output_filename = os.path.join("/home/admin/GIT/naggles/static/images", "cropped_image0.jpg")
    cv2.imwrite(output_filename, image1)
    output_filename = os.path.join("/home/admin/GIT/naggles/static/images", "cropped_image2.jpg")
    cv2.imwrite(output_filename, image2)

    # save the new resized images
    output_filename = os.path.join("/home/admin/GIT/naggles/static/images", "resized_image0.jpg")
    cv2.imwrite(output_filename, resized_image1)
    output_filename = os.path.join("/home/admin/GIT/naggles/static/images", "resized_image2.jpg")
    cv2.imwrite(output_filename, resized_image2)

    return resized_image1, resized_image2

# Set up logging configuration
logging.basicConfig(filename='detection_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Camera and resolution configuration
image_resolution = [640, 480]
camera_fov = [60, 47]
SAVE_IMAGES = True

if __name__ == "__main__":
    min_matches = 20 
    usb_camera = CameraSystem(cameras_distance=0.235, max_cameras=10)
    camera_indices = usb_camera.list_available_cameras()

    detector = YOLOv5Detector(
        model_name='yolov5n',
        img_size=320,
        conf_threshold=0.6,
        width=image_resolution[0],
        height=image_resolution[1],
        output_dir="/home/admin/GIT/naggles/static/images",
        horizontal_fov=camera_fov[0],
        vertical_fov=camera_fov[1]
    )

    if len(camera_indices) == 4:
        try:
            while True:
                for camera_index in camera_indices:
                    if camera_index == 0:
                        results_r, frame_r = detector.detect_and_describe(source=camera_index, save_images=SAVE_IMAGES)
                        detections_r = detector.process_detections(results_r, frame_r, image_resolution[0], image_resolution[1], camera_fov[0], camera_fov[1])
                    else:
                        results_l, frame_l = detector.detect_and_describe(source=camera_index, save_images=SAVE_IMAGES)
                        detections_l = detector.process_detections(results_l, frame_l, image_resolution[0], image_resolution[1], camera_fov[0], camera_fov[1])

                if detections_r and detections_l: 
                    matched_objects = []
                    correlation_results_matrix = np.zeros((len(detections_l), len(detections_r)), dtype=object)

                    for i, obj_r in enumerate(detections_r):
                        for j, obj_l in enumerate(detections_l):
                            cropped_img_r = obj_r['cropped_image']
                            cropped_img_l = obj_l['cropped_image']

                            if cropped_img_r is None or cropped_img_l is None:
                                print(f"Error: Cropped images are invalid for object {i}. Skipping comparison.")
                                continue

                            if cropped_img_r.size == 0 or cropped_img_l.size == 0:
                                print(f"Error: One of the cropped images has zero size for object {i}. Skipping comparison.")
                                continue

                            cropped_img_r_resized, cropped_img_l_resized = resize_images_to_same_size(cropped_img_r, cropped_img_l)
                            is_similar, orb_result = compare_orb(cropped_img_r_resized, cropped_img_l_resized)

                            if is_similar:
                                #print(f"Object {i} in right camera matches object {j} in left camera with {orb_result} matches.")

                                matched_object_data = {
                                    'object_id_i': i,
                                    'object_id_j': j,
                                    'object_name_right': obj_r['object'],
                                    'object_name_left': obj_l['object'],
                                    'right_image_aov': obj_r['angle_of_view'],
                                    'left_image_aov': obj_l['angle_of_view'],
                                    'orb_result': orb_result
                                }
                                matched_objects.append(matched_object_data)

                            correlation_results_matrix[j, i] = (is_similar, orb_result, matched_object_data)

                    print('\n-----------------------------------')
                    print(correlation_results_matrix)
                    print('\n-----------------------------------')

                    #this part finds the best matches
                    #it's not working well at all and needed to deal with it!!!!!
                    #אולי הדרך הטובה ביותר היא לעבוד על מטריצה שמכילה רק את התוצאות ולמצוא את הזוגות הטובים ביותר שם
                    #אחרי שנמצאו הזוגות הטובים ביותר, הולכים לפי האינדקסים של הזוגות ומהם מוצאים את האיידי על מנת לחשב את המרחק
                    print("\nMatched objects:")
                    max_indices_per_row = []

                    for j in range(correlation_results_matrix.shape[0]):
                        row_values = [
                            (correlation_results_matrix[j, i][1], i) 
                            for i in range(correlation_results_matrix.shape[1])
                            if correlation_results_matrix[j, i][2]['object_name_right'] == correlation_results_matrix[j, i][2]['object_name_left']
                        ]

                        if row_values:
                            max_value, max_index = max(row_values, key=lambda x: x[0])
                            max_indices_per_row.append(max_index)

                            matched_object = correlation_results_matrix[j, max_index][2]
                            log_message = (
                                f"Matched object at row {j}, column {max_index}: "
                                f"Object_i {matched_object['object_id_i']} ({matched_object['object_name_right']}) ({matched_object['object_name_left']}) - "
                                f"Right AOV: {matched_object['right_image_aov']}, "
                                f"Left AOV: {matched_object['left_image_aov']}, "
                                f"ORB Matches: {matched_object['orb_result']}"
                            )
                            print(log_message)
                            logging.info(log_message)
                        else:
                            print(f"No valid matched object with the same name for row {j}")

                    matched_indices = set()

                    for j, max_index in enumerate(max_indices_per_row):
                        if max_index not in matched_indices:
                            matched_indices.add(max_index)

                            matched_object = correlation_results_matrix[j, max_index][2]

                            right_aov = matched_object['right_image_aov']
                            left_aov = matched_object['left_image_aov']
                            distance_to_object = usb_camera.calculate_distance(right_aov, left_aov)

                            distance_log = (
                                f"Distance to object at row {j} (left object {matched_object['object_id_j']}) "
                                f"matching with (right object {matched_object['object_id_i']}): "
                                f"{distance_to_object:.2f} meters"
                            )
                            print(distance_log)
                            logging.info(distance_log)

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                print(f"Current time: {current_time}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Detection stopped by user.")
            detector.clean_up()

    else:
        print("There is no 2 cameras connected!!!")
