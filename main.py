import logging
import time
import cv2
import numpy as np
import os
from datetime import datetime
from OOP_webcam import CameraSystem
from OOP_yolo_detector import YOLOv5Detector

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

def find_maximum_orb_result_pairs(matrix):
    max_pairs = []
    used_indices = set()  # Set to track used (row, col) pairs

    # Loop through each row in the matrix
    for i in range(len(matrix)):
        max_value = None
        max_index = None

        # Loop through each element in the row
        for j in range(len(matrix[i])):
            _, _, details = matrix[i][j]
            orb_result = details['orb_result']

            # Check if current orb_result is the max and not yet used
            if (i, j) not in used_indices and (max_value is None or orb_result > max_value):
                max_value = orb_result
                max_index = (i, j)

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
#image_resolution = [640, 480]
image_resolution = [1920, 1080]
camera_fov = [62.2, 48.8]
f = 3.04 # focal length in [mm]
SAVE_IMAGES = True
TAKE_NEW_IMAGES = False

if __name__ == "__main__":
    min_matches = 20 
    usb_camera = CameraSystem(cameras_distance=0.235, max_cameras=10)
    #camera_indices = usb_camera.list_available_cameras()
    camera_indices = [1,0] # [right index, left index]

    detector = YOLOv5Detector(
        model_name='yolov5s',
        img_size=320,
        conf_threshold=0.75,
        width=image_resolution[0],
        height=image_resolution[1],
        output_dir="code\static",
        horizontal_fov=camera_fov[0],
        vertical_fov=camera_fov[1]
    )

    while True:
        print("Processing frames...")
        if TAKE_NEW_IMAGES == True:
            #in case you want to take every time new images, allow the line below
            frames = usb_camera.capture_images_from_cameras(camera_indices,save_path="code\static\images", width=image_resolution[0], height=image_resolution[1])
        else:
            # Paths to your saved images
            #left_image_path = "code\images\RIGHT_image_1_20241030_145532.jpg"
            #right_image_path = "code\images\LEFT_image_1_20241030_145532.jpg"
            left_image_path = "code\images\RIGHT_image_1_20241030_170144.jpg"
            right_image_path = "code\images\LEFT_image_1_20241030_170144.jpg"

            # Load images from disk
            frame_left = cv2.imread(left_image_path)
            frame_right = cv2.imread(right_image_path)

            # Store loaded images in frames dictionary as if they were captured from cameras
            frames = {
                1: frame_left,  # Assuming 1 represents the left camera
                0: frame_right  # Assuming 0 represents the right camera
            }
            
        for camera_index, frame in frames.items():
            print(f"Running detection on camera {camera_index}")
            if camera_index == 1:
                results_r, frame_r = detector.detect_and_describe(frame=frame, save_images=SAVE_IMAGES, camera_index=camera_index)
                detections_r = detector.process_detections(results_r, frame_r, image_resolution[0], image_resolution[1], camera_fov[0], camera_fov[1])
                detector.print_detections(detections_r)
            else:
                results_l, frame_l = detector.detect_and_describe(frame=frame, save_images=SAVE_IMAGES, camera_index=camera_index)
                detections_l = detector.process_detections(results_l, frame_l, image_resolution[0], image_resolution[1], camera_fov[0], camera_fov[1])
                detector.print_detections(detections_l)
        time.sleep(1)  # Add a small delay if needed

        if detections_r and detections_l: 
            matched_objects = []
            correlation_results_matrix = np.zeros((len(detections_l), len(detections_r)), dtype=object)

            for r, obj_r in enumerate(detections_r):
                for l, obj_l in enumerate(detections_l):
                    cropped_img_r = obj_r['cropped_image']
                    cropped_img_l = obj_l['cropped_image']

                    if cropped_img_r is None or cropped_img_l is None:
                        print(f"Error: Cropped images are invalid for object {r}. Skipping comparison.")
                        continue

                    if cropped_img_r.size == 0 or cropped_img_l.size == 0:
                        print(f"Error: One of the cropped images has zero size for object {r}. Skipping comparison.")
                        continue

                    cropped_img_r_resized, cropped_img_l_resized = resize_images_to_same_size(cropped_img_r, cropped_img_l)
                    is_similar, orb_result = compare_orb(cropped_img_r_resized, cropped_img_l_resized)

                    if is_similar:
                        #print(f"Object {r} in right camera matches object {l} in left camera with {orb_result} matches.")

                        matched_object_data = {
                            'object_id_r': r,
                            'object_id_l': l,
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

            print('\n-----------------------------------')
            #print(correlation_results_matrix)
            print('\n-----------------------------------')

        # Call the function with the data
        result = find_maximum_orb_result_pairs(correlation_results_matrix)

        # Display the result
        print("Maximum orb_result pairs and their values:", result)
        for i in range(len(result)):
            matched_object = correlation_results_matrix[result[i][0]][2]
            
            # if you want to calculate with our furmula
            # right_aov = matched_object['right_image_aov']
            # left_aov = matched_object['left_image_aov']
            # distance_to_object = usb_camera.calculate_distance(right_aov, left_aov)

            # if you want to calculate with niconielsen furmula            
            right_center_object = matched_object['object_center_right']
            left_center_object = matched_object['object_center_left']
            distance_to_object = usb_camera.calculate_distance_niconielsen32(right_center_object, left_center_object, frame_right, frame_left, usb_camera.cameras_distance, f, camera_fov[0])
            print(f"distance to object is: {distance_to_object}")
