import cv2
import numpy as np
import matplotlib.pyplot as plt

# Camera and resolution configuration
image_resolution = [1920, 1080]  # Resolution for capturing images
camera_indices = [0, 1]  # Indices for left and right cameras
autofocus_values = [0, 1]  # 0: Disable autofocus, 1: Enable autofocus

# Helper Function to Calculate Average Color
def calculate_average_color(image, frame_name=""):
    if image is None:
        print(f"Error: {frame_name} image is None.")
        return None

    # Calculate the mean of each channel
    avg_color_bgr = image.mean(axis=0).mean(axis=0)  # BGR format
    avg_color_rgb = avg_color_bgr[::-1]  # Convert to RGB if needed
    print(f"Average color of {frame_name} frame (RGB): {avg_color_rgb}")
    return avg_color_bgr, avg_color_rgb

# Main Function
if __name__ == "__main__":
    avg_brightness_left = []
    avg_brightness_right = []
    # Initialize cameras
    cap_left = cv2.VideoCapture(camera_indices[0])
    cap_right = cv2.VideoCapture(camera_indices[1])

    # Verify cameras are opened
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: One or both cameras could not be opened.")
        exit(1)

    # Set resolution
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, image_resolution[0])
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, image_resolution[1])
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, image_resolution[0])
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, image_resolution[1])

    # Loop through autofocus settings
    for autofocus in autofocus_values:
        print(f"\nTesting autofocus: {'Enabled' if autofocus == 1 else 'Disabled'}")

        # Set autofocus for both cameras
        cap_left.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
        cap_right.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)

        # Capture images
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Error: Failed to capture images.")
            continue

        # Calculate average color for left and right frames
        avg_color_left_bgr, avg_color_left_rgb = calculate_average_color(frame_left, "Left")
        avg_color_right_bgr, avg_color_right_rgb = calculate_average_color(frame_right, "Right")

        # Display the captured frames
        cv2.imshow("Left Camera Frame", frame_left)
        cv2.imshow("Right Camera Frame", frame_right)

        # Save images (optional)
        save_path = "code/static/images/autofocus_tests"
        cv2.imwrite(f"{save_path}/left_image_autofocus_{autofocus}.jpg", frame_left)
        cv2.imwrite(f"{save_path}/right_image_autofocus_{autofocus}.jpg", frame_right)
        print(f"Images saved with autofocus {'enabled' if autofocus == 1 else 'disabled'} in {save_path}")

        avg_brightness_left.append(avg_color_left_rgb)
        avg_brightness_right.append(avg_color_right_rgb)

        # Wait for 1 second between autofocus tests
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('q'):
            print("Exiting loop...")
            break

    plt.plot(autofocus_values, [np.mean(brightness) for brightness in avg_brightness_left], label="Left Camera")
    plt.plot(autofocus_values, [np.mean(brightness) for brightness in avg_brightness_right], label="Right Camera")
    plt.xlabel("Autofocus (0: Disabled, 1: Enabled)")
    plt.ylabel("Average Brightness")
    plt.title("Autofocus vs. Average Brightness")
    plt.legend()
    plt.show()

    # Release resources
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
