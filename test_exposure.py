import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def adjust_exposure(camera, target_brightness=128, tolerance=10, max_iterations=10):
    """
    מכוון את החשיפה של המצלמה באופן דינמי לפי הבהירות הממוצעת של התמונה.
    
    :param camera: אובייקט מצלמה (cv2.VideoCapture)
    :param target_brightness: ערך בהירות ממוצע רצוי (0-255)
    :param tolerance: טולרנס לערך הבהירות הרצוי
    :param max_iterations: מספר מקסימלי של איטרציות
    """
    # אתחול ערך החשיפה הנוכחי
    exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
    brightness_values = []  # לאחסון הבהירות הממוצעת בכל שלב
    images = []  # לאחסון התמונות לכל אורך ההרצה
    
    for _ in range(max_iterations):
        # צלם תמונה
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        # המרה לגווני אפור
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # חישוב בהירות ממוצעת
        avg_brightness = np.mean(gray_frame)
        brightness_values.append(avg_brightness)
        images.append(frame)
        
        print(f"Current brightness: {avg_brightness}")

        # בדיקה אם אנחנו בטווח הרצוי
        if abs(avg_brightness - target_brightness) <= tolerance:
            print("Brightness is within the desired range.")
            break

        # עדכון החשיפה
        if avg_brightness < target_brightness:
            # בהירות נמוכה מדי - הגדל חשיפה
            exposure += 0.5
        else:
            # בהירות גבוהה מדי - הקטן חשיפה
            exposure -= 0.5

        # עדכון ערך החשיפה במצלמה
        camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print(f"Adjusting exposure to: {exposure}")
        time.sleep(1)
    
    return exposure, brightness_values, images


def create_collage(images, cols=5):
    """
    יוצר קולאז' של תמונות.
    
    :param images: רשימה של תמונות
    :param cols: מספר עמודות בקולאז'
    :return: תמונת קולאז'
    """
    rows = (len(images) + cols - 1) // cols  # חישוב מספר השורות
    height, width, _ = images[0].shape  # גודל תמונה אחת
    collage = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        collage[row * height:(row + 1) * height, col * width:(col + 1) * width] = image

    return collage


# פתח את המצלמה
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# הגדר למצב חשיפה ידני
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

# כוון את החשיפה
final_exposure, brightness_values, images = adjust_exposure(cap, target_brightness=128, tolerance=10)

print(f"Final exposure value: {final_exposure}")

# גרף הבהירות
plt.plot(brightness_values, marker='o')
plt.title("Average Brightness Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Brightness")
plt.grid()
plt.show()

# יצירת קולאז'
collage = create_collage(images)

# הצגת הקולאז'
cv2.imshow("Collage", collage)
cv2.waitKey(0)

# שחרור וסגירת מצלמה
cap.release()
cv2.destroyAllWindows()
