import threading
import pygame
import cv2
from pygame.locals import *

pygame.init()
screen = pygame.display.set_mode((320, 240), 0, 32)
pygame.key.set_repeat(100)

def runFocus():
    temp_val = 512
    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    print('UP')
                    if temp_val < 1000:
                        temp_val += 10
                    print(f"Focus + : {temp_val}")
                elif event.key == K_DOWN:
                    print('DOWN')
                    if temp_val > 12:
                        temp_val -= 10
                    print(f"Focus - : {temp_val}")

def runCamera():
    cap = cv2.VideoCapture(2)  # Use default webcam
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    t1 = threading.Thread(target=runFocus)
    t1.daemon = True  # Updated daemon setting
    t1.start()

    runCamera()
