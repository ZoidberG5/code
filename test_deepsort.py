import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# טעינת מודל YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# הגדרת DeepSORT
object_tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3)

# פתיחת קובץ וידאו או מצלמת רשת
cap = cv2.VideoCapture(0)  # או 0 למצלמת רשת

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # זיהוי אובייקטים בפריים
    results = model(frame)

    # הכנת רשימת הזיהויים עבור DeepSORT
    detections = []
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, int(cls)))

    # עדכון המעקב
    tracks = object_tracker.update_tracks(detections, frame=frame)

    # ציור התוצאות על הפריים
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # הצגת הפריים
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
