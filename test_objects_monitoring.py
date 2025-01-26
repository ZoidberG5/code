import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# this is a function that matches objects between two rounds of detections
def match_objects1(previous_detections, current_detections):
    matched = {}
    for i, prev_obj in enumerate(previous_detections):
        best_match = None
        best_distance = float('inf')
        for j, curr_obj in enumerate(current_detections):
            distance = np.linalg.norm(
                [prev_obj['distance'] - curr_obj['distance'],
                 prev_obj['angle_x'] - curr_obj['angle_x'],
                 prev_obj['angle_y'] - curr_obj['angle_y']]
            )
            if distance < best_distance:
                best_distance = distance
                best_match = j
        if best_match is not None:
            matched[i] = best_match
    return matched


# דוגמה לנתוני הסיבובים (כפי שסיפקת)
monitoring_rounds_of_detections = [
    (0, [{'name': 'car', 'distance': 5.21, 'angle_x': -26.08, 'angle_y': -2.08}, {'name': 'car', 'distance': 7.65, 'angle_x': 2.27, 'angle_y': -2.48}, {'name': 'car', 'distance': 5.31, 'angle_x': -18.84, 'angle_y': -2.28}, {'name': 'person', 'distance': 9.86, 'angle_x': 9.48, 'angle_y': -3.98}]), 
    (1, [{'name': 'car', 'distance': 3.63, 'angle_x': -26.05, 'angle_y': -2.15}, {'name': 'car', 'distance': 8.43, 'angle_x': 2.22, 'angle_y': -2.42}, {'name': 'car', 'distance': 4.96, 'angle_x': -18.85, 'angle_y': -2.35}, {'name': 'person', 'distance': 10.74, 'angle_x': 9.46, 'angle_y': -4.07}]), 
    (2, [{'name': 'car', 'distance': 4.45, 'angle_x': -26.09, 'angle_y': -2.1}, {'name': 'car', 'distance': 8.61, 'angle_x': 2.17, 'angle_y': -2.44}, {'name': 'car', 'distance': 4.98, 'angle_x': -18.79, 'angle_y': -2.4}, {'name': 'person', 'distance': 11.42, 'angle_x': 9.39, 'angle_y': -4.13}]), 
    (3, [{'name': 'car', 'distance': 4.45, 'angle_x': -26.09, 'angle_y': -2.08}, {'name': 'car', 'distance': 8.26, 'angle_x': 2.14, 'angle_y': -2.44}, {'name': 'car', 'distance': 4.97, 'angle_x': -18.82, 'angle_y': -2.4}]), 
    (4, [{'name': 'car', 'distance': 4.06, 'angle_x': -26.09, 'angle_y': -2.12}, {'name': 'car', 'distance': 8.43, 'angle_x': 2.25, 'angle_y': -2.44}, {'name': 'car', 'distance': 5.2, 'angle_x': -18.82, 'angle_y': -2.35}, {'name': 'person', 'distance': 9.6, 'angle_x': 9.49, 'angle_y': -4.16}]), 
    (5, [{'name': 'car', 'distance': 6.14, 'angle_x': -26.16, 'angle_y': -2.15}, {'name': 'car', 'distance': 8.61, 'angle_x': 2.17, 'angle_y': -2.46}, {'name': 'car', 'distance': 4.97, 'angle_x': -18.82, 'angle_y': -2.35}, {'name': 'person', 'distance': 10.13, 'angle_x': 9.46, 'angle_y': -4.04}]), 
    (6, [{'name': 'car', 'distance': 5.77, 'angle_x': -26.14, 'angle_y': -2.15}, {'name': 'car', 'distance': 7.51, 'angle_x': 2.22, 'angle_y': -2.44}, {'name': 'car', 'distance': 4.97, 'angle_x': -18.82, 'angle_y': -2.35}, {'name': 'person', 'distance': 10.13, 'angle_x': 9.46, 'angle_y': -4.02}]), 
    (7, [{'name': 'car', 'distance': 5.21, 'angle_x': -26.08, 'angle_y': -2.12}, {'name': 'car', 'distance': 7.51, 'angle_x': 2.25, 'angle_y': -2.42}, {'name': 'car', 'distance': 4.78, 'angle_x': -18.79, 'angle_y': -2.37}, {'name': 'person', 'distance': 9.86, 'angle_x': 9.48, 'angle_y': -3.98}]), 
    (8, [{'name': 'car', 'distance': 7.27, 'angle_x': -26.03, 'angle_y': -2.19}, {'name': 'car', 'distance': 7.95, 'angle_x': 2.14, 'angle_y': -2.42}, {'name': 'car', 'distance': 4.58, 'angle_x': -18.82, 'angle_y': -2.46}, {'name': 'person', 'distance': 9.88, 'angle_x': 9.41, 'angle_y': -3.86}])]

# פונקציה להתאמת אובייקטים בין סבבים
# פונקציה זו משווה בין כל הזיהויים בסבב הקודם לכל הזיהויים בסבב הנוכחי
def match_objects(previous_detections, current_detections, max_distance=2.0, max_angle=5.0):
    matched = {}
    unmatched = set(range(len(current_detections)))
    
    for i, prev_obj in enumerate(previous_detections):
        best_match = None
        best_score = float('inf')
        
        for j in unmatched:
            curr_obj = current_detections[j]
            dist_diff = abs(prev_obj['distance'] - curr_obj['distance'])
            angle_diff = abs(prev_obj['angle_x'] - curr_obj['angle_x'])
            score = dist_diff + angle_diff  # משקל למרחק וזווית
            
            if score < best_score and dist_diff <= max_distance and angle_diff <= max_angle:
                best_score = score
                best_match = j
        
        if best_match is not None:
            matched[i] = best_match
            unmatched.remove(best_match)
    
    return matched, unmatched

# זיהוי ומעקב אחרי אובייקטים
tracked_objects = {}
object_id_counter = 0

for round_num, detections in monitoring_rounds_of_detections:
    if round_num == 0:
        # סבב ראשון - הוספה של כל האובייקטים
        for obj in detections:
            tracked_objects[object_id_counter] = {'name': obj['name'], 'rounds': [round_num], 'distances': [obj['distance']], 'angles': [obj['angle_x']]}
            object_id_counter += 1
    else:
        # סבבים נוספים - התאמה בין זיהויים
        previous_detections = [{'distance': tracked_objects[obj_id]['distances'][-1], 'angle_x': tracked_objects[obj_id]['angles'][-1]} for obj_id in tracked_objects]
        matched, unmatched = match_objects(previous_detections, detections)
        
        # עדכון אובייקטים קיימים
        for prev_idx, curr_idx in matched.items():
            obj_id = list(tracked_objects.keys())[prev_idx]
            tracked_objects[obj_id]['rounds'].append(round_num)
            tracked_objects[obj_id]['distances'].append(detections[curr_idx]['distance'])
            tracked_objects[obj_id]['angles'].append(detections[curr_idx]['angle_x'])
        
        # הוספת אובייקטים חדשים
        for curr_idx in unmatched:
            obj = detections[curr_idx]
            tracked_objects[object_id_counter] = {'name': obj['name'], 'rounds': [round_num], 'distances': [obj['distance']], 'angles': [obj['angle_x']]}
            object_id_counter += 1

# המרת זוויות לקואורדינטות פולריות
def polar_to_cartesian(distance, angle_x):
    angle_rad = np.radians(angle_x)  # המרת זווית לרדיאנים
    x = distance * np.cos(angle_rad)
    y = distance * np.sin(angle_rad)
    return x, y

# ציור גרפים נפרדים לכל אובייקט
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

for obj_id, data in tracked_objects.items():
    theta = np.radians(data['angles'])
    r = data['distances']
    ax.plot(theta, r, marker='o', label=f"Object {obj_id} ({data['name']})")

ax.set_title("Radar Map of Object Movements", va='bottom', fontsize=16)
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.show()

# הפיכת הנתונים ל-DataFrame
data = []
for round_num, detections in monitoring_rounds_of_detections:
    for detection in detections:
        data.append({'round': round_num, **detection})

df = pd.DataFrame(data)

# הצגת הנתונים
print(df)

# ניתוח דוגמה - ממוצע מרחקים לפי סוג אובייקט
print(df.groupby('name')['distance'].mean())
