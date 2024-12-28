import torch

# load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/shai.y/Documents/Naggles_tmp-main/yolov5/runs/train/exp/weights/best.pt')
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# print the lables
labels = model.names
print("Labels in the model:")
for idx, label in enumerate(labels):
    print(f"{idx}: {label}")


import yaml
# טען את קובץ data.yaml
with open('C:/Users/shai.y/Documents/Naggles_tmp-main/datasets/University-Outdoor.v5i.yolov5pytorch/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
# הדפס את שמות המחלקות
labels = data['names']
print("Labels in the model:")
for idx, label in enumerate(labels):
    print(f"{idx}: {label}")

# עדכן את שמות המחלקות במודל מתוך קובץ data.yaml
model.names = data['names']
# הדפס שוב את המחלקות המעודכנות
print("Updated Labels in the model:")
for idx, label in enumerate(model.names):
    print(f"{idx}: {label}")

