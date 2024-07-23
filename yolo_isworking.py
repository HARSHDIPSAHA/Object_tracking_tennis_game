from ultralytics import YOLO

model=YOLO("model/yolo5_last.pt")
result=model.predict("input_data/input_video.mp4",conf=0.4,save=True)


for box in result[0].boxes:
    print(box)