
from ultralytics import YOLOv10

#import torch
#import onnx


# Load a pre-trained YOLOv8 model
model = YOLOv10("yolov10n.pt")
#model = model.export(format="onnx")

# Specify the source image
source = "./Torres.jpg"
#model = YOYOLOv10LO("yolov8n.onnx")


results = model.predict(source="https://www.youtube.com/watch?v=DTZnuolwxj8",name="Test_result",project="/var/proyectos/YoloWorldProyect/Test_results", save=True)
print(results)
# Make predictions
#results = model.predict(source,name="Test_result",project="/var/proyectos/YoloWorldTest/volumeTest/examples", save=True, imgsz=320, conf=0.5,classes=[0])

# Extract bounding box dimensions
#print(results)
#boxes = results[0].boxes.xywh.cpu()
#for box in boxes:
 #   x, y, w, h = box
  #  print(f"Width of Box: {w}, Height of Box: {h}")