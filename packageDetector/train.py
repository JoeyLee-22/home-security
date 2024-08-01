from ultralytics import YOLO
from roboflow import Roboflow

f = open("key.txt", "r")
rf = Roboflow(api_key=f.read())
f.close()
project = rf.workspace("homesecurity-upzel").project("package-human-detection")
version = project.version(3)
dataset = version.download("yolov8")

model = YOLO("yolov8n.pt")
model.train(data="package&human-detection-3/data.yaml", epochs=50, device="mps", plots=True, verbose=True)