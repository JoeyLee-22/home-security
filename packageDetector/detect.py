from ultralytics import YOLO

def detect_package(img):
    model = YOLO("best.pt")
    results = model(img, stream=True)
    return results