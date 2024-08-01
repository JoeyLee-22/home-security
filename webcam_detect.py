import cv2
import math
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = ["face", "package"]

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
text_color = (255, 255, 255)
box_color = (27, 181, 16)
thickness = 1

model = YOLO("packageFaceDetector/detectBothV1.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            string = f"{classNames[cls]}: {confidence}"

            # text box
            text_size, _ = cv2.getTextSize(string, font, fontScale, thickness)
            text_w, text_h = text_size
            cv2.rectangle(img, (x1, y1), (x1 + text_w + 2, y1 - text_h - 4), box_color, -1)

            # text
            org = [x1+1, y1-4]
            cv2.putText(img, string, org, font, fontScale, text_color, thickness)

            # bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()