# Import the class
import cv2
from nudenet_detector import NudeNetDetector
from ultralytics import YOLO

# Use in your existing code
detector = NudeNetDetector(confidence_threshold=0.05, pixel_size=15, padding=5)
result = detector.process_image("data/Analvids.jpg", "output.jpg")

model = YOLO("yolo_v8_model/runs/detect/train15/weights/best.pt")
results = model("data/1383_9dc65c89_thumb.jpg")[0]

img = cv2.imread("data/1383_9dc65c89_thumb.jpg")
for box in results.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    roi = img[y1:y2, x1:x2]
    roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
    img[y1:y2, x1:x2] = roi_blur

cv2.imwrite("output.jpg", img)