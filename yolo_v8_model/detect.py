import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train15/weights/best.pt")
results = model("../data/brazzers.jpg")[0]

img = cv2.imread("../data/brazzers.jpg")
for box in results.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    roi = img[y1:y2, x1:x2]
    roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
    img[y1:y2, x1:x2] = roi_blur

cv2.imwrite("blurred.jpg", img)