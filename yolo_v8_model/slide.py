import cv2
import torch
from ultralytics import YOLO

def sliding_window(image, size=640, stride=512):
    h, w = image.shape[:2]
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            yield (x, y, image[y:y + size, x:x + size])

def run_sliding_window_inference(image_path, model_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    full_detections = []

    for (x_offset, y_offset, crop) in sliding_window(image, size=160, stride=120):
        results = model(crop, verbose=False)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            x1 += x_offset
            x2 += x_offset
            y1 += y_offset
            y2 += y_offset
            full_detections.append((x1, y1, x2, y2))

    # Draw detections
    for (x1, y1, x2, y2) in full_detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite("detected.jpg", image)
    print("Saved: detected.jpg")

run_sliding_window_inference("../data/Cliphunter.jpg", "runs/detect/train4/weights/best.pt")  # your model path here