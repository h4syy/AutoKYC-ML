import os
import sys
import json
from ultralytics import YOLO
import gc

model = YOLO('D:/ML-BACKEND/models/last.pt')
# model = YOLO('D:/ML-BACKEND/models/best.pt')

async def run_inference(input_image_path: str):
    print("log1")
    # Check if the file exists
    if not os.path.exists(input_image_path):
        return {"error": f"File {input_image_path} does not exist"}

    # Check the file extension
    if not input_image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        return {"error": "Unsupported file format. Please upload a JPEG or PNG image."}

    # Run inference on the single image
    results = model.predict(source=input_image_path, save=False, save_txt=False, save_conf=True, device='cpu')

      # Collect raw data (no parsing here, only raw result returned)
    raw_detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().tolist()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy().tolist()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().tolist()  # Class IDs from the model
        class_names = [model.names[int(cls)] for cls in class_ids]  # Get the class names using the model's `names` attribute

        raw_detections.append({
            "boxes": boxes,
            "confidences": confidences,
            "classes": class_names  # Use class names directly
        })
    # Return raw detections to controller
    return {"raw_detections": raw_detections}
