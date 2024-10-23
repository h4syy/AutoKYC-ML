from fastapi import Request, UploadFile, HTTPException
import os
from services.inference_service import run_inference

async def handle_inference(file: UploadFile):
    print("Inside handle_inference")
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save the uploaded file
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    file_path = os.path.join(uploads_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

  # Run inference using the service and get raw results
    raw_results = await run_inference(file_path)
    
    print("Raw inference results:", raw_results)

    # Check if 'raw_detections' key exists
    if 'raw_detections' not in raw_results:
        raise HTTPException(status_code=500, detail="Inference service did not return raw detections.")

    # Parse the raw results into the desired format
    detections = []
    for raw_detection in raw_results['raw_detections']:
        boxes = raw_detection.get("boxes", [])
        confidences = raw_detection.get("confidences", [])
        classes = raw_detection.get("classes", [])

        # Iterate over detections and format them
        for i in range(len(boxes)):
            detection = {
                "predicted_class": classes[i],  # Assuming this is the class index, can map to names if necessary
                "confidence_score": confidences[i],
                "bounding_box": boxes[i]
            }
            detections.append(detection)

    print("Parsed results from inference:", detections)

    # Optionally, remove the file after inference
    os.remove(file_path)

    # Return the parsed results
    return {
        "message": "YOLO inference completed successfully",
        "detections": detections
    }