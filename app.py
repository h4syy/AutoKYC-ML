from fastapi import FastAPI, UploadFile, File, HTTPException,status
from pydantic import BaseModel
from typing import List, Dict, Any
import shutil
import os
from ultralytics import YOLO
import boto3

app = FastAPI()

# Load the trained YOLO model
model = YOLO('best.pt')

output_dir = 'D:/ml backend for autokyc/ML-BACKEND/AutoKYC-ML/output'
os.makedirs(output_dir, exist_ok=True)

class Detection(BaseModel):
    classification: str
    bounding_box: List[float]
    confidence_score: float
    msisdn: str
    session_id: str

class DetectionResponse(BaseModel):
    detections: List[Detection]

@app.post("/document-detection/inference", response_model=DetectionResponse)
async def detect_document(file: UploadFile = File(...)):
    temp_image_path = os.path.join(output_dir, file.filename)
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(source=temp_image_path, save=True, save_txt=True, save_conf=True)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().tolist()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy().tolist()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().tolist()  # Class IDs from the model
        class_names = [model.names[int(cls)] for cls in class_ids]  # Class names

        for box, confidence, class_name in zip(boxes, confidences, class_names):
            detections.append(Detection(
                bounding_box=box,
                confidence_score=confidence,
                classification=class_name,
                msisdn = '1234567890',
                session_id= '89bb23de-c331-4cae-bcb3-babb55ebcbfe'
            ))

    # # Optionally, delete the temporary image
    # os.remove(temp_image_path)

    return DetectionResponse(detections=detections)

AWS_ACCESS_KEY_ID = "AKIAZQ3DTJ4GVMWRSCVR" 
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "us-east-1"

# Initialize Rekognition client using boto3
rekognition_client = boto3.client(
    'rekognition',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

class FaceComparisonResult(BaseModel):
    similarity: float
    bounding_box: Dict[str, float]

class FaceComparisonResponse(BaseModel):
    source_image_bounding_box: Dict[str, float]
    face_matches: List[FaceComparisonResult]
    unmatched_faces: List[Dict[str, Any]]
    msisdn: str
    session_id: str

@app.post("/face/face-compare", response_model=FaceComparisonResponse)
async def face_compare(document_front: UploadFile = File(...), liveness_document: UploadFile = File(...)):
    # Define temporary file paths
    temp_front_document = os.path.join(output_dir, document_front.filename)
    temp_liveness_path = os.path.join(output_dir, liveness_document.filename)
    
    
    
    with open(temp_front_document, "wb") as buffer:
        shutil.copyfileobj(document_front.file, buffer)
    with open(temp_liveness_path, "wb") as buffer:
        shutil.copyfileobj(liveness_document.file, buffer)

    # Perform face comparison
    try:
        result = await run_face_comparison(temp_front_document, temp_liveness_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        os.remove(temp_front_document)
        os.remove(temp_liveness_path)

    return result

async def run_face_comparison(source_image_path: str, target_image_path: str):

    if not (os.path.exists(source_image_path) and os.path.getsize(source_image_path) > 0):
        raise HTTPException(status_code=400, detail="Source image file is missing or empty")
    if not (os.path.exists(target_image_path) and os.path.getsize(target_image_path) > 0):
        raise HTTPException(status_code=400, detail="Target image file is missing or empty")

    with open(source_image_path, 'rb') as source_image:
        source_bytes = source_image.read()
    with open(target_image_path, 'rb') as target_image:
        target_bytes = target_image.read()

    if not source_bytes or not target_bytes:
        raise HTTPException(status_code=400, detail="One or both image files are empty")
    
    try:
        response = rekognition_client.compare_faces(
            SourceImage={'Bytes': source_bytes},
            TargetImage={'Bytes': target_bytes},
            SimilarityThreshold=0
        )
        
        face_matches = response.get('FaceMatches', [])
        unmatched_faces = response.get('UnmatchedFaces', [])
        source_bounding_box = response.get('SourceImageFace', {}).get('BoundingBox', {})
        msisdn = '1234567890'
        session_id = '89bb23de-c331-4cae-bcb3-babb55ebcbfe'
        
        results = {
            "source_image_bounding_box": source_bounding_box,
            "face_matches": [],
            "unmatched_faces": unmatched_faces,
            "msisdn" : msisdn,
            "session_id": session_id
        }

        for match in face_matches:
            similarity = match['Similarity']
            bounding_box = match['Face']['BoundingBox']
            results['face_matches'].append({
                "similarity": similarity,
                "bounding_box": bounding_box,
            })

        return FaceComparisonResponse(**results)

    except Exception as e:
        print(f"Error during face comparison: {e}")
        raise Exception("Face comparison failed.")
    
# Run the app with uvicorn if executing directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)