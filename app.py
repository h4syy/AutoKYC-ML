from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, HTTPException,status
from pydantic import BaseModel
from typing import List, Dict, Any
import shutil
import os
from ultralytics import YOLO
import boto3
import dbconfig
import json
import aiomysql

app = FastAPI()
@app.on_event("startup")
async def startup_event():
    await dbconfig.init_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    await dbconfig.close_db_pool()

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

    # Optionally, delete the temporary image
    os.remove(temp_image_path)

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



async def insert_face_compare_result(session_id, csid, confidence, similarity, details, msisdn):
    query = """
    INSERT INTO FaceCompare (SessionId, CreatedDate, CSID, Confidence, Similarity, Details, MSISDN)
    VALUES (%s, NOW(), %s, %s, %s, %s, %s)
    """
    async with dbconfig.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (session_id, csid, confidence, similarity, json.dumps(details), msisdn))
            await conn.commit()

@app.post("/face/face-compare", response_model=FaceComparisonResponse)
async def face_compare(document_front: UploadFile, liveness_document: UploadFile):

    temp_front_document = os.path.join(output_dir, document_front.filename)
    temp_liveness_path = os.path.join(output_dir, liveness_document.filename)

    with open(temp_front_document, "wb") as buffer:
        shutil.copyfileobj(document_front.file, buffer)
    with open(temp_liveness_path, "wb") as buffer:
        shutil.copyfileobj(liveness_document.file, buffer)

    # Perform face comparison
    try:
        result = await run_face_comparison(temp_front_document, temp_liveness_path)
        # Extract necessary data for Database insertion
        print("Result reached this point:", result)

        session_id = result.session_id
        msisdn = result.msisdn
        face_matches = result.face_matches

        for match in face_matches:
            confidence = match.similarity  # Corrected to use .similarity
            similarity = match.similarity  # Assuming 'confidence' and 'similarity' are the same here
            details = match.dict()  # Convert Pydantic model to dictionary for JSON serialization

            await insert_face_compare_result(
                session_id=session_id,
                csid=None,  # Adjust as needed, or remove if you won't use this field
                confidence=confidence,
                similarity=similarity,
                details=details,
                msisdn=int(msisdn)
            )

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
           "msisdn": msisdn,
           "session_id": session_id
       }

       for match in face_matches:
           similarity = match['Similarity']
           face_bounding_box = match['Face']['BoundingBox']

           bounding_box = {
               "width": face_bounding_box['Width'],
               "height": face_bounding_box['Height'],
               "left": face_bounding_box['Left'],
               "top": face_bounding_box['Top'],
           }

           face_match_result = FaceComparisonResult(
               similarity=similarity,
               bounding_box=bounding_box
           )
           
           results['face_matches'].append(face_match_result)

       return FaceComparisonResponse(**results)

    except Exception as e:
        print(f"Error during face comparison: {e}")
        raise HTTPException(status_code=500, detail="Face comparison failed.")
    
@app.post("/liveness/post-data")
async def post_data(request: Request):
    # log request body data
    request_data = await request.json()
    print(f"Received request data: {request.body}")
    liveness_data = {
        "SessionId": request_data.get("SessionId"),
        "MSISDN": request_data.get("MSISDN"),
        "CreatedDate": datetime.now(),
        "Confidence": request_data.get("Confidence"),
        # "BoundingBox": json.dumps(request_data.get("ReferenceImage", {}).get("BoundingBox")), // Not from ReferenceImage,from Audit Image
        "BoundingBox": json.dumps(request_data.get("AuditImages", [{}])[0].get("BoundingBox")),
        "CSID": request_data.get(""),
        "Status": request_data.get("Status"),
        # "LivenessPhotopath": request_data.get("ReferenceImage", {}).get("S3Object", {}).get("Name"),// Not from ReferenceImage,from Audit Image
        "LivenessPhotopath": request_data.get("AuditImages", [{}])[0].get("S3Object", {}).get("Name"),

        "Details": json.dumps({
            "ReferenceImage": request_data.get("ReferenceImage"),
            "AuditImages": request_data.get("AuditImages")
        })
    }
    return {"message": "Data processed successfully", "data": liveness_data}
    # return {"message": "Data received successfully"}


        
# Run the app with uvicorn if executing directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
