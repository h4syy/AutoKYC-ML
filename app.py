from fastapi import FastAPI, UploadFile, File,Form, HTTPException, status, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import shutil
import os
from ultralytics import YOLO
import boto3
import json
import dbconfig
import asyncio
import logging
import aiomysql

# Define logger configuration
class MySQLHandler(logging.Handler):
    def __init__(self, db_config):
        super().__init__()
        self.db_config = db_config

    def emit(self, record):
        # Run the asynchronous emit logic in the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._async_emit(record))
        else:
            loop.run_until_complete(self._async_emit(record))

    async def _async_emit(self, record):
        log_entry = self.format(record)
        try:
            conn = await aiomysql.connect(**self.db_config)
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "INSERT INTO logs (level, message) VALUES (%s, %s)",
                    (record.levelname, log_entry)
                )
                await conn.commit()
            conn.close()
        except Exception as e:
            print(f"Failed to log to database: {e}")

# Logger setup
logger_db_config = {
    "host": "localhost",
    "user": "root",
    "password": "12345",
    "db": "autokyc",
    "port": 3306
}
logger = logging.getLogger("db_logger")
logger.setLevel(logging.INFO)
logger_handler = MySQLHandler(logger_db_config)
logger_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(logger_handler)

app = FastAPI()
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete.")
    await dbconfig.init_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    await dbconfig.close_db_pool()

model = YOLO('best.pt')

output_dir = '/output'
os.makedirs(output_dir, exist_ok=True)

#datatypes configuration
class Detection(BaseModel):
    session_id: str
    csid: str
    predicted_class: str
    document_photo_path: str
    bounding_box: str
    confidence: float
    details: dict
    msisdn: int

class DetectionResponse(BaseModel):
    detections: List[Detection]
 
async def insert_detections_into_db(detections: List[Detection]):
    async with dbconfig.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            insert_query = """
            INSERT INTO documentdetection (SessionId, CSID, PredictedClass, DocumentPhotopath, BoundingBox, Confidence, Details, MSISDN)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            for detection in detections:
                await cursor.execute(insert_query, (
                    detection.session_id,
                    detection.csid,
                    detection.predicted_class,
                    detection.document_photo_path,
                    detection.bounding_box,
                    detection.confidence,
                    json.dumps(detection.details),  # Convert dict to JSON string for `Details` column
                    detection.msisdn
                ))
            await conn.commit()
            logger.info("Detections inserted into the database.")

@app.post("/document-detection/inference", response_model=DetectionResponse)
async def detect_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    csid: str = Form(...),
    msisdn: int = Form(...)
):
    
    max_file_size = 10 * 1024 * 1024  # 10 MB
    # Check file size
    file_size = len(await file.read())
    if file_size > max_file_size:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 10MB.")
    
    logger.info("Document detection inference started.")
    temp_image_path = os.path.join(output_dir, file.filename)
    
    try:
        # Save the uploaded file to a temporary location
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {temp_image_path}.")

        # Perform model inference
        results = model.predict(source=temp_image_path, save=True, save_txt=True, save_conf=True)
        detections = []

        # Parse the results from the model
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy().tolist()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().tolist()  # Class IDs from the model
            class_names = [model.names[int(cls)] for cls in class_ids]  # Class names

            for box, confidence, class_name in zip(boxes, confidences, class_names):
                detections.append(Detection(
                    session_id=session_id,
                    csid=csid,
                    predicted_class=class_name,
                    document_photo_path=temp_image_path,
                    bounding_box=str(box),
                    confidence=confidence,
                    details={},
                    msisdn=msisdn
                ))
        
        # Insert detections into the database
        await insert_detections_into_db(detections)

        logger.info("Document detection inference completed.")
        return DetectionResponse(detections=detections)

    except Exception as e:
        logger.error(f"Error during document detection: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during document detection.")
    
    finally:
        # Ensure the temporary image file is removed
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            logger.info(f"Temporary file {temp_image_path} removed.")

# Initialize Rekognition client using boto3
rekognition_client = boto3.client(
    'rekognition',
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
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
    logger.info("Face comparison inference started.")
    temp_front_document = os.path.join(output_dir, document_front.filename)
    temp_liveness_path = os.path.join(output_dir, liveness_document.filename)
    with open(temp_front_document, "wb") as buffer:
        shutil.copyfileobj(document_front.file, buffer)
    with open(temp_liveness_path, "wb") as buffer:
        shutil.copyfileobj(liveness_document.file, buffer)

    try:
        result = await run_face_comparison(temp_front_document, temp_liveness_path)

        session_id = result.session_id
        msisdn = result.msisdn
        face_matches = result.face_matches

        for match in face_matches:
            confidence = match.similarity
            similarity = match.similarity
            details = match.dict()

            await insert_face_compare_result(
                session_id=session_id,
                csid='CS987654321',
                confidence=confidence,
                similarity=similarity,
                details=details,
                msisdn=int(msisdn)
            )

        logger.info("Face comparison completed.")
    except Exception as e:
        logger.error(f"Error during face comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_front_document)
        os.remove(temp_liveness_path)
        logger.info("Temp images removed.")

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
        logger.error(f"Error during face comparison: {e}")
        raise HTTPException(status_code=500, detail="Face comparison failed.")
    
async def insert_liveness_result(session_id, csid, liveness_photopath, bounding_box, confidence, status, msisdn, details):
    print(type(session_id), type(csid), type(liveness_photopath), type(bounding_box),type((confidence)),type(status), type(msisdn), type((details)) )
    query = """
    INSERT INTO liveness (SessionId, CreatedDate, CSID, LivenessPhotopath, BoundingBox, Confidence, Status, MSISDN, Details)
    VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s)
    """
    async with dbconfig.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (session_id, csid, liveness_photopath, bounding_box, (confidence), status, msisdn, details))
            await conn.commit()

@app.post("/liveness/post-data")
async def post_data(request: Request):
    # log request body data
    request_data = await request.json()
    print(f"Received request data: {request.body}")
    session_id = request_data.get("SessionId")
    csid = request_data.get("CSID")
    liveness_photopath = request_data.get("AuditImages", [{}])[0].get("S3Object", {}).get("Name")
    bounding_box = json.dumps(request_data.get("AuditImages", [{}])[0].get("BoundingBox"))
    confidence = request_data.get("Confidence")
    msisdn = request_data.get("MSISDN")
    status = request_data.get("Status")
    referenceImg = request_data.get("ReferenceImage")
    auditImages = request_data.get("AuditImages")
    liveness_data = {
        "SessionId": session_id,
        "MSISDN": msisdn,
        "CreatedDate": datetime.now(),
        "Confidence": confidence,
        "BoundingBox": bounding_box,
        "CSID": csid,
        "Status": status,
        "LivenessPhotopath": liveness_photopath,
        "Details": json.dumps({
            "ReferenceImage": referenceImg,
            "AuditImages": auditImages
        })
    }
    await insert_liveness_result(session_id,csid,liveness_photopath,bounding_box,float(confidence),status, msisdn, liveness_data["Details"])
    return {"message": "Data processed successfully", "data": liveness_data}

# Run the app with uvicorn if executing directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
