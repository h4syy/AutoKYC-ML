import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from schema.schemas import Detection, DetectionResponse
from utilities.logger import logger
from database import dbconfig
from utilities.face_compare import face_compare_auto
import shutil
import os
import torch
import json
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from utilities.config import get_image_save_path_minio, client
# Path to the cloned YOLOv5 repository
#If you have linux (or deploying for linux) use:
from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

load_dotenv()

router = APIRouter()

MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# Paths to the YOLOv5 repository and model file
YOLO_REPO_DIR = Path("yolov5").resolve()  # YOLOv5 repository path
MODEL_PATH = Path("yolo/best.pt").resolve()  # Model file path

# Load the YOLOv5 model using the local repository
model = torch.hub.load(
    str(YOLO_REPO_DIR),  
    'custom',            
    path=str(MODEL_PATH),  
    source='local',       
    force_reload=False    
)

@router.post("/document-detection/inference/front")
async def detect_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    csid: str = Form(...),
    msisdn: int = Form(...)
):
    logger.info(f"Document detection inference started for msisdn_id: {msisdn}")
    
    # Generate the MinIO path for the file
    document_photo_path_front = get_image_save_path_minio(msisdn, session_id, "Id_front")
    
    try:
        # Upload the file to MinIO
        file_content = await file.read()

        # Save the uploaded file directly to MinIO
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            with open(tmp_filename, 'wb') as temp_file:
                temp_file.write(file_content)

        # Upload to MinIO using the MinIO client and the correct path
        client.fput_object(
            bucket_name=MINIO_BUCKET,  # Bucket name
            object_name=document_photo_path_front,  # Path in MinIO
            file_path=tmp_filename  # Local temporary file path
        )
        logger.info(f"File successfully uploaded to MinIO at: {document_photo_path_front}")

        # Retrieve the image from MinIO for inference
        response = client.get_object(MINIO_BUCKET, document_photo_path_front)
        
        # Save the retrieved image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            with open(tmp_filename, 'wb') as temp_file:
                temp_file.write(response.read())

        logger.info(f"Image successfully retrieved from MinIO: {document_photo_path_front}")

        # Run inference with YOLOv5 by passing the retrieved image file path
        results = model(tmp_filename) 
        if len(results.xyxy[0]) == 0:
            raise HTTPException(status_code=400, detail="No document detected.")

        detection_data = results.xyxy[0].tolist()[0]

        if len(detection_data) < 6: 
            raise HTTPException(status_code = 400, detail = "Invalid detection data format.")

        *box, confidence, cls = detection_data
        class_name = results.names[int(cls)]
        id_type_mapping = {
            "CS": 1,  # Citizenship
            "DL": 2,  # Driving License
            "NID": 3, # National ID
            "PP": 4   # Passport
        }
        suffix = class_name[-1]
        prefix = class_name[:-1] if suffix in {"F", "B"} else class_name
        id_type = id_type_mapping.get(prefix, -1)

        detection = Detection(
            session_id=session_id,
            csid=csid,
            id_type=id_type,
            predicted_class=class_name,
            document_photo_path=document_photo_path_front,
            bounding_box=str(box),
            confidence=confidence,
            details={},
            msisdn=msisdn
        )
        # Insert detections into DB
        dd_status_decoded = await insert_detections_into_db([detection])
        if id_type == 2:
            async with dbconfig.db_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.callproc('SP_FETCH_PHOTO_URL', (msisdn,))
                    paths_result = await cursor.fetchall()

                    if not paths_result or len(paths_result[0]) < 2:
                        raise HTTPException(status_code=500, detail="Failed to fetch required document paths.")
                    
                    liveness_document_path, document_front_path = paths_result[0]

                    # Normalize MinIO paths
                    document_front_path = os.path.normpath(document_front_path)
                    liveness_document_path = os.path.normpath(liveness_document_path)

            result = await face_compare_auto(
                document_front=document_front_path,
                liveness_document=liveness_document_path,
                session_id=session_id,
                csid=csid,
                msisdn=msisdn
            )
            logger.info(f"Face Compare result: {result}")  
            payload = {
                "ResponseData": {
                    "IsDocumentScanCompleted": True,
                    "IsVerified": False,
                    "IsBackDocumentNeed": False,
                    "DocumentType": id_type,
                },
                "ResponseCode": 300,
                "ResponseDescription": "Success"
            }
        elif dd_status_decoded == 0: 
            payload = {
                "ResponseData": {
                    "IsDocumentScanCompleted": False,
                    "IsVerified": False,
                    "IsBackDocumentNeed": True,
                    "DocumentType": id_type,
                },
                "ResponseCode": 200,
                "ResponseDescription": "Redirect to Front"
            }
        else:  
            payload = {
                "ResponseData": {
                    "IsDocumentScanCompleted": False,
                    "IsVerified": False,
                    "IsBackDocumentNeed": True,
                    "DocumentType": id_type,
                },
                "ResponseCode": 100,
                "ResponseDescription": "Success direct to DDB"
            }

        logger.info("Document detection inference completed successfully.")
        return payload
    
    except Exception as e:
        logger.error(f"Error during document detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(document_photo_path_front):
            logger.info(f"Saved {msisdn}_{document_photo_path_front}.")

async def insert_detections_into_db(detections: list[Detection]):
    async with dbconfig.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            for detection in detections:
                await cursor.callproc(
                    'SP_INSERT_DD',
                    (
                        detection.msisdn,
                        detection.session_id,
                        detection.csid,
                        detection.id_type,
                        detection.predicted_class,
                        detection.document_photo_path,
                        detection.bounding_box,
                        detection.confidence,
                        json.dumps(detection.details),
                        0
                    )
                )

                result = await cursor.fetchall()
                if result:
                    row = result[0]
                    msg = row[0]
                    dd_status = row[1]
                    sp_code = row[2]

                    dd_status_decoded = int.from_bytes(dd_status, byteorder='big')
                    logger.info(f"Stored procedure response dd_status: {dd_status}")

                else:
                    dd_status = None
                    logger.warning("No dd_status returned from the stored procedure.")

                logger.info(f"SP_INSERT_DD result: msg={msg}, dd_status={dd_status_decoded}, sp_code={sp_code}")

            await conn.commit()
            logger.info("Detections inserted into the database via stored procedure.")
            return dd_status_decoded  