from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from MODELS.schemas import Detection, DetectionResponse
from UTILS.logger import logger
from database import dbconfig
import shutil
import os
import torch
import json

router = APIRouter() 

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

output_dir = '/output'
os.makedirs(output_dir, exist_ok=True)

@router.post("/document-detection/inference/front", response_model=DetectionResponse)
async def detect_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    csid: str = Form(...),
    msisdn: int = Form(...)
):
    logger.info(f"Document detection inference started for session_id: {session_id}")
    temp_image_path = os.path.join(output_dir, file.filename)
        #face compare bata aako response fail bhayepachi yaha rakhne 
    id_type_mapping = {
        "CS": 1,   
        "DL": 2, 
        "NID": 3,  
        "PP": 4   
    }
    
    try:
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {temp_image_path}")

        results = model(temp_image_path)
        if len(results.xyxy[0]) == 0:
            raise HTTPException(status_code=400, detail="No document detected.")
        
        detection_data = results.xyxy[0].tolist()[0]  
        *box, confidence, cls = detection_data
        class_name = results.names[int(cls)]
        suffix = class_name[-1] 
        prefix = class_name[:-1] if suffix in {"F", "B"} else class_name
        id_type = id_type_mapping.get(prefix, -1)

        async with dbconfig.db_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT ID_Type, PredictedClassF FROM documentdetection WHERE SessionId = %s", 
                    (session_id,)
                )
                existing_entries = await cursor.fetchall()

                if not existing_entries:
                    if suffix != "F":
                        raise HTTPException(
                            status_code=400, 
                            detail="First upload must be a front ID document."
                        )
                else:
                    last_id_type, last_class = existing_entries[0]
                    if id_type != last_id_type:
                        raise HTTPException(
                            status_code=400,
                            detail="ID type mismatch. Upload back ID of the same type as the front ID."
                        )
        detection = Detection(
            session_id=session_id,
            csid=csid,
            id_type=id_type,
            predicted_class=class_name,
            document_photo_path=temp_image_path,
            bounding_box=str(box),
            confidence=confidence,
            details={},
            msisdn=msisdn
        )

        await insert_detections_into_db([detection])

        logger.info("Document detection inference completed successfully.")
        return DetectionResponse(detections=[detection])

    except Exception as e:
        logger.error(f"Error during document detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            logger.info(f"Temporary file {temp_image_path} removed.")

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
                        json.dumps(detection.details)
                    )
                )
            await conn.commit()
            logger.info("Detections inserted into the database via stored procedure.")
