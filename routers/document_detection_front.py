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

@router.post("/document-detection/inference/front")
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

        dd_status_decoded = await insert_detections_into_db([detection])
        print(dd_status_decoded)
        if id_type == 2:
            payload = {
                "ResponseData": {
                    "IsDocumentScanCompleted": True,
                    "IsVerified": False,
                    "IsBackDocumentNeed": False,
                    "DocumentType": id_type,
                },
                "ResponseCode": 300,
                "ResponseDescription": "Success direct to Face Compare"
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
                "ResponseDescription": "Redirect to Fornt"
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
                        json.dumps(detection.details),
                        0
                    )
                )
                # Fetch the result set
                result = await cursor.fetchall()
                if result:

                    row = result[0] 
                    msg = row[0] 
                    dd_status = row[1]
                    sp_code = row[2]  # Extract the first value (e.g., '1')
                    
                    dd_status_decoded =  int.from_bytes(dd_status, byteorder='big')


                    logger.info(f"Stored procedure response dd_status: {dd_status}")
                else:
                    dd_status = None
                    logger.warning("No dd_status returned from the stored procedure.")

                print(dd_status)
                print(msg)
                print(dd_status_decoded)
                print(sp_code)

            await conn.commit()
            logger.info("Detections inserted into the database via stored procedure.")
            return dd_status_decoded  # Return the extracted status
