from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
import json
from io import BytesIO
from dotenv import load_dotenv
import os
from database import dbconfig
from utilities.config import get_image_save_path_minio, client
from utilities.logger import logger

load_dotenv()
router = APIRouter()
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

@router.post("/liveness/post-data")
async def post_data(
    request: Request,
    referenceImage: UploadFile = File(...),
    confidence: float = Form(...),
    sessionId: str = Form(...),
    csid: str = Form(...),
    boundingBox: str = Form(...),
    msisdn: int = Form(...),
    status: str = Form(...)
):
    try:
        form_data = await request.form()
        logger.info(f"Received form data: {form_data}")
        logger.info(f"sessionId={sessionId}, csid={csid}, msisdn={msisdn}, confidence={confidence}, status={status}, boundingBox={boundingBox}")

        try:
            bounding_box_dict = json.loads(boundingBox)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid boundingBox JSON: {boundingBox}, error: {e}")
            raise HTTPException(status_code=400, detail="Invalid boundingBox format")

        livenessPhotoPath = get_image_save_path_minio(msisdn, sessionId, "Liveness")
        logger.info(f"Liveness photo will be saved at: {livenessPhotoPath}")

        file_content = await referenceImage.read()
        file_size = len(file_content)
        try:
            client.put_object(
                bucket_name=MINIO_BUCKET,
                object_name=livenessPhotoPath,
                data=BytesIO(file_content),
                length=file_size,
                content_type=referenceImage.content_type
            )
            logger.info(f"File uploaded successfully to MinIO: {livenessPhotoPath}")
        except Exception as e:
            logger.error(f"Error uploading to MinIO: {e}")
            raise HTTPException(status_code=500, detail="File upload failed")

        liveness_data = {
            "SessionId": sessionId,
            "MSISDN": msisdn,
            "Confidence": confidence / 100,
            "BoundingBox": bounding_box_dict,
            "CSID": csid,
            "Status": status,
            "LivenessPhotoPath": livenessPhotoPath,
            "Details": {}
        }

        try:
            lv_status = await insert_liveness_result(
                msisdn=liveness_data["MSISDN"],
                sessionId=liveness_data["SessionId"],
                confidence=liveness_data["Confidence"],
                csid=liveness_data["CSID"],
                livenessPhotoPath=liveness_data["LivenessPhotoPath"],
                bounding_box=liveness_data["BoundingBox"],
                details=liveness_data["Details"]
            )
            
            if lv_status is None:
                logger.error("Database operation failed: No status returned")
                raise HTTPException(status_code=500, detail="Database operation failed")
            
            logger.info(f"Database operation completed with status: {lv_status}")
            
            payload = {
                "ResponseData": {
                    "IsLivenessCompleted": bool(lv_status == 1),
                    "IsDocumentScanCompleted": False,
                    "IsVerified": False,
                    "IsBackDocumentNeed": True,
                    "DocumentType": "Null",
                },
                "ResponseCode": 10 if lv_status == 1 else 11,
                "ResponseDescription": "Success. Proceed to DDF" if lv_status == 1 else "Fail.",
            }
            
            return JSONResponse(content=payload)
            
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise HTTPException(status_code=500, detail="Database operation failed")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing liveness data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def insert_liveness_result(msisdn, sessionId, confidence, csid, livenessPhotoPath, bounding_box, details):
    sp_query = "CALL SP_INSERT_LIVENESS(%s, %s, %s, %s, %s, %s, %s)"
    try:
        async with dbconfig.db_pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sp_query, (msisdn, sessionId, confidence, csid, livenessPhotoPath, json.dumps(bounding_box), json.dumps(details)))
                result = await cursor.fetchall()
                
                if not result:
                    logger.error("No result returned from stored procedure")
                    return None
                    
                row = result[0]
                msg = row[0]
                lv_status = row[1]
                
                await conn.commit()
                return int.from_bytes(lv_status, byteorder='big')

    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")