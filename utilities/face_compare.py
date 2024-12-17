from fastapi import APIRouter, UploadFile, HTTPException, File, Form
from schema.schemas import FaceComparisonResponse, FaceComparisonResult
from utilities.aws_rekognition import run_face_comparison
from utilities.logger import logger
import os
import shutil
import json
from database import dbconfig
from utilities.image_cropper import image_cropper
from utilities.config import get_image_save_path

router = APIRouter()

@router.post("/face/compare", response_model=FaceComparisonResponse)
async def face_compare_auto(
    document_front: str = Form(...), 
    liveness_document: str = Form(...), 
    session_id: str = Form(...),
    csid: str = Form(...),
    msisdn: int = Form(...)
):
    logger.info("Face comparison inference started.")
    
    try:
        document_front_path = document_front  
        liveness_document_path = liveness_document  

        result = await run_face_comparison(document_front_path, liveness_document_path)
        source_image_details = result["source_image_bounding_box"]

        cropped_image_path = None
        if source_image_details:
            cropped_image_path = get_image_save_path(msisdn, "cropped_image")
            crop_result = image_cropper(document_front_path, source_image_details)
            shutil.move(crop_result, cropped_image_path)
            logger.info(f"Cropped image saved at: {cropped_image_path}")
        else:
            logger.warning("Source image bounding box not found in the result.")

        face_matches = result['face_matches']
        for match in face_matches:
            similarity = match['similarity']
            confidence = match['confidence']
            details = {
                "confidence": confidence,
                "similarity": similarity,
                "bounding_box": match['bounding_box'],
            }

        fc_status_decoded = await insert_face_compare_result(
            session_id=session_id,
            csid=csid,
            Cropped_img_path=cropped_image_path,
            confidence=confidence,
            similarity=similarity,
            details=details,
            msisdn=msisdn,
        )

        if fc_status_decoded == 1:
            payload = {
                "ResponseData": {
                    "IsDocumentScanCompleted": True,
                    "IsVerified": True,
                    "IsBackDocumentNeed": False,
                    "DocumentType": 1,
                },
                "ResponseCode": 400,
                "ResponseDescription": "Done"
            }
        else:
            payload = {
                "ResponseData": {
                    "IsDocumentScanCompleted": False,
                    "IsVerified": False,
                    "IsBackDocumentNeed": True,
                    "DocumentType": 1,
                },
                "ResponseCode": 400,
                "ResponseDescription": "Redirect to Back"
            }

            logger.info("Document detection inference completed successfully.")
            return payload

        logger.info("Face comparison completed.")
        return payload

    except Exception as e:
        logger.error(f"Error during face comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def insert_face_compare_result(session_id, csid, similarity, confidence, details, msisdn, Cropped_img_path):
    sp_query = "CALL SP_INSERT_FACECOMPARE(%s, %s, %s, %s, %s, %s, %s)"
    async with dbconfig.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                sp_query,
                (msisdn, session_id, csid, confidence, similarity, Cropped_img_path, json.dumps(details)),
            )

            result = await cursor.fetchall()
            if result:
                row = result[0]
                msg = row[0]
                fc_status = row[1]
                sp_code = row[2]

                fc_status_decoded = int.from_bytes(fc_status, byteorder="big")
                logger.info(f"Stored procedure response fc_status: {fc_status}")
            else:
                fc_status_decoded = None
                logger.warning("No fc_status returned from the stored procedure")

            await conn.commit()
            logger.info("Detections inserted into database via stored procedure")
            return fc_status_decoded
