from fastapi import APIRouter, UploadFile, HTTPException, File, Form
from schema.schemas import FaceComparisonResponse, FaceComparisonResult
from utilities.aws_rekognition import run_face_comparison
from utilities.logger import logger
import os
import shutil
import json
from database import dbconfig
from utilities.image_cropper import image_cropper
from utilities.config import get_image_save_path_minio, MINIO_BUCKET, download_from_minio, client
from tempfile import NamedTemporaryFile

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
        # Create temporary files to store the images
        document_front_local_path = NamedTemporaryFile(delete=False, suffix=".jpg").name
        liveness_document_local_path = NamedTemporaryFile(delete=False, suffix=".jpg").name

        # Download images from MinIO to temporary files
        download_from_minio(document_front, document_front_local_path)
        download_from_minio(liveness_document, liveness_document_local_path)

        # Run face comparison with the downloaded images
        result = await run_face_comparison(document_front_local_path, liveness_document_local_path)
        source_image_details = result.get("source_image_bounding_box")

        cropped_image_path = None
        if source_image_details:
            # Generate the path where the cropped image will be saved in MinIO
            cropped_image_path = get_image_save_path_minio(msisdn, session_id, "cropped_image")
            
            # Perform image cropping and get the cropped image as a byte stream
            cropped_image_stream = image_cropper(document_front_local_path, source_image_details)

            if cropped_image_stream:
                # Upload cropped image directly to MinIO
                client.put_object(
                    bucket_name=MINIO_BUCKET,
                    object_name=cropped_image_path,
                    data=cropped_image_stream,
                    length=len(cropped_image_stream.getvalue()),
                    content_type="image/jpeg"
                )
                logger.info(f"Cropped image uploaded to MinIO at: {cropped_image_path}")
            else:
                logger.error("Error while cropping the image.")
        else:
            logger.warning("Source image bounding box not found in the result.")

        face_matches = result.get('face_matches', [])
        confidence = None
        similarity = None
        details = {}

        for match in face_matches:
            similarity = match['similarity']
            confidence = match['confidence']
            details = {
                "confidence": confidence,
                "similarity": similarity,
                "bounding_box": match['bounding_box'],
            }

        # Insert face comparison result into the database
        fc_status_decoded = await insert_face_compare_result(
            session_id=session_id,
            csid=csid,
            Cropped_img_path=cropped_image_path,
            confidence=confidence / 100,
            similarity=similarity / 100,
            details=details,
            msisdn=msisdn,
        )

        # Prepare the response payload
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
                fc_status = row[1]
                fc_status_decoded = int.from_bytes(fc_status, byteorder="big")
                logger.info(f"Stored procedure response fc_status: {fc_status}")
            else:
                fc_status_decoded = None
                logger.warning("No fc_status returned from the stored procedure")

            await conn.commit()
            logger.info("Detections inserted into database via stored procedure")
            return fc_status_decoded