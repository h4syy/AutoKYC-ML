from fastapi import APIRouter, UploadFile, HTTPException, File, Form
from MODELS.schemas import FaceComparisonResponse, FaceComparisonResult
from UTILS.aws_rekognition import run_face_comparison
from UTILS.logger import logger
import os
import shutil
import json
from database import dbconfig
from UTILS.image_cropper import image_cropper

router = APIRouter()

@router.post("/face/compare", response_model=FaceComparisonResponse)
async def face_compare(
    document_front: UploadFile = File(...),
    liveness_document: UploadFile = File(...),
    session_id: str = Form(...),
    csid: str = Form(...),
    msisdn: int = Form(...)
):
    logger.info("Face comparison inference started.")
    temp_front_document = os.path.join("/output", document_front.filename)
    temp_liveness_path = os.path.join("/output", liveness_document.filename)
    
    # Save uploaded files temporarily
    with open(temp_front_document, "wb") as buffer:
        shutil.copyfileobj(document_front.file, buffer)
    with open(temp_liveness_path, "wb") as buffer:
        shutil.copyfileobj(liveness_document.file, buffer)

    try:
        result = await run_face_comparison(temp_front_document, temp_liveness_path)
        source_image_details = result["source_image_bounding_box"]
        print(source_image_details,flush=True)
        if source_image_details:
            crop_result = image_cropper(temp_front_document, source_image_details)

            print(f"Cropped image saved at: {crop_result}", flush=True)
        else:
            print("Source image bounding box not found in the result.", flush=True)
        
        face_matches = result['face_matches']

        for match in face_matches:
            logger.info("this is for match", match)
            similarity = match['similarity']
            confidence = match['confidence']
            details = {
                "confidence": confidence,
                "similarity": similarity,
                "bounding_box": match['bounding_box']
            }


        fc_status_decoded =  await insert_face_compare_result(
                session_id=session_id,
                csid=csid,
                Cropped_img_path= temp_liveness_path,
                confidence=confidence,
                similarity=similarity,
                details=details,
                msisdn=msisdn
                )
            
        if fc_status_decoded == 1:
                payload = {
                    "ResponseData":{
                        "IsDocumentScanCompleted": True,
                        "IsVerified": True,
                        "IsBackDocumentNeed": False,
                        "DocumentType": 1,
                    },
                    "ResponseCode": 500,
                    "ResponseDescription": "Success"
                }

        else:
                payload = {
                    "ResponseData":{
                        "IsDocumentScanCompleted": False,
                        "IsVerified": False,
                        "IsBackDocumentNeed": False,
                        "DocumentType": 1,
                    },
                    "ResponseCode": 500,
                    "ResponseDescription": "Redirect to Document Detection Front"
                }

                logger.info("Face comparison completed.")
                return payload

        logger.info("Face comparison completed.")
    except Exception as e:
        logger.error(f"Error during face comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        os.remove(temp_front_document)
        os.remove(temp_liveness_path)
        logger.info("Temporary images removed.")

    # return FaceComparisonResponse(
    #     source_image_bounding_box=result['source_image_bounding_box'],
    #     face_matches=[FaceComparisonResult(**match) for match in face_matches],
    #     unmatched_faces=result['unmatched_faces'],
    #     msisdn=msisdn,
    #     session_id=session_id,
    #     confidence = confidence
    # )

async def insert_face_compare_result(session_id, csid, similarity, confidence, details, msisdn, Cropped_img_path):
    sp_query = "CALL SP_INSERT_FACECOMPARE(%s, %s, %s, %s, %s, %s, %s)"
    async with dbconfig.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sp_query, (msisdn, session_id, csid, confidence, similarity, Cropped_img_path, json.dumps(details)))
            
            result = await cursor.fetchall()
            if result:
                row = result[0]
                msg = row[0]
                fc_status = row[1]
                sp_code = row[2]

                fc_status_decoded = int.from_bytes(fc_status, byteorder='big')

                logger.info(f"Stored procedure response fc_status: {fc_status}")

            else:
                fc_status = None
                logger.warning("No ff_status retured form the stored procedure")

            print(fc_status)
            print(msg)
            print(fc_status_decoded)
            print(sp_code)
            
            await conn.commit()
            logger.info("Detections inserted into database via stored procedure")
            return fc_status_decoded