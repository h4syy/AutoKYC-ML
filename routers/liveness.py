from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import json
from database import dbconfig
from UTILS.logger import logger
router = APIRouter()

@router.post("/liveness/post-data")
async def post_data(request: Request):
    request_data = await request.json()
    logger.info(f"Trying to insert liveness data:  {type(request_data)}")
    logger.info(f"Trying to insert type of liveness data:  {request_data}")
    session_id = request_data.get("SessionId")
    csid = request_data.get("CSID")
    liveness_photo_path = request_data.get("AuditImages", [{}])[0].get("S3Object", {}).get("Name")
    bounding_box = json.dumps(request_data.get("AuditImages", [{}])[0].get("BoundingBox"))
    confidence = request_data.get("Confidence")
    msisdn = request_data.get("MSISDN")
    status = request_data.get("Status")
    reference_img = request_data.get("ReferenceImage")
    audit_images = request_data.get("AuditImages")

    liveness_data = {
        "SessionId": session_id,
        "MSISDN": msisdn,
        "CreatedDate": datetime.now(),
        "Confidence": confidence,
        "BoundingBox": bounding_box,
        "CSID": csid,
        "Status": 1 if status == "SUCCEEDED" else 0,
        "LivenessPhotoPath": liveness_photo_path,
        "Details": json.dumps({
            "ReferenceImage": reference_img,
            "AuditImages": audit_images
        })
    }
    await insert_liveness_result(session_id, csid, liveness_photo_path, bounding_box, float(confidence), 1 if status == "SUCCEEDED" else 0, msisdn, liveness_data["Details"])
    return {"message": "Data processed successfully", "data": liveness_data}

async def insert_liveness_result(msisdn, session_id, confidence, csid, liveness_photo_path, bounding_box, details):
    sp_query = "CALL SP_INSERT_LIVENESS(%s, %s, %s, %s, %s, %s, %s)"
    
    async with dbconfig.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(sp_query, (msisdn, session_id, confidence, csid, liveness_photo_path, bounding_box, details))
            
            result = await cursor.fetchall()
            print(result)

            if result:
                row = result[0]

                msg = row[0]
                lv_status = row[1]
                sp_code = row[2]
                lv_status_decoded = int.from_bytes(lv_status, byteorder='big')

                print("Message:", msg)
                print("LV Status:", lv_status_decoded)
                print("SP Code (int):", sp_code)

            await conn.commit()