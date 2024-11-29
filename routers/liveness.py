from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import json
from database import dbconfig

router = APIRouter()

@router.post("/liveness/post-data")
async def post_data(request: Request):
    request_data = await request.json()

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
        "Status": status,
        "LivenessPhotoPath": liveness_photo_path,
        "Details": json.dumps({
            "ReferenceImage": reference_img,
            "AuditImages": audit_images
        })
    }
    await insert_liveness_result(session_id, csid, liveness_photo_path, bounding_box, float(confidence), status, msisdn, liveness_data["Details"])
    return {"message": "Data processed successfully", "data": liveness_data}

async def insert_liveness_result(session_id, csid, liveness_photo_path, bounding_box, confidence, status, msisdn, details):
    query = """
    INSERT INTO liveness (SessionId, CreatedDate, CSID, LivenessPhotopath, BoundingBox, Confidence, Status, MSISDN, Details)
    VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s)
    """
    async with dbconfig.db_pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (session_id, csid, liveness_photo_path, bounding_box, confidence, status, msisdn, details))
            await conn.commit()