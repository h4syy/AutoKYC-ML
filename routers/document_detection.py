# from fastapi import APIRouter, UploadFile, File, HTTPException, Form
# from models.schemas import Detection, DetectionResponse
# from utils.logger import logger
# from database import dbconfig
# import shutil
# import os
# import torch
# import json

# router = APIRouter()  # Ensure APIRouter is created and named 'router'

# # Load the YOLO model once when the server starts
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

# output_dir = '/output'
# os.makedirs(output_dir, exist_ok=True)

# @router.post("/document-detection/inference", response_model=DetectionResponse)
# async def detect_document(
#     file: UploadFile = File(...),
#     session_id: str = Form(...),
#     csid: str = Form(...),
#     msisdn: int = Form(...)
# ):
#     logger.info("Document detection inference started.")
#     temp_image_path = os.path.join(output_dir, file.filename)
    
#     try:
#         with open(temp_image_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         logger.info(f"File saved to {temp_image_path}.")

#         results = model(temp_image_path)
#         detections = []

#         for *box, confidence, cls in results.xyxy[0].tolist():
#             class_name = results.names[int(cls)]

#             detections.append(Detection(
#                 session_id=session_id,
#                 csid=csid,
#                 predicted_class=class_name,
#                 document_photo_path=temp_image_path,
#                 bounding_box=str(box),
#                 confidence=confidence,
#                 details={},
#                 msisdn=msisdn
#             ))
        
#         await insert_detections_into_db(detections)

#         logger.info("Document detection inference completed.")
#         return DetectionResponse(detections=detections)

#     except Exception as e:
#         logger.error(f"Error during document detection: {e}")
#         raise HTTPException(status_code=500, detail="An error occurred during document detection.")
    
#     finally:
#         if os.path.exists(temp_image_path):
#             os.remove(temp_image_path)
#             logger.info(f"Temporary file {temp_image_path} removed.")

# async def insert_detections_into_db(detections: list[Detection]):
#     async with dbconfig.db_pool.acquire() as conn:
#         async with conn.cursor() as cursor:
#             insert_query = """
#             INSERT INTO documentdetection (SessionId, CSID, PredictedClass, DocumentPhotopath, BoundingBox, Confidence, Details, MSISDN)
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#             """
#             for detection in detections:
#                 await cursor.execute(insert_query, (
#                     detection.session_id,
#                     detection.csid,
#                     detection.predicted_class,
#                     detection.document_photo_path,
#                     detection.bounding_box,
#                     detection.confidence,
#                     json.dumps(detection.details),
#                     detection.msisdn
#                 ))
#             await conn.commit()
#             logger.info("Detections inserted into the database.")