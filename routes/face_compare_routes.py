# routes/face_compare_router.py
from fastapi import APIRouter, UploadFile
from controllers.face_compare_controller import handle_face_comparison

router = APIRouter()

# Face comparison route
@router.post("/compare-faces")
async def compare_faces(source: UploadFile, target: UploadFile):
    return await handle_face_comparison(source, target)
