from fastapi import APIRouter, Depends, UploadFile, File
from controllers.inference_controller import handle_inference

router = APIRouter()

@router.post("/run-inference")
async def upload_file(file: UploadFile = File(...)):
    print("log router")
    return await handle_inference(file=file)
