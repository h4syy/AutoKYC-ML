# controllers/face_compare_controller.py
from fastapi import UploadFile, HTTPException
from services.face_compare_service import run_face_comparison
import os

async def handle_face_comparison(source: UploadFile, target: UploadFile):
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Save the uploaded files temporarily
    source_path = os.path.join(uploads_dir, source.filename)
    target_path = os.path.join(uploads_dir, target.filename)

    with open(source_path, "wb") as buffer:
        buffer.write(await source.read())

    with open(target_path, "wb") as buffer:
        buffer.write(await target.read())

    try:
        # Call the service for face comparison
        comparison_result = await run_face_comparison(source_path, target_path)

        # Optionally remove files after processing
        os.remove(source_path)
        os.remove(target_path)

        # Return the result
        return {
            "message": "Face comparison completed successfully",
            "results": comparison_result
        }

    except Exception as e:
        print(f"Error during face comparison: {e}")
        raise HTTPException(status_code=500, detail="Error during face comparison")
