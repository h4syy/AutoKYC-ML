import boto3
import os
from dotenv import load_dotenv
from fastapi import HTTPException

   # Load environment variables
load_dotenv()

   # Initialize Rekognition client using environment variables
rekognition_client = boto3.client(
       'rekognition',
       region_name=os.getenv("AWS_REGION"),
       aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
       aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
   )

async def run_face_comparison(source_image_path: str, target_image_path: str):
    """Compares two faces using AWS Rekognition."""
    
    # Check if image files exist and are not empty
    if not (os.path.exists(source_image_path) and os.path.getsize(source_image_path) > 0):
        raise HTTPException(status_code=400, detail="Source image file is missing or empty")
    if not (os.path.exists(target_image_path) and os.path.getsize(target_image_path) > 0):
        raise HTTPException(status_code=400, detail="Target image file is missing or empty")
    
    with open(source_image_path, 'rb') as source_image:
        source_bytes = source_image.read()
    with open(target_image_path, 'rb') as target_image:
        target_bytes = target_image.read()

    if not source_bytes or not target_bytes:
        raise HTTPException(status_code=400, detail="One or both image files are empty")
    
    try:
        response = rekognition_client.compare_faces(
            SourceImage={'Bytes': source_bytes},
            TargetImage={'Bytes': target_bytes},
            SimilarityThreshold=0
        )

        face_matches = response.get('FaceMatches', [])
        unmatched_faces = response.get('UnmatchedFaces', [])
        source_bounding_box = response.get('SourceImageFace', {}).get('BoundingBox', {})
        
        results = {
            "source_image_bounding_box": source_bounding_box,
            "face_matches": [],
            "unmatched_faces": unmatched_faces,
            "msisdn": '1234567890',  # Placeholder value
            "session_id": '89bb23de-c331-4cae-bcb3-babb55ebcbfe'  # Placeholder value
        }

        for match in face_matches:
            similarity = match['Similarity']
            face_bounding_box = match['Face']['BoundingBox']
            bounding_box = {
                "width": face_bounding_box['Width'],
                "height": face_bounding_box['Height'],
                "left": face_bounding_box['Left'],
                "top": face_bounding_box['Top'],
            }
            results['face_matches'].append({
                "similarity": similarity,
                "bounding_box": bounding_box
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(e)}")