# services/face_compare_service.py
import boto3
import os

# Define your AWS credentials (load these securely in production)
AWS_ACCESS_KEY_ID = "AKIAZQ3DTJ4GVMWRSCVR"
AWS_SECRET_ACCESS_KEY = "jST4l/iMISfFWmbW9Z4v0FCHuq52PRnr08ij27U3"
AWS_REGION = "us-east-1"

# Initialize Rekognition client using boto3
rekognition_client = boto3.client(
    'rekognition',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

async def run_face_comparison(source_image_path: str, target_image_path: str):
    with open(source_image_path, 'rb') as source_image:
        source_bytes = source_image.read()

    with open(target_image_path, 'rb') as target_image:
        target_bytes = target_image.read()

    try:
        # Call AWS Rekognition to compare faces
        response = rekognition_client.compare_faces(
            SourceImage={'Bytes': source_bytes},
            TargetImage={'Bytes': target_bytes},
            SimilarityThreshold=70
        )

        # Parse results
        face_matches = response.get('FaceMatches', [])
        unmatched_faces = response.get('UnmatchedFaces', [])
        source_bounding_box = response.get('SourceImageFace', {}).get('BoundingBox', {})

        results = {
            "source_image_bounding_box": source_bounding_box,
            "face_matches": [],
            "unmatched_faces": unmatched_faces
        }

        for match in face_matches:
            similarity = match['Similarity']
            bounding_box = match['Face']['BoundingBox']
            results['face_matches'].append({
                "similarity": similarity,
                "bounding_box": bounding_box
            })

        # Return the results
        return results

    except Exception as e:
        print(f"Error during face comparison: {e}")
        raise Exception("Face comparison failed.")
