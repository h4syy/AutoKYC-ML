import os
from datetime import datetime
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from io import BytesIO
from utilities.logger import logger
from PIL import Image, ImageDraw
from base64 import b64decode

# Load environment variables
load_dotenv()

# Constants
Photo_Root = "Photo"
MINIO_URL = os.getenv("MINIO_URL")
MINIO_USER = os.getenv("MINIO_USER")
MINIO_PASS = os.getenv("MINIO_PASS")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# MinIO Client with HTTPS
client = Minio(
    MINIO_URL.replace("https://", "").replace("http://", ""),  # Remove protocol for MinIO client
    access_key=MINIO_USER,
    secret_key=MINIO_PASS,
    secure=MINIO_URL.startswith("https"),  # True for HTTPS
)

def get_image_save_path_minio(msisdn: int, session_id: str, suffix: str):
    """Generate the remote path for storing the image in MinIO."""
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%B")
    day = now.strftime("%D").replace("/", "-")  # MinIO doesn't support "/" in object names

    file_name = f"{msisdn}_{session_id}_{suffix}.jpg"
    return os.path.join(Photo_Root, year, month, day, file_name).replace("\\", "/")

def upload_to_minio(msisdn: int, suffix: str):
    """Create a file structure in MinIO and upload content."""
    try:
        # Ensure bucket exists
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            print(f"Bucket '{MINIO_BUCKET}' created.")
        else:
            print(f"Bucket '{MINIO_BUCKET}' already exists.")

        # Create remote path
        remote_path = get_image_save_path_minio(msisdn, session_id, suffix)

        # Generate file content in memory
        file_content = BytesIO(b"This is a sample content for the image.")
        file_size = file_content.getbuffer().nbytes

        # Upload to MinIO
        client.put_object(
            bucket_name=MINIO_BUCKET,
            object_name=remote_path,
            data=file_content,
            length=file_size,
            content_type="image/jpeg",  # Adjust content type if needed
        )
        logger.info(f"File successfully uploaded to MinIO at '{remote_path}'.")
    except S3Error as exc:
        logger.error(f"Error occurred while uploading to MinIO: {exc}")
        raise
    except Exception as e:
        logger.error(f"Failed to process and upload image: {e}")
        raise


def download_from_minio(object_name: str, download_path: str):
    """Download file from MinIO to the local path."""
    try:
        # Download the object from MinIO
        client.fget_object(MINIO_BUCKET, object_name, download_path)
        logger.info(f"Downloaded {object_name} from MinIO to {download_path}")
    except S3Error as exc:
        logger.error(f"Error occurred while downloading from MinIO: {exc}")
        raise
    except Exception as e:
        logger.error(f"Failed to download image from MinIO: {e}")
        raise


