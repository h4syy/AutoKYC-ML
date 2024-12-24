from PIL import Image
from io import BytesIO
import os

def image_cropper(image_path: str, bounding_box: dict) -> BytesIO:
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size

            left = int(bounding_box['Left'] * img_width)
            top = int(bounding_box['Top'] * img_height)
            right = left + int(bounding_box['Width'] * img_width)
            bottom = top + int(bounding_box['Height'] * img_height)

            cropped_img = img.crop((left, top, right, bottom))

            # Save the cropped image to a BytesIO stream instead of a file
            cropped_image_stream = BytesIO()
            cropped_img.save(cropped_image_stream, format='JPEG')
            cropped_image_stream.seek(0)  # Go to the start of the stream

            return cropped_image_stream
    except Exception as e:
        print(f"Error occurred while cropping image: {e}")
        return None
