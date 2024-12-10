import os
from PIL import Image

def image_cropper(image_path: str, bounding_box: dict) -> str:
    try :

        with Image.open(image_path) as img:
            img_width, img_height = img.size

            left = int(bounding_box['Left'] * img_width)
            top = int(bounding_box['Top'] * img_height)
            right = left + int(bounding_box['Width'] * img_width)
            bottom = top + int(bounding_box['Height'] * img_height)

            cropped_img = img.crop((left, top, right, bottom))

            cropped_image_name = f"cropped_{os.path.basename(image_path)}"
            cropped_image_path = os.path.join('datastore', cropped_image_name)
            cropped_img.save(cropped_image_path)
    except Exception as e:
        print("Error aayo", e,flush=True)
    return cropped_image_path
