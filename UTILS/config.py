import os
from datetime import datetime

Photo_Root = "Photo"

def get_image_save_path(msisdn: int, suffix: str = "front_image"):
    now = datetime.now()
    year = now.strftime("%Y")        
    month = now.strftime("%b")         
    day = now.strftime("%d")           

    year_path = os.path.join(Photo_Root, year)
    month_path = os.path.join(year_path, month)
    day_path = os.path.join(month_path, day)

    os.makedirs(day_path, exist_ok=True)

    file_name = f"{msisdn}_{suffix}.jpg"

    return os.path.join(day_path, file_name)
