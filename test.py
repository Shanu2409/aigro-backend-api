import os
import shutil
import time

import torch
from ultralytics import YOLO
import base64
from PIL import Image
from typing import Union
from io import BytesIO

def save_image_base64(image_path: Union[str, BytesIO]) -> str:
    # Open the image file
    with Image.open(image_path) as img:
        # Create a BytesIO object
        with BytesIO() as buffer:
            # Save image to buffer
            img.save(buffer, format='PNG')
            # Get the byte data
            img_byte_data = buffer.getvalue()
    # Convert bytes to base64 string
    base64_str = base64.b64encode(img_byte_data).decode()
    return base64_str

def clear_cuda_memory():
    torch.cuda.empty_cache()

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert('RGB')
    return img

def object_detection_yolo(base64_image, model_path="./models/best.pt", conf_threshold=0.30, iou_threshold=0.30):
    # Clear GPU memory cache
    clear_cuda_memory()

    # Load YOLO model
    yolo_model = YOLO(model_path)

    # Convert base64 image to PIL Image
    image = base64_to_image(base64_image)

    # Perform object detection
    results = yolo_model(image, show=False, save=True, conf=conf_threshold, iou=iou_threshold)

    return results

# Example usage:

def start(base64_image: str):
    detection_results = object_detection_yolo(base64_image)
    # show the file name from 'runs/detect/predict'
    print(os.listdir('runs/detect/predict'))
    jpg_files = [file for file in os.listdir('runs/detect/predict') if file.endswith('.jpg')]

    # If there is at least one jpg file
    if jpg_files:
        # Open and display the first jpg file
        # img = Image.open(os.path.join('runs/detect/predict', jpg_files[0]))
        # img.show()

        base64_str = save_image_base64(os.path.join('runs/detect/predict', jpg_files[0]))
        shutil.rmtree('runs')

        # show base64 in image form
        # img_data = base64.b64decode(base64_str)
        # img = Image.open(BytesIO(img_data)).convert('RGB')
        # img.show()

        return base64_str
    else:
        print("No jpg files found in the directory.")