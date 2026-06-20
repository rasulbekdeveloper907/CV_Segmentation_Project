from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI(title="Road Crack Detection API")

# Model load
model = YOLO("../models/best.pt")


@app.get("/")
def home():
    return {"message": "Road Crack Detection API is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = np.array(image)

    results = model.predict(image, conf=0.25, verbose=False)[0]

    detections = []
    crack_count = 0
    total_area = 0

    if results.masks is not None:

        masks = results.masks.data.cpu().numpy()
        crack_count = len(masks)

        for i, mask in enumerate(masks):

            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)

            area = int(mask.sum())
            total_area += area

            detections.append({
                "id": i,
                "area": area
            })

    if total_area < 5000:
        severity = "Low"
    elif total_area < 15000:
        severity = "Medium"
    else:
        severity = "High"

    return {
        "crack_count": crack_count,
        "total_area": total_area,
        "severity": severity,
        "detections": detections
    }