from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np
import io

app = FastAPI(title="Auto Damage Segmentation API")

yolo_model = YOLO("../model/best.onnx", task="segment")
sam_model = SAM("sam_b.pt")

@app.get("/")
def root():
    return {"message": "Auto Damage Segmentation API — YOLO + SAM Pipeline"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)

    yolo_results = yolo_model.predict(img_array, conf=0.25, verbose=False)[0]

    detections = []
    boxes = []

    if yolo_results.boxes is not None:
        for box in yolo_results.boxes:
            bbox = box.xyxy[0].tolist()
            boxes.append(bbox)
            detections.append({
                "class": yolo_results.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": bbox
            })

    sam_masks_count = 0
    if boxes:
        sam_results = sam_model(img_array, bboxes=boxes)
        if sam_results and sam_results[0].masks is not None:
            sam_masks_count = len(sam_results[0].masks)

    return JSONResponse({
        "total_detected": len(detections),
        "sam_masks": sam_masks_count,
        "detections": detections
    })