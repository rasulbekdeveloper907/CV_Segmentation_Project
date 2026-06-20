import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("../models/best.pt")


def detect(image):

    results = model.predict(image, conf=0.25, verbose=False)[0]

    output = image.copy()

    crack_count = 0
    total_area = 0

    if results.masks is not None:

        masks = results.masks.data.cpu().numpy()
        crack_count = len(masks)

        for mask in masks:

            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)

            total_area += int(mask.sum())

            color = np.zeros_like(image, dtype=np.uint8)
            color[:, :, 2] = mask * 255

            output = cv2.addWeighted(output, 1, color, 0.4, 0)

    if total_area < 5000:
        severity = "Low"
    elif total_area < 15000:
        severity = "Medium"
    else:
        severity = "High"

    text = f"""
Crack Count: {crack_count}
Total Area: {total_area}
Severity: {severity}
"""

    return output, text


demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Result"),
        gr.Textbox(label="Report")
    ],
    title="🚧 Road Crack Detection (YOLOv11)",
    description="Upload road image → detect cracks"
)

demo.launch()