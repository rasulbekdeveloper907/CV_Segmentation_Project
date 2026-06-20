from ultralytics import YOLO
import gradio as gr
import cv2
import numpy as np

# Model yuklash
model = YOLO("../models/best.pt")


def detect_crack(image):

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    output = image.copy()

    crack_count = 0
    total_area = 0

    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()

        crack_count = len(masks)

        for mask in masks:

            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0])
            )

            area = int(mask.sum())

            total_area += area

            colored_mask = np.zeros_like(image)

            colored_mask[:, :, 0] = (
                mask * 255
            ).astype(np.uint8)

            output = cv2.addWeighted(
                output,
                1,
                colored_mask,
                0.4,
                0
            )

    severity = "Low"

    if total_area > 5000:
        severity = "Medium"

    if total_area > 15000:
        severity = "High"

    report = f"""
Road Crack Analysis

Crack Count: {crack_count}

Crack Area: {total_area} pixels

Severity: {severity}
"""

    return output, report


app = gr.Interface(
    fn=detect_crack,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Segmentation Result"),
        gr.Textbox(label="Report")
    ],
    title="Road Crack Detection",
    description="YOLOv11 Segmentation + Gradio"
)

app.launch()