# Road Crack Detection using YOLOv11 Segmentation — Project Report

## 1. Project Overview

### Project Title

Road Crack Detection using YOLOv11 Segmentation

### Author

Rasulbek Ruzmetov

### Domain

Computer Vision / Deep Learning / Infrastructure Inspection

### Objective

The objective of this project is to automatically detect and segment road cracks from road surface images using a deep learning-based instance segmentation model.

The system helps transportation authorities and maintenance teams identify damaged road sections quickly and efficiently.

---

## 2. Problem Statement

Manual road inspection is:

* Time-consuming
* Expensive
* Error-prone
* Difficult to scale

Traditional inspection methods require engineers to visually inspect roads and document damage manually.

This project automates the process using Computer Vision and Deep Learning.

---

## 3. Proposed Solution

A YOLOv11 Segmentation model was trained to identify and segment road cracks at the pixel level.

The model receives a road image as input and returns:

* Crack locations
* Segmentation masks
* Crack count
* Crack area estimation
* Severity assessment

---

## 4. Dataset

### Source

Roboflow Universe

### Task Type

Instance Segmentation

### Annotation Format

YOLO Segmentation Format

### Image Resolution

640 × 640

### Dataset Split

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 20%        |
| Test       | 10%        |

---

## 5. Data Preprocessing

The following preprocessing techniques were applied:

* Auto Orientation
* Image Resizing (640×640)
* Normalization

### Data Augmentation

* Horizontal Flip
* Vertical Flip
* Rotation
* Brightness Adjustment
* Contrast Adjustment
* Gaussian Blur
* Noise Injection

These augmentations improve model generalization.

---

## 6. Model Architecture

### Base Model

YOLOv11n-seg

### Training Configuration

| Parameter  | Value |
| ---------- | ----- |
| Epochs     | 50    |
| Batch Size | 16    |
| Image Size | 640   |
| Optimizer  | SGD   |
| Pretrained | COCO  |

### Framework

Ultralytics YOLO

---

## 7. Training Pipeline

Dataset Collection
↓
Data Annotation
↓
Data Preprocessing
↓
YOLOv11 Segmentation Training
↓
Validation
↓
Model Evaluation
↓
Deployment using Gradio

---

## 8. Evaluation Metrics

The model was evaluated using the following metrics:

### Precision

Measures how many detected cracks are correct.

### Recall

Measures how many real cracks were successfully detected.

### mAP50

Mean Average Precision at IoU 0.50.

### F1 Score

Harmonic mean of Precision and Recall.

---

## 9. Output Analysis

The system generates:

### Crack Count

Number of detected cracks.

### Crack Area

Estimated crack area in pixels.

### Segmentation Mask

Pixel-level crack visualization.

### Severity Classification

Based on crack area.

| Crack Area    | Severity |
| ------------- | -------- |
| < 5000 px     | Low      |
| 5000–15000 px | Medium   |
| > 15000 px    | High     |

---

## 10. Deployment

A Gradio application was developed for interactive testing.

### Workflow

Upload Image
↓
YOLOv11 Inference
↓
Crack Segmentation
↓
Severity Assessment
↓
Visualization

---

## 11. Results

The model successfully identifies road cracks and highlights damaged regions using segmentation masks.

Benefits:

* Faster inspection
* Reduced manual effort
* Consistent detection
* Scalable infrastructure monitoring

---

## 12. Future Improvements

Potential improvements include:

* Video-based crack detection
* Real-time camera support
* Drone inspection integration
* Crack length measurement
* Crack width estimation
* PDF report generation
* FastAPI deployment
* ONNX optimization
* Mobile deployment

---

## 13. Technologies Used

* Python 3.11
* YOLOv11 Segmentation
* PyTorch
* OpenCV
* NumPy
* Roboflow
* Gradio
* Git
* GitHub

---

## 14. Conclusion

This project demonstrates how modern Computer Vision techniques can be applied to infrastructure monitoring and road maintenance.

The developed system provides an automated and scalable solution for detecting and segmenting road cracks, reducing inspection costs and improving maintenance efficiency.

The project serves as a practical application of Deep Learning, Object Detection, and Instance Segmentation in real-world scenarios.
