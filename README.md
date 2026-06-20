# 🛣️ Road Crack Detection using YOLOv11 Segmentation

Yo'l yuzasidagi yoriqlarni (road cracks) avtomatik aniqlash va segmentatsiya qilish tizimi.

Ushbu loyiha yo'llarni monitoring qilish, texnik xizmat ko'rsatishni rejalashtirish va infratuzilma holatini baholash uchun mo'ljallangan.

---

## 🔗 Links

|              | Link                                                            |
| ------------ | --------------------------------------------------------------- |
| 🤗 Live Demo | Hugging Face Space (Coming Soon)                                |
| 📦 Dataset   | Roboflow Road Crack Segmentation Dataset                        |
| 💻 GitHub    | https://github.com/rasulbekdeveloper907/CV_Segmentation_Project |

---

## 🧠 Pipeline

Roboflow Dataset → YOLOv11-seg Training → Model Evaluation → Gradio Deployment

---

## 📦 Technologies

* YOLOv11-seg — Road Crack Instance Segmentation
* Roboflow — Dataset Management & Annotation
* Gradio — Interactive Web Application
* OpenCV — Image Processing
* NumPy — Numerical Operations
* Python 3.11
* PyTorch
* Ultralytics

---

## 📁 Project Structure

```text
CV_Segmentation/
│
├── api/
│   └── main.py
│
├── notebook/
│   └── train.ipynb
│
├── datasets/
│
├── models/
│   └── best.pt
│
├── report/
│   └── report.md
│
├── requirements.txt
│
└── README.md
```

---

## 🎯 Problem Statement

Yo'l yoriqlarini qo'lda aniqlash ko'p vaqt talab qiladi va inson xatoliklariga olib kelishi mumkin.

Maqsad:

* Yo'l yoriqlarini avtomatik aniqlash
* Pixel-level segmentatsiya yaratish
* Yoriq maydonini hisoblash
* Yo'l holatini baholash

---

## 📊 Dataset

* Source: Roboflow Universe
* Task: Road Crack Segmentation
* Annotation Type: Segmentation Masks
* Image Size: 640×640

### Preprocessing

* Auto Orient
* Resize 640×640

### Augmentation

* Horizontal Flip
* Vertical Flip
* Rotation ±15°
* Brightness ±20%
* Gaussian Blur
* Random Noise

---

## 🤖 Model

### Architecture

* YOLOv11n-seg

### Training Configuration

* Epochs: 50
* Batch Size: 16
* Image Size: 640
* Optimizer: SGD
* Pretrained: COCO

---

## 📈 Evaluation Metrics

| Metric    | Value |
| --------- | ----- |
| mAP50     | TBD   |
| mAP50-95  | TBD   |
| Precision | TBD   |
| Recall    | TBD   |
| F1 Score  | TBD   |

---

## 🔬 Output Analysis

Model quyidagi ma'lumotlarni qaytaradi:

* Crack Count
* Crack Area (Pixels)
* Segmentation Mask
* Severity Level

### Severity Levels

| Crack Area      | Severity |
| --------------- | -------- |
| < 5000 px       | Low      |
| 5000 - 15000 px | Medium   |
| > 15000 px      | High     |

---

## 🖼️ Example Output

```json
{
    "crack_count": 3,
    "crack_area": 12450,
    "severity": "Medium"
}
```

---

## 🚀 Deployment

### Gradio Web Application

Foydalanuvchi:

1. Rasm yuklaydi
2. Model inferens qiladi
3. Segmentatsiya natijasi ko'rsatiladi
4. Hisobot qaytariladi

### Run Locally

```bash
pip install -r requirements.txt
```

```bash
cd api
python main.py
```

---

## 💡 Future Improvements

* Video Crack Detection
* Drone Road Inspection
* Crack Length Measurement
* PDF Report Generation
* FastAPI REST API
* ONNX Deployment
* Real-Time Camera Support

---

## 📚 Technologies Used

* Python
* YOLOv11 Segmentation
* OpenCV
* Gradio
* NumPy
* Roboflow
* PyTorch
* Git & GitHub

---

## 👨‍💻 Author

Rasulbek Ruzmetov

AI / Machine Learning Engineer

Computer Vision • Deep Learning • Data Science
