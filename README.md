# 🚗 Auto Damage Segmentation

Mashina qismlarini avtomatik aniqlash va segmentatsiya qilish tizimi.
Sug'urta va ta'mirlash markazlari uchun shikastlangan qismlarni tezkor aniqlash.

## 🔗 Links

| | Link |
|---|---|
| 🤗 Live Demo | [Hugging Face Space](https://huggingface.co/spaces/Sarvarbek13/auto-damage-segmentation) |
| 📦 Dataset | [Roboflow Car Parts](https://universe.roboflow.com/person-detector/car-parts-segmentation) |
| 💻 GitHub | [CV Final Exam Repo](https://github.com/Sarvarbek-Erniyazov/CV_final-exam_auto-damage-segmentation) |

## 🧠 Pipeline

Roboflow Dataset → YOLOv8-seg Fine-tuning → ONNX Export → YOLO + SAM → FastAPI + Gradio

## 📦 Texnologiyalar

- YOLOv8-seg — instance segmentation (fine-tuned)
- SAM — Segment Anything Model (Meta AI)
- FastAPI — lokal REST API
- Gradio — web demo (Hugging Face)
- ONNX — production deployment format
- Roboflow — dataset management

## 📁 Struktura

    auto-damage-segmentation/
    ├── api/
    │   ├── main.py
    │   └── requirements.txt
    ├── model/
    │   ├── best.pt
    │   └── best.onnx
    ├── notebook/
    │   └── train.ipynb
    ├── report/
    │   └── hisobot.md
    └── README.md

## 📊 Metrikalar

| Metrika | Qiymat |
|---------|--------|
| mAP50 | ~0.73 |
| Mask mAP50 | ~0.73 |
| Precision | ~0.66 |
| Recall | ~0.79 |
| Classes | 19 |
| Dataset | 1503 rasm |

## 🧪 Demo Natijalari (YOLO + SAM)

| Rasm | Detections | SAM Masks |
|------|-----------|-----------|
| Ezilgan mashina | 8 | 8 |
| Orqa zarar | 3 | 3 |
| Kuchli zarar | 8 | 8 |

# Auto Damage Segmentation — Hisobot

## Muammo
Avtomobil qismlarini avtomatik aniqlash va segmentatsiya qilish.
Sug'urta kompaniyalari va ta'mirlash markazlari uchun shikastlangan qismlarni
tezkor aniqlash tizimi.

## Yechim
YOLOv8-seg + SAM (Segment Anything Model) pipeline asosida
instance segmentation tizimi yaratildi va FastAPI orqali deploy qilindi.

### Pipeline
YOLO (bounding box) → SAM (aniq pixel-level mask) → FastAPI (JSON response)

## Dataset
- Manba: Roboflow Universe (Car Parts Segmentation)
- Asl rasmlar: 603
- Augmentatsiyadan keyin: 1503 (3x)
- Classlar soni: 19
- Train/Val/Test: 1350 / 75 / 78
- Preprocessing: Auto-Orient, Resize 640x640
- Augmentation: Flip, Rotation ±15°, Brightness ±25%, Blur 2px, Noise 2%

## Model
- Arxitektura: YOLOv8n-seg (pretrained COCO → fine-tuned Car Parts)
- Epochs: 50
- Image size: 640x640
- Export: ONNX (12.7 MB)
- SAM: sam_b.pt (Meta AI)

## Metrikalar (Validation)
- mAP50: ~0.73
- Mask mAP50: ~0.73
- Precision: ~0.66
- Recall: ~0.79

## Demo Natijalari

### SAMsiz (YOLO only — best.pt)
| Rasm | Aniqlangan | Top detection |
|------|-----------|---------------|
| Rasm 1 (ezilgan mashina) | 7 ta | front_glass 94.3% |
| Rasm 2 (daraxt ostida) | 4 ta | tailgate 50.2% |
| Rasm 3 (kuchli zarar) | 9 ta | back_right_door 97.7% |

### SAM bilan (YOLO + SAM pipeline — best.onnx)
| Rasm | YOLO detections | SAM masks | Top detection |
|------|----------------|-----------|---------------|
| Rasm 1 (ezilgan mashina) | 8 ta | **8 ta mask** | right_mirror 86.3% |
| Rasm 2 (orqa zarar) | 3 ta | **3 ta mask** | back_glass 60.4%, tailgate 58.9% |
| Rasm 3 (kuchli zarar) | 8 ta | **8 ta mask** | back_right_door 97.7%, wheel 91.9% |

### SAM qo'shilishi bilan nima yaxshilandi
- YOLO faqat bounding box beradi
- SAM har bir qism uchun pixel-level aniq mask chiqaradi
- Deployment: best.pt → best.onnx (production ready, 20% kichikroq)

## Deployment
- **Roboflow**: dataset versiyalash va boshqaruv
- **FastAPI lokal**: /predict endpoint — rasm → JSON
- **ONNX**: cross-platform, CPU/GPU da ishlaydi
- **SAM**: Meta AI segment anything model

## Texnologiyalar
- YOLOv8-seg (Ultralytics)
- SAM — Segment Anything Model (Meta AI)
- FastAPI + Uvicorn
- ONNX Runtime
- Roboflow
- Python 3.11