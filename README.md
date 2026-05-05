---
title: Zindi Lacuna Malaria Detection
emoji: 🦟
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Zindi Lacuna Malaria Detection Challenge
https://www.kaggle.com/code/noob786/zindi-lacuna-malaria-detection-challenge-ai

This project implements a **YOLO11x** object detection model to identify malaria parasites and white blood cells in blood smear images as part of the [Zindi Lacuna Malaria Detection Challenge](https://zindi.africa/competitions/lacuna-malaria-detection-challenge). The model is designed to assist healthcare professionals by automatically detecting and classifying **Trophozoite** (malaria parasite), **White Blood Cells (WBC)**, and **Negative** regions in microscopy images.

## About the Project

Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites. Rapid and accurate detection of malaria parasites in blood samples is critical for effective treatment. This project leverages state-of-the-art YOLO11x (Ultralytics) object detection, trained on a multi‑GPU setup, to deliver robust performance even with challenging microscopy data.

### Key Features
- **Modern Architecture**: YOLO11x – the latest YOLO model with improved accuracy and efficiency.
- **Multi‑GPU Training**: Utilized 4× NVIDIA L4 GPUs with Distributed Data Parallel (DDP) for faster training.
- **Advanced Augmentations**: Mosaic, mixup, copy-paste, random perspective, HSV shifts, and more.
- **Early Stopping**: Patience of 8 epochs to avoid overfitting.
- **Deployment Ready**: Can be integrated into diagnostic support systems via the provided Gradio app.

## Model Performance

The best model was achieved at **epoch 34** (early stopping triggered after 42 total epochs). Validation results on the held‑out set:

| Metric                     | Overall / Class     | Value   |
|----------------------------|---------------------|---------|
| **Precision (all classes)** |                     | 45.6%   |
| **Recall (all classes)**    |                     | 22.1%   |
| **mAP@0.5**                 |                     | 22.1%   |
| **mAP@0.5:0.95**            |                     | 10.9%   |
| **Trophozoite**             | Precision / Recall  | 54.3% / 30.3% |
|                             | mAP@0.5             | 29.3%   |
| **WBC**                     | Precision / Recall  | 82.6% / 35.8% |
|                             | mAP@0.5             | 37.0%   |
| **Negative**                | Precision / Recall  | 0.0% / 0.0% |

> **Note**: The Negative class had no true positive detections, likely due to class imbalance or definition. The model focuses on detecting parasite and WBC regions.

Training converged after 42 epochs in **0.9 hours** on 4 GPUs. The final model weights are saved as `best.pt` (114 MB).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Zindi-Lacuna-Malaria-Detection-Challenge-AI
   
Requirements
   Core dependencies (see requirements.txt for exact versions): 
   ultralytics – YOLO11 training and inference
   torch, torchvision – PyTorch with CUDA support
   opencv-python, numpy, pandas, matplotlib
   gradio – for the interactive demo app

Results & Discussion
   The YOLO11x model achieves reasonable detection of Trophozoite (mAP50 ≈ 29%) and WBC (mAP50 ≈ 37%), but struggles with the Negative class (likely due to definition or extreme class imbalance). The overall mAP50 of 22% reflects the difficulty of the task – parasites are small and often overlap with WBCs.

Future improvements could include:
   Test‑time augmentation (TTA)
   Model ensemble
   Hard negative mining
   Class‑weighted loss functions

Contributing
   Contributions are welcome! You can help by:
   Improving data augmentation strategies
   Tuning hyperparameters (learning rate, mosaic/mixup probabilities)
   Adding post‑processing heuristics
   Optimising the Gradio interface for clinical use
   Please open an issue or pull request.

License
   This project is part of the Zindi Lacuna Malaria Detection Challenge. Model weights and code are shared for research and educational purposes.

Author
   Muhammad Qasim Shabbeer

Acknowledgements
   Zindi Africa for hosting the competition and providing the dataset 
   Ultralytics for the YOLO11 framework and training utilities
   Kaggle for providing GPU resources (4× NVIDIA L4)