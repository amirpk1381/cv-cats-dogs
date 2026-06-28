# Cats vs Dogs Image Classification using PyTorch

A robust Computer Vision project implementing a Deep Learning pipeline to classify images of cats and dogs. This project focuses on end-to-end data preprocessing, deep Convolutional Neural Network (CNN) training, and evaluating model generalization on unseen validation data using PyTorch.

## 📊 Project Highlights
 Framework: Fully built using PyTorch and `torchvision`.
 *Data Pipeline: Implemented dynamic data augmentation (resizing, cropping, horizontal flips) and normalization to prevent overfitting.
 *Evaluation Metrics: Monitored Training/Validation Loss and Accuracy curves to find the optimal training checkpoint.

---

## 🏗️ Technical Workflow

1. Data Pipeline & Augmentation:
    Utilized `torchvision.transforms` for on-the-fly image manipulation.
    Loaded data efficiently using PyTorch `Dataset` and `DataLoader` with optimized batch sizes and multi-process data loading (`num_workers`).

2. Model Architecture:
    Built and experimented with customized CNN architectures / transfer learning backbones (`ResNet` / `AlexNet`).
    Designed proper Fully Connected (FC) head layers matching the binary classification output (2 classes).

3. Training & Optimization:
    *Loss Function: Used `CrossEntropyLoss` (or `BCEWithLogitsLoss`) for stable binary classification.
    *Optimizer: Optimized parameters using `Adam` / `SGD` with tuned learning rates.
    *Hardware Acceleration: Fully integrated CUDA support for fast GPU-accelerated training loops.

---

## 📈 Learning Curves & Progress
During training, the model's accuracy steadily improved while convergence was audited by plotting loss curves. The pipeline demonstrates strong capabilities in feature extraction—distinguishing subtle pet features such as ears, snout shapes, and fur textures.

---

## 🛠️ Requirements
 Python 3.x
 PyTorch
 Torchvision
 Matplotlib (for plotting learning curves)
 NumPy
