# Tuberculosis Detection from Chest X-Ray using Convolutional Neural Networks (CNN)

## 📌 Project Overview

This project is developed as part of the **Computer Vision coursework** and focuses on applying deep learning techniques for medical image analysis. The objective is to design and implement a **Convolutional Neural Network (CNN)** that can automatically classify chest X-ray images as **Tuberculosis (TB)** or **Normal**.

Tuberculosis is a serious infectious disease that primarily affects the lungs. Early detection using radiographic imaging can significantly improve treatment outcomes. This project demonstrates how computer vision can assist in **automated disease screening**, reducing dependency on manual interpretation.

---

## 🎯 Objectives

The main objectives of this project are:

* To understand and implement **image classification using CNNs**
* To apply **computer vision techniques** on medical imaging data
* To preprocess and augment X-ray images for improved model performance
* To evaluate model performance using appropriate metrics
* To build a complete pipeline from **data loading → training → evaluation → prediction**

---

## 🧠 Methodology

### 1. Dataset

* Dataset used: Tuberculosis Chest X-ray Dataset (via KaggleHub)
* Classes: Tuberculosis, Normal

The dataset consists of chest radiographs collected for research purposes and is widely used for benchmarking medical image classification models.

---

### 2. Data Preprocessing

* Image resizing to **224 × 224 pixels**
* Normalization (pixel values scaled to [0,1])
* Data augmentation: Horizontal flipping & Zoom transformations
* Train-validation split (85% training, 15% validation)

---

### 3. Model Architecture

A **Convolutional Neural Network (CNN)** is implemented with the following layers:

* Convolutional layers for feature extraction
* MaxPooling layers for dimensionality reduction
* Fully connected (Dense) layers for classification
* Dropout layer to reduce overfitting
* Sigmoid activation for binary classification

This architecture enables the model to learn important visual patterns such as lung textures, opacities, and abnormalities associated with TB.

---

### 4. Training

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Evaluation Metric: Accuracy
* Number of Epochs: 10

The model is trained using TensorFlow/Keras with real-time data augmentation.

---

### 5. Evaluation Metrics

The performance of the model is analyzed using:

* Accuracy
* Loss curves (training vs validation)
* Confusion Matrix (optional enhancement)
* Classification Report (precision, recall, F1-score)

---

### 6. Model Output

* The trained model is saved as:
  **`tuberculosis_model.h5`**

This model can be reused for inference or integrated into applications.

---

## ⚙️ Requirements

* Python 3.8+
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* OpenCV (cv2)
* Pillow
* Scikit-learn
* KaggleHub
* ipywidgets (optional)

---

## 🚀 Installation

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

pip install --upgrade pip
pip install tensorflow scikit-learn opencv-python pillow matplotlib seaborn kagglehub ipywidgets
```

---

## ▶️ Usage

1. Ensure Kaggle access is configured (if required by KaggleHub)

2. Run the main script:

```bash
python TBdetection.py
```

---

## 🔍 Workflow

The script performs the following steps:

* Downloads dataset using KaggleHub
* Loads and preprocesses images
* Applies data augmentation
* Trains the CNN model
* Evaluates performance
* Saves trained model
* Plots accuracy and loss graphs

---

## ⚠️ Assumptions and Limitations

* Dataset paths are hardcoded and may need modification depending on the system
* Model is relatively simple and may not achieve clinical-level accuracy
* Limited epochs and architecture restrict performance

---

## 🔧 Future Improvements

* Use **Transfer Learning** (e.g., MobileNetV2, ResNet50)
* Hyperparameter tuning
* Class imbalance handling
* Deployment using Flask or web dashboard
* Integration with IoT-based healthcare systems

---

## 🎓 Academic Relevance

This project demonstrates key concepts from the **Computer Vision domain**, including:

* Image preprocessing and augmentation
* Feature extraction using CNNs
* Model training and evaluation
* Application of AI in healthcare

It serves as a practical implementation of theoretical concepts taught in the course.




