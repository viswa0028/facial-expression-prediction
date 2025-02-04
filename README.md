# Facial Expression Detection using FER2013

## Overview
This project implements a **Facial Expression Detection** system using deep learning techniques. The model is trained on the **FER2013** dataset to classify images into different facial expressions. The system processes input images, extracts facial features, and predicts emotions with high accuracy.

## Dataset
The **FER2013** (Facial Expression Recognition 2013) dataset consists of grayscale images (48x48 pixels) categorized into seven emotions:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

## Features
- **Preprocessing**: Image resizing, normalization, and data augmentation for improved generalization.
- **Model**: Convolutional Neural Network (CNN) designed for emotion classification.
- **Training**: The model is trained on FER2013 with optimized loss functions and hyperparameters.
- **Real-time Detection**: Uses OpenCV to capture and classify facial expressions from a webcam.
- **Visualization**: Displays real-time predictions with bounding boxes around detected faces.

## Model Architechture
- **Input Layer**: 48x48 grayscale images
- **CNN Layers**: Convolution, Batch Normalization, ReLU, and Pooling layers
- **Fully Connected Layers**: For classification
## Dependencies
Ensure you have the following installed:
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib


