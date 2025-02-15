# Theia

Tired of not being able to see through walls? Want to detect presence without cameras? Theia lets you do just that! Using ESP32's WiFi signals and machine learning, this project turns your space into a smart sensing environment - no cameras or motion sensors needed.

## Features

- Real-time presence detection using ESP32 WiFi CSI data
- Convolutional Neural Network (CNN) based classification
- Data preprocessing pipeline for CSI signals
- Training and validation visualization tools

## Technical Stack

- TensorFlow for deep learning
- Python for data processing
- ESP32 for CSI data collection
- scikit-learn for data preparation

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- scikit-learn
- matplotlib

## Usage

1. Collect CSI data from ESP32
2. Process the data using the preprocessing functions
3. Train the model using the provided notebook
4. Deploy the trained model for real-time detection

## Model Architecture

The system uses a CNN architecture specifically designed for ESP32 CSI data, featuring:

- Multiple convolutional layers
- Batch normalization
- Dropout for regularization
- Binary classification output (presence/absence)
