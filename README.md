# Theia
Tired of not being able to see through walls? Want to detect presence without cameras? Theia lets you do just that! Using ESP32's WiFi signals and machine learning, we finally achieve somewhat X-ray vision.

## Features
- Real-time presence detection using ESP32 WiFi CSI data
- Convolutional Neural Network (CNN) based classification
- Data preprocessing pipeline for CSI signals  
- Training and validation visualization tools

## Model Architecture 
Our neural network uses a sophisticated multi-task architecture designed specifically for processing WiFi CSI data:

The Input Layer takes raw CSI amplitude data from 128 WiFi subcarriers sampled across 50 time windows. The data first passes through batch normalization to ensure stable training.

Next is our Feature Extraction backbone consisting of 3 convolutional blocks with progressively increasing filters (16, 32, and 64). Each block contains a 2D convolution with 3x3 kernels, followed by batch normalization, ReLU activation, and 2x2 max pooling. We add a dropout layer (0.2) after the final convolution to prevent overfitting.

The network then splits into two task-specific branches:
- The Presence Detection branch flattens the features and passes them through a Dense layer with 32 units and ReLU activation, followed by a binary classification output with sigmoid activation
- The Location Estimation branch also flattens but uses two Dense layers (64 and 32 units) with ReLU before outputting 2D coordinates

This dual-task architecture allows simultaneous training of both presence detection and location estimation, improving overall performance through shared feature learning.

## Usage
1. Collect CSI data from ESP32
2. Process the data using the preprocessing functions  
3. Train the model using the provided notebook
4. Deploy the trained model for real-time detection
