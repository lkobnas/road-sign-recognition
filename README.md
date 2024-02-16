# Real-time Traffic Sign Detection

![banner](/res/demo-full.gif)


## Contents
- [<ins>Background</ins>](#background)
- [<ins>Object Detection and Recognition</ins>](#object-detection-and-recognition)
- [<ins>Transfer learning with SSD-MobileNet</ins>](#transfer-learning-with-ssd-mobilenet)
- [<ins>Getting Started</ins>](#getting-started)


## Background

This project aims to develop a deep learning-based system for detecting and recognizing traffic signs in Nvidia's JetBot. The object recognition program uses a Convolutional Neural Network (CNN) to accurately classify various types of traffic signs, such as speed limit signs, stop signs, traffic lights.

## Object Detection and Recognition

We used the jetson-inference as the foundation, which is a high-performance repository that utilizes NVIDIA TensorRT to deploy neural networks onto the Jetson Nano development board. We utilize the detectNet library from the DNN vision library to achieve real-time object detection and image object detection.

SSD-MobileNet-V2 is a deep neural network architecture that is designed for real-time object detection. It is a variant of the popular MobileNet architecture, which was introduced in 2017 and has since become a widely-used framework for mobile object detection. It can run on mobile devices and process high-resolution images (up to 1080p) at frame rates of up to 30 fps with high detection accuracy and precision.

![example-mobilenet](/res/example-mobilenet.jpg)

## Transfer learning with SSD-MobileNet

To start our traffic sign recognition program, we utilized transfer learning by using the SSD-MobileNet-V2 architecture. This deep neural network architecture, initially designed for real-time object detection, proved to be highly effective in our application. By training our traffic sign recognition program with the SSD-MobileNet-V2 model, we were able to use the pre-trained weights and knowledge from object detection tasks, benefiting from the network's ability to detect objects accurately and efficiently.


## Getting Started

## Data Collection

## Training



