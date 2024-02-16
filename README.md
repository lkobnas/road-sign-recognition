# Real-time Traffic Sign Detection

![banner](/res/demo-full.gif)


## Content
- [<ins>Background</ins>](#background)
- [<ins>Object Detection and Recognition</ins>](#object-detection-and-recognition)
- [<ins>Transfer learning with SSD-MobileNet</ins>](#transfer-learning-with-ssd-mobilenet)
- [<ins>Getting Started</ins>](#getting-started)


## Background

This project aims to develop a deep learning-based system for detecting and recognizing traffic signs in Nvidia's JetBot. The object recognition program uses a Convolutional Neural Network (CNN) to accurately classify various types of traffic signs, such as speed limit signs, stop signs, traffic lights.

## Object Detection and Recognition

We used [jetson-inference](https://github.com/dusty-nv/jetson-inference) as the foundation, which is a high-performance repository that utilizes NVIDIA TensorRT to deploy neural networks onto the Jetson Nano development board. We utilize the detectNet library from the DNN vision library to achieve real-time object detection and image object detection.

SSD-MobileNet-V2 is a deep neural network architecture that is designed for real-time object detection. It is a variant of the popular MobileNet architecture, which was introduced in 2017 and has since become a widely-used framework for mobile object detection. It can run on mobile devices and process high-resolution images (up to 1080p) at frame rates of up to 30 fps with high detection accuracy and precision.

![example-mobilenet](/res/example-mobilenet.jpg)

## Transfer learning with SSD-MobileNet

To start our traffic sign recognition program, we utilized transfer learning by using the SSD-MobileNet-V2 architecture. This deep neural network architecture, initially designed for real-time object detection, proved to be highly effective in our application. By training our traffic sign recognition program with the SSD-MobileNet-V2 model, we were able to use the pre-trained weights and knowledge from object detection tasks, benefiting from the network's ability to detect objects accurately and efficiently.


## Getting Started

### Data Collection
To begin the process, we first prepared prototypes of road signs and traffic lights. From left to right, we have a stop sign, two speed limit signs, a parking sign, and two traffic lights.

![Road Signs](/res/signs.jpg)

Next, create a new folderin order to train our custom object detection model, we also need to crate a label file that defined the class labels for our dataset. Check this [**MEGA useful link**](https://www.youtube.com/watch?v=2XMkPW_sIGg) for step-by-step tutorial!

![Label](/res/label.png)

Follow the above tutorial for using Data Capture Control tool. This tool enabled us to capture video feeds, freeze the image and manually labeling each individual object by enclosing it with a colored rectangle. Try to capture the images from different orientations, camera perspectives, lighting conditions, and backgrounds. This approach was crucial in constructing a robust model that could effectively handle environmental noise and adapt to changes in the surrounding conditions.

Make sure all the label file and images are saved in a new folder, for example:
```
/jetson-inference/python/training/detection/ssd/data/Road_signs
```

### Training

Once we completed the labeling process for all the captured images, we proceeded to retrain the ssd-mobilenet object detection model using our custom dataset.

Run
```
cd jetson-inference/python/training/detection/ssd
python3 train_ssd.py --dataset-type=voc \
--data=data/Road_signs --model-dir=models/Road_signs \
--epochs=30
```

Since detectNet uses Open Neural Network Exchange (ONNX) as the model format, we need to convert our trained model from PyTorch to ONNX through the command below:
```
python3 onnx_export.py --model-dir=models/Road_signs
```

### Results

To check the model performance, run:
```
detectnet --model=models/Road_signs/ssd-mobilenet.onnx --labels=models/Road_signs/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            csi://0
```

![]()

## Reference
[Collecting your own Detection Datasets](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md)
[Training Object Detection Models](https://www.youtube.com/watch?v=2XMkPW_sIGg)


