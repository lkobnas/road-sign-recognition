# Real-time Traffic Sign Detection - JetBot

![banner](/res/demo-full.gif)


## Content
- [<ins>Background</ins>](#background)
- [<ins>Object Detection and Recognition</ins>](#object-detection-and-recognition)
- [<ins>Transfer learning with SSD-MobileNet</ins>](#transfer-learning-with-ssd-mobilenet)
- [<ins>Getting Started</ins>](#getting-started)


## Background

This project aims to develop a deep learning-based system for detecting and recognizing traffic signs in Nvidia's JetBot. The object recognition program uses a Convolutional Neural Network (CNN) to accurately classify various types of traffic signs, such as speed limit signs, stop signs, traffic lights. Then control JetBot according to the information on different road signs.

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
```bash
$ cd jetson-inference/python/training/detection/ssd
$ python3 train_ssd.py --dataset-type=voc \
--data=data/Road_signs --model-dir=models/Road_signs \
--epochs=30
```

Since detectNet uses Open Neural Network Exchange (ONNX) as the model format, we need to convert our trained model from PyTorch to ONNX through the command below:
```bash
$ python3 onnx_export.py --model-dir=models/Road_signs
```

### Results

To check the model performance, run:
```bash
$ detectnet --model=models/Road_signs/ssd-mobilenet.onnx --labels=models/Road_signs/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            csi://0
```

![training-result](/res/training-result.jpg)

## Control JetBot

In this part, apply the trained detection model to JetBot and program JetBot to stop, speed up or slow down according to the instruction of the road models.

In order to integrate the traffic sign recognition program with the JetBot driving system, several modifications and additions are needed.

First, we accessed the training model directory and made changes to the detectnet.py file, we made a new copy called [detectnet_mod.py](/src/detectnet_mod.py), which is responsible for launching the real-time object detection program.

Add the following code for importing the JetBot vehicle control library and adding an extra argument to the detectNet command to capture the program's output for controlling JetBot:
```py
from robot import Robot
robot = Robot()
```
```py
parser.add_argument("--use_motor", action='store_true', help="enable motor for sign detection")
```
Since we were using the jetson-inference repository for this task while the JetBot library was stored in a separate repository called "jetbot," we couldn't directly use the robot driver function. To overcome this, we copied the necessary files (robot.py and motor.py) from the JetBot repository and included them in the current directory. Additionally, we added required libraries such as Adafruit_GPIO, Adafruit_MotorHAT, and Adafruit_PureIO to enable the use of the JetBot motor. The additional codes and libraries can be found [**here**](/src/ssd/), simply copy them all and paste it inside the ```ssd``` folder with the ```detectnet_mod.py``` file.

After that, we added a response function in ```detectnet_mod.py``` to decide what actions are needed to be performed when a specific traffic sign was detected.

|     Traffic sign	    |          Action           |
|:---------------------:|:-------------------------:|
| Stop sign	    	    |           Stop            |
| Speed limit 30	    |  Slow down to speed 0.15  |
| Speed limit 50	    |	Speed up to speed 0.2   |
| Parking sign		    |           Stop            |
| Red traffic light	    |	        Stop            |
| Green traffic light   |       Start moving        |


```py
robot_status = False #False : stop, True: active
robot_speed = 0.15

def start_robot(detection):
	global robot_status
	global robot_speed
	global last_stop
	if robot_status == True:
		robot.forward(robot_speed)

		if detection .ClassID == 1 or detection.ClassID == 4 or detection.ClassID == 5: #STOP_SIGN, PARKING_SIGN, RED_TRAFFIC_LIGHT
			robot_status = False

		elif detection.ClassID == 2:	#SPEED_LIMIT_30
			robot_speed = 0.1

		elif detection.ClassID == 3:	#SPEED_LIMIT_50
			robot_speed = 0.15

	elif robot_status == False:
		robot.stop()
		if detection.ClassID == 6:	#GREEN_TRAFFIC_LIGHT
			robot.forward(robot_speed)
			robot_status = True
```

Lastly, we called the response function into the for loop of the detection program so that the function can be triggered whenever the frame has been updated.

```py
for detection in detections:
    print(detection)
    if opt.use_motor:
        start_robot(detection)
```

To launch the detection program


## Reference
[Collecting your own Detection Datasets](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md) <br>
[Training Object Detection Models](https://www.youtube.com/watch?v=2XMkPW_sIGg)


