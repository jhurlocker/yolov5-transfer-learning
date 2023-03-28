# YOLOv5 Transfer Learning on RHODS

This repository provides instructions and example on how to use Transfer Learning to adjust YOLOv5 to recognize a custom set of images.

## Introduction

### YOLO and YOLOv5

YOLO (You Only Look Once) is a popular object detection and image segmentation model developed by Joseph Redmon and Ali Farhadi at the University of Washington. The first version of YOLO was released in 2015 and quickly gained popularity due to its high speed and accuracy.

YOLOv2 was released in 2016 and improved upon the original model by incorporating batch normalization, anchor boxes, and dimension clusters. YOLOv3 was released in 2018 and further improved the model's performance by using a more efficient backbone network, adding a feature pyramid, and making use of focal loss.

In 2020, YOLOv4 was released which introduced a number of innovations such as the use of Mosaic data augmentation, a new anchor-free detection head, and a new loss function.

In 2021, Ultralytics released [YOLOv5](https://github.com/ultralytics/yolov5/), which further improved the model's performance and added new features such as support for panoptic segmentation and object tracking.

YOLO has been widely used in a variety of applications, including autonomous vehicles, security and surveillance, and medical imaging. It has also been used to win several competitions, such as the COCO Object Detection Challenge and the DOTA Object Detection Challenge.

### Transfer Learning

Transfer learning is a machine learning technique in which a model trained on one task is repurposed or adapted to another related task. Instead of training a new model from scratch, transfer learning allows the use of a pre-trained model as a starting point, which can significantly reduce the amount of data and computing resources needed for training.

The idea behind transfer learning is that the knowledge gained by a model while solving one task can be applied to a new task, provided that the two tasks are similar in some way. By leveraging pre-trained models, transfer learning has become a powerful tool for solving a wide range of problems in various domains, including natural language processing, computer vision, and speech recognition.

Ultralytics have fully integrated the transfer learning process in YOLOv5, making it easy for us to do. Let's go!

## Environment and prerequisites

- This training should be done in a **Data Science Project** (see below why).
- YOLOv5 is using **PyTorch**, so in RHODS it's better to start with a notebook image already including this library, rather than having to install it afterwards.
- PyTorch is internally using shared memory (`/dev/shm`) to exchange data between its internal worker processes. However, default container engine configurations limit this memory to the bare minimum, which can make the process exhaust this memory and crash. The solution is to manually increase this memory by mounting a specific volume with enough space at this emplacement.
  - In your Data

Finally, a GPU is strongly recommended for this type of training.

## Prepare the DataSet

To train the model we will of course need some data. In this case a sufficient number of images for the various classes we want to recognize, along with their labels and the definitions of the bounding boxes for the object we want to detect.

In this example we will use images from [Google's Open Images](https://storage.googleapis.com/openimages/web/index.html). We will work with 3 classes: **Bicycle**, **Car** and **Traffic sign**.

We have selected only a few classes in this example to speed up the process, but of course feel free to adapt and choose the ones you want.

For this first step, open the notebook `01-data_preparation.ipynb` and follow thee instructions.

