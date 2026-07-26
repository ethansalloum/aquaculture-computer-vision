# aquaculture-computer-vision
Repository for ECE 499 Capstone Project - Lilypad-Mounted Multi-Modal Sensing for Net Pens and Shellfish Leases

**Hardware**

We will be using an NVIDIA Jetson Orin Nano Super to process motion detected video clips and generate object detection tracks and salmon species counts. The model to be used is YOLOv8 (https://yolov8.com/)

Ultralytics setup guide for the board with YOLOv8: https://docs.ultralytics.com/guides/nvidia-jetson#what-is-nvidia-jetson

Another setup guide from a similar project can be found here: https://github.com/Salmon-Computer-Vision/salmon-computer-vision/blob/master/utils/jetson/README.md

I think we will also need to install Jetpack (Jetpack 7.2 should work): https://developer.nvidia.com/embedded/jetpack

**Dataset**

The dataset to be used is NorFisk: https://dataverse.no/file.xhtml?persistentId=doi:10.18710/H5G3K5/QK355D&version=1.1

It contains about 2.5 GB of salmon pictures.

**Repository structure and Files Overview**

*main.py*

The main file stitches everything together from the YOLO model to the hardware. In order, it does the following:
  1) Set a pixel and frame threshold to activate the model
  2) Open camera stream with OpenCV library (640 x 480 pixels)
  3) Temporarily convert image to grayscale for motion detection
  4) If enough pixels in the frames change, colour those pixels white and trace its outline, wake up YOLO if the shape is big enough
  5) Draw a green box around the outline of the fish
  6) Hand off active frame to YOLO to detect fish and give it an ID
  7) Save bounding box history for every detected ID
  8) Show real-time feed to the user with overlaid status updates ("Tracking Engine Live" or "System Idle")
  9) Recalibrate or shut down camera stream from the users input
  10) Post-processing: Compute analytics upon shutdown (fish lengths and observation count)
  11) Export analytics to a csv file

*calc_lengths.py*

This file
