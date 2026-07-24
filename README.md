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
  1) 
