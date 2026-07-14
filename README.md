# aquaculture-computer-vision
Repository for ECE 499 Capstone Project - Lilypad-Mounted Multi-Modal Sensing for Net Pens and Shellfish Leases

This repository will serve as a stepping stone for the computer vision aspect of Mostar Labs' aquaculture module for their Lilypad.

The goal of our model is to:
  •	Identify sea lice and other parasites on salmon
  •	Identify other fish or marine lifeforms present in the salmon pens
  •	Detect dangerous water conditions and harmful algae blooms 

We will be using an NVIDIA Jetson Orin Nano Super to process motion detected video clips and generate object detection tracks and salmon species counts. The model to be used is YOLOv8 (https://yolov8.com/)

Ultralytics setup guide for the board with YOLOv8: https://docs.ultralytics.com/guides/nvidia-jetson#what-is-nvidia-jetson

Another setup guide from a similar project can be found here: https://github.com/Salmon-Computer-Vision/salmon-computer-vision/blob/master/utils/jetson/README.md

I think we will also need to install Jetpack (Jetpack 7.2 should work): https://developer.nvidia.com/embedded/jetpack


The dataset to be used is: https://dataverse.no/file.xhtml?persistentId=doi:10.18710/H5G3K5/QK355D&version=1.1

Steps (to be continuously updated):
  1) Set up outlines of the scripts and folders needed (done)
  2) Download the dataset (done)
  3) Train the model to correctly detect the presence of a fish in a frame.
  4) 
