# aquaculture-computer-vision
Repository for ECE 499 Capstone Project - Lilypad-Mounted Multi-Modal Sensing for Net Pens and Shellfish Leases

**File Overview**

***main.py***

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

***calc_lengths.py***

This file assists the main file with 2 important functions. 

*get_bboxes*

1) Loop through each YOLO detection result object
2) Check if frame contains bounding box prediction and assigned tracking identity
3) Extract bounding box coordinates from memory and turn into NumPy array
4) Give this fish a unique ID and pair it with its bounding box
5) Record this fish and its corresponding bounding box to later calculate its length

*get_physical_lengths*

1) Loop through each fish ID and its corresponding bounding box history
2) If the frame threshold is passed, loop through all bounding box instances and find the largest
3) Find the largest found bounding box to most accurately measure its real length
4) Convert pixel measurements to centimeters (currently 1 pixel = 0.12 cm)

***yolo_model.py***

This file helps fine-tune a baseline YOLOv8 model with specific training hyperparameters.

1) Load pretrained YOLOv8 network
2) Execute model training by setting specific hyperparameters such as epoch count, batch size, learning rates, colour augmentation, and more
3) Out of all the training runs, find the most accurate one and save it as best_model
4) Test the "best" model with a fresh set of images and save successful predictions

***dataset.yaml***

Acts as the bridge between the folder directory and the YOLO model during training. It points the training pipeline to training or validation images, and identifies the type of object being identified (0 for salmon, 1 for salmon with sea lice, 2 for biomass, etc)

**Hardware**

In a real-world implementation, we would use an NVIDIA Jetson Orin Nano Super in combination with either a PoE or USB camera.

Configuring the board - Ultralytics and Jetpack 7.2 must be installed. See setup guides below

https://docs.ultralytics.com/guides/nvidia-jetson#what-is-nvidia-jetson
https://developer.nvidia.com/embedded/jetpack

**Dataset**

The used to identify salmon is from NorFisk: https://dataverse.no/file.xhtml?persistentId=doi:10.18710/H5G3K5/QK355D&version=1.1

It contains about 2.5 GB of salmon pictures.

**Clarifications**

1) Parasite detection to be implemented in the future

Our computer vision work only includes identifying and recording lengths and observation counts of healthy salmon. No sea lice/parasite dataset could be found, and more code must be written once enough of this data is found or captured. We would want the model to save images of fish with sever parasite infestations so the farmer could have more than a simple lice count.

2) Why calculate salmon lengths? How do we make this measurement more accurate?

Calculating the sizes of identified salmon is important for parasite detection and health modeling. For example, a large fish with 2 parasites is manageable and not an immediate concern thanks to its immune system. However, a smaller fish with 2 parasites is at greater risk. Farm operators can get a better understanding of the severity of the situation rather than just seeing a total number of parasites.

The method we have implemented to measure the length of a fish is not currently entirely accurate. Some measures are taken to help this concern by using a minimum pixel threshold that the fish must surpass for the model to activate. However, once this threshold is reached, the fish can swim closer and closer to the camera, increasing its size measurement. The current conversion of pixels to centimeters is a 1:0.12 ratio, so an average 70 cm salmon would take up around 580 of the 640 pixels wide. 

Ways to get more accurate measurements include:
  - A. Constrained target zone - Set up LEDs to only capture fish within a 1 to 1.5 meter distance from the camera. Fish further than this would be too dark to track
  - B. Dual cameras - With two cameras with known focal lengths are set up a certain distance away, math can be completed to dynamically adjust the pixel ratio of        each frame
  - C. Depth-sensing cameras or underwater laser scalers can project light onto the fish, providing a depth map and a distance to the fish

3) Risk of processing constraints

Currently with the healthy salmon scanning in place, the Jetson would have very minimal performance issues identifying up to several dozens of fish because it can get away with using a low 640x480 resolution. When the parasite detection is set up, the camera quality should be upgraded to at least 1080p to look for the few pixels containing parasites. This change will inevitably lower performance.

A few tricks to increase performance during parasite detection are below.
  - A. Enable parasite detection only for fish closer to the camera
  - B. Convert YOLO weights to TensorRT. It is an SDK that optimizes PyTorch models, and should theoretically double FPS and cut VRAM usage in half
  - C. Frame skipping - Run YOLO every 2 or 3 frames, it will still be smooth
