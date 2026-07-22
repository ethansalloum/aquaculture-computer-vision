import os
import cv2
import sys
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

# Importing the updated calculations module
from calc_lengths import get_bboxes, get_physical_lengths

# ---------------------------------------------------------
# EDGE HARDWARE CONFIGURATIONS
# ---------------------------------------------------------
# Set camera_source to 0 for USB webcam testing.
# For a CSI camera hooked directly to the Jetson Orin Nano, swap this 
# with your NVARGUS GStreamer pipeline string.
CAMERA_SOURCE = 0 

# Threshold paths & metrics
DIR_PATH = os.getcwd()
MODEL_PATH = os.path.join(DIR_PATH, "Program", "norfisk_run", "weights", "best.pt")        #path to YOLOv8 PyTorch model weights
OUTPUT_CSV_PATH = os.path.join(DIR_PATH, "lilypad_biomass_report.csv")                     #export csv in main folder

# Tweak these for your net-pen environment conditions
MOTION_THRESHOLD_AREA = 5000                              #size of moving object to trigger YOLO
MIN_TRACK_FRAMES = 3                                      #filter out fleeting noise artifacts
PIXELS_TO_CM_RATIO = 0.12                                 #calibration metric (1 pixel = X cm at lens focus plane)

def main():
    # 1. Initialize YOLO Model
    if not os.path.exists(MODEL_PATH):
        print(f"Custom weights not located at {MODEL_PATH}. Initializing standard yolov8n.pt...")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(MODEL_PATH)
        
    # 2. Open Camera Stream
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Hardware Error: Unable to access camera source {CAMERA_SOURCE}")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State Variables
    background_frame = None
    tracked_boxes = defaultdict(list)
    master_records = []
    
    print("\n--- LilyPad Sensing Module Online ---")
    print("Press 'q' to safely release hardware, 'r' to reset baseline background tracking.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image buffer. Exiting processing loop.")
            break
            
        # Motion Pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                               #convert colour to grayscale
        gray = cv2.GaussianBlur(gray, (21, 21), 0)                                                   #smooth image by reducing high-frequency noise and detail

        if background_frame is None:
            background_frame = gray
            continue

        # Motion Filtering
        frame_delta = cv2.absdiff(background_frame, gray)                                            #created if a pixel changes
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)                           #pixel change below 25 is discarded as ripple noise or static, make higher values white
        thresh = cv2.dilate(thresh, None, iterations=2)                                              #merge white blobs (head vs tail) into single silhouette
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    #trace outline, wakes up YOLO if big enough

        # Identify Fish
        motion_detected = False                                                                      #start fresh frame assuming no movement
        for contour in contours:                                                                     #loop through every outline
            if cv2.contourArea(contour) >= MOTION_THRESHOLD_AREA:                                    #if the blob is bigger than our fish threshold of 5000 pixels
                motion_detected = True                                                               #deploy AI tracking
                (x, y, w, h) = cv2.boundingRect(contour)                                             #draw green box around fish
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hand-off execution to YOLOv8 + ByteTrack Tracker if movement is found
        if motion_detected:
            cv2.putText(frame, "TRACKING ENGINE ACTIVE", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Run tracking directly on the single frame buffer
            results = model.track(
                source=frame,
                tracker="bytetrack.yaml",                                          #turn on tracking
                persist=True,                                                      #look at previous frames to see if the current fish matches a previous ID
                conf=0.25,
                verbose=False
            )
            
            # Map tracking identities and spatial configurations, if nothing > 5000 pixels go idle
            get_bboxes(tracked_boxes, results)
        else:
            cv2.putText(frame, "System Idle (Scanning Pens)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Show stream displays (can comment out later for headless deployment)
        cv2.imshow("LilyPad Multi-Modal Optical Stream", frame)                    #output frame to viewer
        
        if cv2.waitKey(1) & 0xFF == ord('q'):                                      #shut down at press of 'q'
            break
        elif cv2.waitKey(1) & 0xFF == ord('r'):                                    #clear background_frame and recalibrate at press of 'r'
            background_frame = None
            print("Baseline environment profile recalibrated.")

    # Post-Processing: Compute total analytics upon shutdown
    print("\nCompiling final biomass diagnostics...")
    lengths_dict = get_physical_lengths(tracked_boxes, PIXELS_TO_CM_RATIO, MIN_TRACK_FRAMES)    #convert tracked bounding box pixel dimensions to real cm lengths
    
    for fish_id, physical_len in lengths_dict.items():                             #for each fish identity, append total frame count of this fish to master_records
        master_records.append({
            "Fish_Track_ID": fish_id,
            "Calculated_Length_CM": round(physical_len, 2),
            "Observations_Count": len(tracked_boxes[fish_id])
        })
        
    if master_records:                                                             #if minimum 1 fish met the tracking threshold, convert list to a csv file
        df = pd.DataFrame(master_records)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"[LOG DATA] Log successfully structured and written to: {OUTPUT_CSV_PATH}")

    # Clean Release
    cap.release()
    cv2.destroyAllWindows()
    print("Camera locks cleanly unmounted from OS layer.")

if __name__ == '__main__':
    main()
