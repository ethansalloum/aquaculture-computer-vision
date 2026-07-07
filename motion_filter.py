import cv2
import numpy as np

def main():
    # Initialize camera source (0 for built-in webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # ---------------------------------------------------------
    # MOTION DETECTION SETTINGS (Tweak these for your environment)
    # ---------------------------------------------------------
    # Minimum area of change (in pixels) to consider it a "fish" instead of noise
    MOTION_THRESHOLD_AREA = 5000  
    
    # Stores our static baseline background frame
    background_frame = None

    print("Camera running. Press 'q' to quit, 'r' to reset baseline background.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Pre-process the frame for math operations
        # Convert to grayscale (color doesn't matter for raw motion detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply a slight blur to smooth out camera static/sensor noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 2. Establish the initial background baseline
        if background_frame is None:
            background_frame = gray
            continue

        # 3. Calculate the absolute difference between current frame and baseline
        # This outputs an image where pixels that changed are bright, and static ones are black
        frame_delta = cv2.absdiff(background_frame, gray)
        
        # 4. Threshold the delta image to force pixels to be either pure black or pure white
        # Anything that changed significantly becomes white (255)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        
        # Dilate the white spots to fill in small holes/gaps inside a moving object
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 5. Find the outlines (contours) of the white shapes
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False

        # 6. Evaluate the size of the moving shapes
        for contour in contours:
            # Skip the contour if it's too tiny (like floating debris or water bubbles)
            if cv2.contourArea(contour) < MOTION_THRESHOLD_AREA:
                continue
                
            # If we reach this point, an object larger than our threshold is moving!
            motion_detected = True
            
            # Optional: Draw a green bounding box around the motion area on the screen
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ---------------------------------------------------------
        # THE HAND-OFF ZONE
        # ---------------------------------------------------------
        if motion_detected:
            cv2.putText(frame, "MOTION DETECTED - RUNNING YOLOv8", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # >>> THIS IS WHERE YOUR JETSON NANO WILL PASS 'frame' TO YOLOv8 <<<
            # results = model(frame)
            # process_sea_lice(results)
        else:
            cv2.putText(frame, "System Idle (No Motion)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the visual streams
        cv2.imshow("Live Feed (With Motion Boxes)", frame)
        cv2.imshow("What the Computer Sees (Threshold)", thresh)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # If light conditions change, press 'r' to take a new baseline snapshot
            background_frame = None
            print("Baseline background reset!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()