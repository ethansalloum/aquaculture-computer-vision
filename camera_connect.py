import cv2
import sys

def main():
    # ---------------------------------------------------------
    # 1. INITIALIZE THE CAMERA
    # ---------------------------------------------------------
    # Use 0 for a built-in webcam or a standard USB webcam.
    # For a CSI camera connected directly to a Jetson Nano, you 
    # would replace this with a GStreamer pipeline string instead.
    camera_source = 0 
    
    print("Initializing camera feed... Press 'q' to quit.")
    cap = cv2.VideoCapture(camera_source)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {camera_source}")
        sys.exit(1)
        
    # Optional: Set resolution (adjust to match your camera's capability)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ---------------------------------------------------------
    # 2. THE MAIN VIDEO PROCESSING LOOP
    # ---------------------------------------------------------
    while True:
        # Capture frame-by-frame
        # 'ret' is a boolean (True if frame read successfully, False otherwise)
        # 'frame' is the actual image matrix (NumPy array)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab a frame. Exiting loop.")
            break
            
        # -----------------------------------------------------
        # [PLACEHOLDER ZONE FOR YOUR CODE]
        # This is where your custom pre-processing or YOLO model 
        # calls will live later. Example:
        # results = model(frame)
        # -----------------------------------------------------

        # Display the resulting frame in a window named 'Aquaculture Camera'
        cv2.imshow('Aquaculture Camera', frame)
        
        # -----------------------------------------------------
        # 3. BREAK CONDITION (Press 'q' to Exit)
        # -----------------------------------------------------
        # cv2.waitKey(1) waits 1 millisecond for a keypress. 
        # & 0xFF isolates the specific character code on 64-bit systems.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting camera feed...")
            break

    # ---------------------------------------------------------
    # 4. CLEANUP AND RELEASE RESOURCES
    # ---------------------------------------------------------
    # This is critical! If you don't release the hardware, the camera 
    # lock stays active, and your system won't be able to access it next time.
    cap.release()
    cv2.destroyAllWindows()
    print("Camera hardware released successfully.")

if __name__ == "__main__":
    main()