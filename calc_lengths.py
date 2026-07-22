# calc_lengths.py
import numpy as np

def get_bboxes(tracked_boxes, tracking_results):                        #parses tracking frames and maps bounding coordinates to persistent Track IDs.
    for result in tracking_results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()                     #extract coordinates [xmin, ymin, xmax, ymax]
            track_ids = result.boxes.id.cpu().numpy().astype(int)       #give specific fish in the frame an ID
            
            for box, track_id in zip(boxes, track_ids):                 #zip pairs the bounding box with its track_id
                tracked_boxes[track_id].append({
                    "bbox": box                                         #save current bounding box coordinates to this fish's history
                })

def get_physical_lengths(tracked_boxes, pixel_ratio, min_frames=3):     #computes max metric length conversions based on structural bounding geometry.
    max_lengths_physical = {}
    for track_id, instances in tracked_boxes.items():
        if len(instances) < min_frames:                                 #filter out short, unreliable tracking fragments
            continue
            
        max_pixel_diagonal = 0.0
        
        for instance in instances:                                       #loop through every recorded bounding box instance for the fish
            xmin, ymin, xmax, ymax = instance["bbox"]
            width = xmax - xmin
            height = ymax - ymin                                         #unpack coordinates and calculate width and height
       
            diagonal_pixels = np.sqrt(width**2 + height**2)              #compute diagonal pixel length to get the most accurate largest bounding box
            
            if diagonal_pixels > max_pixel_diagonal:
                max_pixel_diagonal = diagonal_pixels
                
        max_lengths_physical[track_id] = max_pixel_diagonal * pixel_ratio            #convert pixel length to physical units
        
    return max_lengths_physical
