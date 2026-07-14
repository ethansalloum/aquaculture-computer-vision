# calc_lengths.py
import numpy as np

def get_bboxes(tracked_boxes, tracking_results):
    """
    Parses tracking frames and maps bounding coordinates to persistent Track IDs.
    """
    for result in tracking_results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Extract coordinates [xmin, ymin, xmax, ymax]
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                # Save the bounding box to this fish's history
                tracked_boxes[track_id].append({
                    "bbox": box
                })

def get_physical_lengths(tracked_boxes, pixel_ratio, min_frames=3):
    """
    Computes max metric length conversions based on structural bounding geometry.
    """
    max_lengths_physical = {}
    
    for track_id, instances in tracked_boxes.items():
        # Filter out short, unreliable tracking fragments
        if len(instances) < min_frames:
            continue
            
        max_pixel_diagonal = 0.0
        
        for instance in instances:
            xmin, ymin, xmax, ymax = instance["bbox"]
            width = xmax - xmin
            height = ymax - ymin
            
            # Compute Euclidean diagonal path across the object frame bounding boundaries
            # This handles fish swimming at diagonal angles across the frame
            diagonal_pixels = np.sqrt(width**2 + height**2)
            
            if diagonal_pixels > max_pixel_diagonal:
                max_pixel_diagonal = diagonal_pixels
                
        # Transform pixel width into exact physical dimensions (e.g. centimeters)
        max_lengths_physical[track_id] = max_pixel_diagonal * pixel_ratio
        
    return max_lengths_physical
