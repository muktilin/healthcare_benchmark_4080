import numpy as np
import torch
from ultralytics import YOLO
from boxmot import create_tracker
from pathlib import Path

def iou_batch(bb_test, bb_gt):
    """
    Computes the IoU matrix between two sets of BBoxes.
    bb_test: (N, 4) [x1, y1, x2, y2]
    bb_gt:   (M, 4) [x1, y1, x2, y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                      
    return o

class BoxMOTTracker:
    def __init__(
        self,
        gpu_id=0,
        tracker_type='botsort',
        reid_weights='osnet_x1_0_msmt17.pt',
        detector_weights='yolo11x-pose.pt',
    ):
        self.device = torch.device(f'cuda:{gpu_id}')
        
        print(f"[GPU {gpu_id}] Initializing BoxMOT ({tracker_type}) + YOLO11-Pose...")
        
        # 1. Load Pose model
        # Ensure weights exist or can be auto-downloaded
        self.detector = YOLO(detector_weights) 
        
        # 2. Load BoxMOT tracker
        self.tracker = create_tracker(
            tracker_type=tracker_type,
            tracker_config=None, # Use default configuration
            reid_weights=Path(reid_weights),
            device=self.device,
            half=True,
            per_class=False 
        )
        
    def process_frame(self, frame):
        """
        Processes a single frame.
        Returns: List[Dict] -> [{track_id, bbox, keypoints, conf}, ...]
        """
        # A. YOLO-Pose Inference
        results = self.detector(frame, classes=[0], conf=0.4, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return []
            
        # Raw detection data
        dets = results[0].boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        
        # Raw skeleton/keypoint data (N, 17, 3)
        if results[0].keypoints is not None:
            raw_kpts = results[0].keypoints.data.cpu().numpy()
        else:
            raw_kpts = np.zeros((len(dets), 17, 3))
        
        # B. BoxMOT Tracking
        # update returns: [x1, y1, x2, y2, id, conf, cls, ind]
        tracker_outputs = self.tracker.update(dets, frame)
        
        if len(tracker_outputs) == 0:
            return []
            
        track_data = []
        
        # C. Keypoints Matching (via IoU)
        tracked_boxes = tracker_outputs[:, :4]
        detected_boxes = dets[:, :4]
        
        iou_matrix = iou_batch(tracked_boxes, detected_boxes)
        # Find the index of the detection box with the maximum IoU for each track box
        matched_indices = np.argmax(iou_matrix, axis=1)
        
        for i, out in enumerate(tracker_outputs):
            x1, y1, x2, y2 = map(int, out[:4])
            track_id = int(out[4])
            conf = float(out[5])
            
            # Retrieve the corresponding skeleton
            det_idx = matched_indices[i]
            # Only consider it a successful match if IoU is high enough
            if iou_matrix[i, det_idx] > 0.5:
                kpts = raw_kpts[det_idx]
            else:
                # If matching fails, provide an empty skeleton
                kpts = np.zeros((17, 3)) 
            
            track_data.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "keypoints": np.round(kpts, 2).tolist(), # Convert to list for easy serialization
                "conf": conf
            })
            
        return track_data
