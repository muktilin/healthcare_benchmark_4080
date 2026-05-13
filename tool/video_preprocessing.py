import cv2
import numpy as np
import os
import glob
import math
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from tool.dbmanager import VideoDB
from tool.track import BoxMOTTracker
from tool.action import ActionRecognizer
from tool.reid import ReIDExtractor
from tool.global_matcher import GlobalMatcher  # Using the rewritten GlobalMatcher
from tool.ocr import TimeOCR
from tool.reconcile_ids import reconcile_ids

TIME_ROI = [0, 0, 600, 100] 
BUFFERING_LABELS = ["buffering", "None", "", None, "Unknown", "Buffering", "Uncertain"]

REID_INTERVAL = 5             # ReID optimization: Extract features every 5 frames
EXPAND_RATIO_ACTION = 0.2     # Action optimization: Expand BBox by 20%
ACTION_INTERVAL = 3           # Action optimization: Perform action prediction every 3 frames


def find_video_files(root_dir):
    """Recursively find and sort all video files in the specified directory"""
    extensions = ['*.mov', '*.mp4', '*.avi']
    video_files = []
    for ext in extensions:
        files = glob.glob(os.path.join(root_dir, "**", ext), recursive=True)
        video_files.extend(files)
    video_files.sort()
    return video_files

def process_dataset_folder(
    dataset_root,
    db_path,
    show_viz=False,
    gpu_id=0,
    skip_action=False,
    skip_ocr=False,
    reid_interval=REID_INTERVAL,
    action_interval=ACTION_INTERVAL,
    action_motion_threshold=0.02,
    detector_weights="yolo11x-pose.pt",
    action_model_path="OpenGVLab/InternVideo2_5_Chat_8B",
    ocr_interval_sec=1.0,
    fast_write=False,
    commit_interval=10,
):
    """Process all videos under the dataset root directory"""
    db = VideoDB(db_path, fast_write=fast_write)
    # Instantiate GlobalMatcher once here to achieve cross-video ID persistence
    id_matcher = GlobalMatcher(threshold=0.75) 
    tracker = BoxMOTTracker(gpu_id=gpu_id, tracker_type='deepocsort', detector_weights=detector_weights)
    action_rec = None if skip_action else ActionRecognizer(device=f'cuda:{gpu_id}', model_path=action_model_path)
    reid_extractor = ReIDExtractor(device=f'cuda:{gpu_id}')
    ocr_reader = None if skip_ocr else TimeOCR(use_gpu=True)

    video_list = find_video_files(dataset_root)
    
    for i, video_path in enumerate(video_list):
        filename = Path(video_path).name
        print(f"\n[{i+1}/{len(video_list)}] Processing: {filename}")
        
        
        process_single_video(
            video_path, db, tracker, action_rec, 
            reid_extractor, id_matcher, ocr_reader, 
            show_viz,
            reid_interval=reid_interval,
            action_interval=action_interval,
            skip_action=skip_action,
            skip_ocr=skip_ocr,
            action_motion_threshold=action_motion_threshold,
        )
        if (i + 1) % max(1, commit_interval) == 0:
            db.commit()
    
    db.close()

def process_single_video(
    video_path,
    db,
    tracker,
    action_rec,
    reid_extractor,
    id_matcher,
    ocr_reader,
    show_viz,
    reid_interval=REID_INTERVAL,
    action_interval=ACTION_INTERVAL,
    skip_action=False,
    skip_ocr=False,
    action_motion_threshold=0.02,
    ocr_interval_sec=1.0,
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    video_name = Path(video_path).stem 
    
    # Directory setup
    check_dir = "check_buffer"
    os.makedirs(check_dir, exist_ok=True)
    
    seen_global_ids = set()
    track_buffers = {} # DB backtracking buffer
    last_action_cache = {}
    last_bbox_cache = {}
    
    # ReID optimization parameters
    last_reid_features = {} 
    
    ocr_interval = max(1, int(fps * ocr_interval_sec))
    current_ocr_text = ""

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing", leave=False)

    while True:
        ret, frame = cap.read()
        if not ret: break
        timestamp = frame_idx / fps

        # --- A. Sampling Interval Logic ---
        is_reid_frame = (frame_idx % reid_interval == 0)
        is_action_frame = (frame_idx % action_interval == 0) 

        # --- B. OCR Recognition ---
        if not skip_ocr and ocr_reader is not None and frame_idx % ocr_interval == 0:
            text = ocr_reader.recognize(frame, roi_bbox=TIME_ROI)
            if text.strip(): 
                current_ocr_text = text
        
        # --- C. Tracking ---
        tracks_data = tracker.process_frame(frame)

        # --- D. Crop & ReID Feature Collection ---
        current_crops_reid = []      
        current_crops_action = []    
        valid_indices = []
        
        if len(tracks_data) > 0:
            for idx, item in enumerate(tracks_data):
                x1, y1, x2, y2 = map(int, item['bbox'])
                
                # 1. ReID Crop: Use strict original BBox
                r_x1, r_y1, r_x2, r_y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                crop_reid = frame[r_y1:r_y2, r_x1:r_x2]

                if crop_reid.size > 0:
                    valid_indices.append(idx)
                    
                    # 2. Action Crop: Use expanded BBox
                    w, h = x2 - x1, y2 - y1
                    nx1 = max(0, int(x1 - w * EXPAND_RATIO_ACTION))
                    ny1 = max(0, int(y1 - h * EXPAND_RATIO_ACTION))
                    nx2 = min(W, int(x2 + w * EXPAND_RATIO_ACTION))
                    ny2 = min(H, int(y2 + h * EXPAND_RATIO_ACTION))
                    crop_action = frame[ny1:ny2, nx1:nx2] 
                    
                    current_crops_reid.append(crop_reid)
                    current_crops_action.append(crop_action)
                
        # --- E. ReID Feature Extraction (Sampling Logic) ---
        reid_feats = []
        default_feat_size = 512
        default_feat = np.zeros(default_feat_size, dtype=np.float32) 
        
        if is_reid_frame and len(current_crops_reid) > 0:
            # Extract new features
            reid_feats_new = reid_extractor.extract(current_crops_reid)
            
            # Store and use new features for current loop
            for i, track_idx in enumerate(valid_indices):
                local_id = tracks_data[track_idx]['track_id']
                feat = reid_feats_new[i]
                reid_feats.append(feat)
                last_reid_features[local_id] = feat 
        else:
            # Reuse historical features (handles default_feat for new IDs during skip frames)
            for track_idx in valid_indices:
                local_id = tracks_data[track_idx]['track_id']
                feat = last_reid_features.get(local_id, default_feat) 
                reid_feats.append(feat)


        # --- F. Action Prediction & DB Entry (Triggered on Action Frames) ---
        
        if is_action_frame:
            
            current_frame_local_ids = set()

            for i, track_idx in enumerate(valid_indices):
                item = tracks_data[track_idx]
                crop_for_action = current_crops_action[i] # Use expanded crop
                feat = reid_feats[i] 
                
                local_id = item['track_id']
                current_frame_local_ids.add(local_id)
                enable_infer = True
                prev_bbox = last_bbox_cache.get(local_id)
                if prev_bbox is not None:
                    px1, py1, px2, py2 = prev_bbox
                    x1, y1, x2, y2 = map(int, item['bbox'])
                    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
                    motion = ((cx - pcx) / pw) ** 2 + ((cy - pcy) / ph) ** 2
                    if motion < action_motion_threshold:
                        enable_infer = False
                last_bbox_cache[local_id] = item['bbox']

                # Global ReID matching and Action prediction
                global_id = id_matcher.get_global_id(local_id, feat)
                if skip_action or action_rec is None:
                    action_label = "Unknown"
                else:
                    action_label = action_rec.predict(local_id, crop_for_action, enable_infer=enable_infer)
                    if (not enable_infer) and local_id in last_action_cache:
                        action_label = last_action_cache[local_id]
                    elif action_label not in BUFFERING_LABELS:
                        last_action_cache[local_id] = action_label
                
                # Construct data entry
                entry = {
                    "video_name": Path(video_path).name,
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 4),
                    "ocr_time": current_ocr_text, 
                    "person_id": global_id,
                    "action": action_label, 
                    "bbox": np.array(item['bbox']),
                    "keypoints": np.array(item['keypoints']),
                    "reid_feature": feat 
                }

                # 1. Add to buffer while waiting for stable Action results
                if local_id not in track_buffers:
                    track_buffers[local_id] = []
                track_buffers[local_id].append(entry)

                # 2. Trigger DB write via backtracking if action is valid
                is_valid_action = action_label not in BUFFERING_LABELS

                if is_valid_action:
                    # Update all buffered entries for this ID with the current stable label
                    for buffered_entry in track_buffers[local_id]:
                        buffered_entry['action'] = action_label
                        db.add_entry(buffered_entry)

                    # Clear successfully written buffer
                    track_buffers[local_id] = []

                    # Check Buffer Visualization (Save first appearance of new Global ID)
                    if global_id != -1 and global_id not in seen_global_ids:
                        seen_global_ids.add(global_id)
                        
                        check_img = frame.copy()
                        bx1, by1, bx2, by2 = map(int, item['bbox'])
                        
                        # Draw original BBox and info labels
                        cv2.rectangle(check_img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                        label_text = f"ID:{global_id} | {action_label}" 
                        text_y = by1 - 10 if by1 - 10 > 10 else by1 + 20
                        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(check_img, (bx1, text_y - text_h - 5), (bx1 + text_w, text_y + 5), (0, 0, 255), -1)
                        cv2.putText(check_img, label_text, (bx1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        save_name = f"{video_name}_frame{frame_idx:06d}_ID{global_id}.jpg"
                        save_path = os.path.join("check_buffer", save_name)
                        cv2.imwrite(save_path, check_img)

                # Real-time visualization
                if show_viz:
                    bx1, by1, bx2, by2 = map(int, item['bbox'])
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    info = f"ID:{global_id} {action_label}"
                    cv2.putText(frame, info, (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # --- G. Cleanup for disappeared entities ---
            active_ids = [t['track_id'] for t in tracks_data]
            if action_rec is not None:
                action_rec.clean(active_ids)
            
            buffered_ids = list(track_buffers.keys())
            for lid in buffered_ids:
                if lid not in active_ids:
                    # 1. Clean ReID cache
                    if lid in last_reid_features:
                        del last_reid_features[lid]
                    if lid in last_action_cache:
                        del last_action_cache[lid]
                    if lid in last_bbox_cache:
                        del last_bbox_cache[lid]
                        
                    # 2. Flush remaining DB Buffer
                    # If person disappears before a valid action is set, mark as "Uncertain"
                    leftover_entries = track_buffers[lid]
                    if leftover_entries:
                        for entry in leftover_entries:
                            entry['action'] = "Uncertain" 
                            db.add_entry(entry)
                        del track_buffers[lid]

        frame_idx += 1
        pbar.update(1)

    # Final cleanup: process any leftover buffers after video ends
    for lid, entries in track_buffers.items():
        for entry in entries:
            entry['action'] = "Uncertain"
            db.add_entry(entry)

    cap.release()

def _worker_process(args):
    gpu_id, videos, db_path, show_viz, skip_action, skip_ocr, reid_interval, action_interval, action_motion_threshold, detector_weights, action_model_path, ocr_interval_sec, fast_write, commit_interval = args
    db = VideoDB(db_path, fast_write=fast_write)
    id_matcher = GlobalMatcher(threshold=0.75)
    tracker = BoxMOTTracker(gpu_id=gpu_id, tracker_type='deepocsort', detector_weights=detector_weights)
    action_rec = None if skip_action else ActionRecognizer(device=f'cuda:{gpu_id}', model_path=action_model_path)
    reid_extractor = ReIDExtractor(device=f'cuda:{gpu_id}')
    ocr_reader = None if skip_ocr else TimeOCR(use_gpu=True)
    for i, video_path in enumerate(videos):
        process_single_video(
            video_path, db, tracker, action_rec,
            reid_extractor, id_matcher, ocr_reader,
            show_viz,
            reid_interval=reid_interval,
            action_interval=action_interval,
            skip_action=skip_action,
            skip_ocr=skip_ocr,
            action_motion_threshold=action_motion_threshold,
            ocr_interval_sec=ocr_interval_sec,
        )
        if (i + 1) % max(1, commit_interval) == 0:
            db.commit()
    db.commit()
    db.close()


def _merge_dbs(target_db, shard_dbs):
    if not shard_dbs:
        return
    main = VideoDB(target_db)
    main.cursor.execute("PRAGMA synchronous=OFF")
    main.cursor.execute("PRAGMA journal_mode=OFF")
    for shard in shard_dbs:
        main.cursor.execute("ATTACH DATABASE ? AS shard", (shard,))
        main.cursor.execute(
            """
            INSERT INTO frames (video_name, frame_idx, timestamp, ocr_time, person_id, action, bbox, keypoints, reid_feature)
            SELECT video_name, frame_idx, timestamp, ocr_time, person_id, action, bbox, keypoints, reid_feature
            FROM shard.frames
            """
        )
        main.cursor.execute("DETACH DATABASE shard")
        main.commit()
    main.close()


def build_db_parallel(
    dataset_dir,
    db_path,
    gpus,
    workers_per_gpu=1,
    show_viz=False,
    skip_action=False,
    skip_ocr=False,
    reid_interval=REID_INTERVAL,
    action_interval=ACTION_INTERVAL,
    action_motion_threshold=0.02,
    detector_weights="yolo11x-pose.pt",
    action_model_path="OpenGVLab/InternVideo2_5_Chat_8B",
    ocr_interval_sec=1.0,
    fast_write=False,
    commit_interval=10,
    reconcile_after_merge=True,
    reconcile_threshold=0.75,
):
    video_list = find_video_files(dataset_dir)
    gpu_ids = [int(x) for x in gpus if str(x).strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]
    num_workers = len(gpu_ids) * max(1, workers_per_gpu)
    chunk_size = math.ceil(len(video_list) / num_workers) if num_workers else len(video_list)
    work = []
    shard_dbs = []
    for i in range(num_workers):
        chunk = video_list[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue
        gpu_id = gpu_ids[i % len(gpu_ids)]
        shard_db = f"{db_path}.part{i}"
        shard_dbs.append(shard_db)
        work.append(
            (
                gpu_id,
                chunk,
                shard_db,
                show_viz,
                skip_action,
                skip_ocr,
                reid_interval,
                action_interval,
                action_motion_threshold,
                detector_weights,
                action_model_path,
                ocr_interval_sec,
                fast_write,
                commit_interval,
            )
        )
    with multiprocessing.Pool(processes=len(work)) as pool:
        pool.map(_worker_process, work)
    _merge_dbs(db_path, shard_dbs)
    if reconcile_after_merge:
        reconcile_ids(db_path, threshold=reconcile_threshold)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/data/lllidy/dataset/healthcare/videos_processed/recording_2019_06_22_9_20_am")
    parser.add_argument("--db", default="/data/lllidy/dataset/healthcare/db/recording_2019_06_22_9_20_am.db")
    parser.add_argument("--show_viz", action="store_true")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--skip_action", action="store_true")
    parser.add_argument("--skip_ocr", action="store_true")
    parser.add_argument("--reid_interval", type=int, default=REID_INTERVAL)
    parser.add_argument("--action_interval", type=int, default=ACTION_INTERVAL)
    parser.add_argument("--action_motion_threshold", type=float, default=0.02)
    parser.add_argument("--detector_weights", default="yolo11x-pose.pt")
    parser.add_argument("--action_model_path", default="OpenGVLab/InternVideo2_5_Chat_8B")
    parser.add_argument("--ocr_interval_sec", type=float, default=1.0)
    parser.add_argument("--fast_write", action="store_true")
    parser.add_argument("--commit_interval", type=int, default=10)
    parser.add_argument("--reconcile_after_merge", action="store_true")
    parser.add_argument("--reconcile_threshold", type=float, default=0.75)
    args = parser.parse_args()

    if Path(args.dataset).exists():
        os.makedirs(Path(args.db).parent, exist_ok=True)
        video_list = find_video_files(args.dataset)
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        if len(gpu_ids) <= 1 and args.workers_per_gpu == 1:
            process_dataset_folder(
                args.dataset,
                args.db,
                show_viz=args.show_viz,
                gpu_id=gpu_ids[0] if gpu_ids else 0,
                skip_action=args.skip_action,
                skip_ocr=args.skip_ocr,
                reid_interval=args.reid_interval,
                action_interval=args.action_interval,
                action_motion_threshold=args.action_motion_threshold,
                detector_weights=args.detector_weights,
                action_model_path=args.action_model_path,
                ocr_interval_sec=args.ocr_interval_sec,
                fast_write=args.fast_write,
                commit_interval=args.commit_interval,
            )
        else:
            num_workers = len(gpu_ids) * max(1, args.workers_per_gpu)
            chunk_size = math.ceil(len(video_list) / num_workers) if num_workers else len(video_list)
            work = []
            shard_dbs = []
            for i in range(num_workers):
                chunk = video_list[i * chunk_size : (i + 1) * chunk_size]
                if not chunk:
                    continue
                gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else 0
                shard_db = f"{args.db}.part{i}"
                shard_dbs.append(shard_db)
                work.append(
                    (
                        gpu_id,
                        chunk,
                        shard_db,
                        args.show_viz,
                        args.skip_action,
                        args.skip_ocr,
                        args.reid_interval,
                        args.action_interval,
                        args.action_motion_threshold,
                        args.detector_weights,
                        args.action_model_path,
                        args.ocr_interval_sec,
                        args.fast_write,
                        args.commit_interval,
                    )
                )
            with multiprocessing.Pool(processes=len(work)) as pool:
                pool.map(_worker_process, work)
            _merge_dbs(args.db, shard_dbs)
            if args.reconcile_after_merge:
                reconcile_ids(args.db, threshold=args.reconcile_threshold)
