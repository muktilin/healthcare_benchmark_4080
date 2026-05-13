import os
import sys

# Suppress OpenCV logging and debug messages
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
import math
import multiprocessing
from ultralytics import YOLO
from tqdm import tqdm

# Path Configuration
SOURCE_ROOT = r"/data/lllidy/dataset/healthcare/videos" 
OUTPUT_ROOT = r"/data/lllidy/dataset/healthcare/videos_processed"

# Processing Configuration
TARGET_HEIGHT = 720 
FRAME_INTERVAL = 5  # Process one frame every 5 frames
MODEL_PATH = 'yolov8n.pt' 

# Hardware Acceleration Configuration
NUM_GPUS = 8            
WORKERS_PER_GPU = 2  

def resize_frame(frame, target_height):
    """Resizes frame while maintaining aspect ratio based on target height."""
    h, w = frame.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    new_h = int(target_height)
    return cv2.resize(frame, (new_w, new_h)), new_w, new_h

def process_video_file(args):
    """Processes a single video: downsampling, resizing, and person detection filtering."""
    src_path, dst_path, gpu_id = args

    # Redirect stderr to devnull to suppress lower-level library logs
    try:
        devnull = open(os.devnull, 'w')
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
    except: pass

    try:
        model = YOLO(MODEL_PATH)
        model.to(f'cuda:{gpu_id}')
    except Exception as e:
        try: os.dup2(old_stderr, sys.stderr.fileno())
        except: pass
        print(f"[Error] GPU {gpu_id} Load Failed: {e}")
        return False

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or math.isnan(original_fps): original_fps = 25
    
    # Calculate target FPS after downsampling
    target_fps = original_fps / FRAME_INTERVAL
    
    writer = None
    frames_written = 0
    frame_idx = 0        
    consecutive_errors = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            # Handle transient read errors or EOF
            consecutive_errors += 1
            if consecutive_errors > 15: break
            else: continue
        
        consecutive_errors = 0 
        frame_idx += 1

        # Skip frames based on interval
        if frame_idx % FRAME_INTERVAL != 0:
            continue

        resized_frame, new_w, new_h = resize_frame(frame, TARGET_HEIGHT)

        # Detect persons (class 0) using the specified GPU
        results = model.predict(resized_frame, classes=[0], conf=0.35, 
                              device=f'cuda:{gpu_id}', verbose=False, stream=True)
        
        has_person = False
        for result in results:
            if len(result.boxes) > 0:
                has_person = True
                break
        
        # Only write frames that contain at least one person
        if has_person:
            if writer is None:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(dst_path, fourcc, target_fps, (new_w, new_h))
            
            writer.write(resized_frame)
            frames_written += 1

    cap.release()
    if writer is not None:
        writer.release()
    
    # Restore stderr
    try:
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(old_stderr)
        devnull.close()
    except: pass

    # Clean up empty files if no persons were detected throughout the video
    if frames_written == 0:
        if os.path.exists(dst_path):
            try: os.remove(dst_path)
            except: pass
        return False 
    
    return True

def worker_process(gpu_id, file_list):
    """Worker loop for a single subprocess assigned to a specific GPU."""
    for src, dst in tqdm(file_list, desc=f"GPU-{gpu_id}", position=gpu_id, leave=False):
        process_video_file((src, dst, gpu_id))

def main():
    print(f"=== Starting Processing: 8-GPU Parallel | Downsampling(1/{FRAME_INTERVAL}) | Resize({TARGET_HEIGHT}p) ===")
    
    all_tasks = []
    print("Scanning files...")
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            # Skip macOS metadata files
            if file.startswith("._"): continue 
            if file.lower().endswith(('.mov', '.mp4', '.avi', '.mkv')):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, SOURCE_ROOT)
                dst_path = os.path.join(OUTPUT_ROOT, rel_path)
                all_tasks.append((src_path, dst_path))

    total_files = len(all_tasks)
    print(f"Discovered {total_files} tasks.")

    if total_files == 0: return

    # Calculate task distribution across GPUs and workers
    num_processes = NUM_GPUS * WORKERS_PER_GPU
    chunk_size = math.ceil(total_files / num_processes)
    process_list = []

    for i in range(num_processes):
        chunk = all_tasks[i * chunk_size : (i + 1) * chunk_size]
        if not chunk: continue
        
        gpu_id = i % NUM_GPUS
        p = multiprocessing.Process(target=worker_process, args=(gpu_id, chunk))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print("\n=== All tasks completed! ===")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()