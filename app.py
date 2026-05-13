import argparse
import glob
import os
import multiprocessing
import sqlite3
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw
from tqdm import tqdm

from tool.dbmanager import VideoDB
from tool.track import BoxMOTTracker
from ultralytics import YOLO
from tool.action import ActionRecognizer
from tool.ocr import TimeOCR
from tool.reid import ReIDExtractor
from tool.generate_summary import get_person_activity_log, query_qwen_summary, _normalize_date
from tool.video_preprocessing import process_dataset_folder, build_db_parallel


VIDEO_ROOT_DEFAULT = "/data/lllidy/dataset/healthcare/videos_processed"
DB_ROOT_DEFAULT = "/data/lllidy/dataset/healthcare/db"
PREVIEW_DETECTOR_WEIGHTS_DEFAULT = "yolo11l-pose.pt"
PREVIEW_DETECTOR_IMGSZ_DEFAULT = 960

_DETECTOR_CACHE = {}

APP_CSS = """
:root {
  --bg: #f4efe6;
  --panel: rgba(255, 251, 245, 0.92);
  --panel-strong: rgba(255, 248, 239, 0.98);
  --ink: #182126;
  --muted: #5f6b73;
  --line: rgba(24, 33, 38, 0.08);
  --accent: #0f766e;
  --accent-2: #d97706;
  --shadow: 0 18px 48px rgba(29, 38, 44, 0.09);
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(217, 119, 6, 0.12), transparent 26%),
    radial-gradient(circle at top right, rgba(15, 118, 110, 0.14), transparent 24%),
    linear-gradient(180deg, #fcf8f2 0%, #f2ece4 100%);
  color: var(--ink);
  font-family: "Avenir Next", "Helvetica Neue", sans-serif;
}

.app-shell {
  max-width: 1480px;
  margin: 0 auto;
  padding: 10px 8px 28px;
}

.hero {
  background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,245,230,0.9));
  border: 1px solid rgba(24, 33, 38, 0.08);
  border-radius: 28px;
  padding: 28px 30px 24px;
  box-shadow: var(--shadow);
  margin-bottom: 18px;
}

.hero h1 {
  margin: 0;
  font-size: 34px;
  letter-spacing: -0.04em;
  font-weight: 700;
}

.hero p {
  margin: 10px 0 0;
  max-width: 900px;
  color: var(--muted);
  font-size: 15px;
  line-height: 1.6;
}

.panel,
.gr-group,
.gr-box,
.gr-accordion {
  background: var(--panel);
  border: 1px solid var(--line) !important;
  border-radius: 22px !important;
  box-shadow: var(--shadow);
}

.section-title {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--muted);
  margin-bottom: 8px;
}

.section-card {
  padding: 6px;
}

button.primary-action,
button.secondary-action {
  border-radius: 999px !important;
  min-height: 48px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
}

button.primary-action {
  background: linear-gradient(135deg, #0f766e, #0b5e58) !important;
  color: white !important;
  border: none !important;
}

button.secondary-action {
  background: linear-gradient(135deg, #d97706, #b45309) !important;
  color: white !important;
  border: none !important;
}

.compact-note {
  color: var(--muted);
  font-size: 13px;
  margin-top: -4px;
}

.output-panel textarea,
.output-panel pre,
.output-panel .scroll-hide {
  font-family: "IBM Plex Mono", "Menlo", monospace !important;
}
"""


def _merge_shards(target_db, shard_dbs):
    if not shard_dbs:
        return
    main = VideoDB(target_db, fast_write=True)
    for shard in shard_dbs:
        if not shard or not os.path.exists(shard):
            continue
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
        try:
            os.remove(shard)
        except Exception:
            pass
    main.close()


def _build_ocr_timeline(video_path, ocr_interval_sec=5.0):
    if not video_path:
        return {}
    ocr_reader = TimeOCR(use_gpu=True)
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps() or 25.0
    ocr_interval = max(1, int(fps * float(ocr_interval_sec)))
    timeline = {}
    current_ocr_text = ""
    for frame_idx, frame in enumerate(vr):
        if frame_idx % ocr_interval != 0:
            continue
        frame_rgb = frame.asnumpy()
        text = ocr_reader.recognize(frame_rgb[:, :, ::-1], roi_bbox=[0, 0, 600, 100])
        if text and text.strip():
            current_ocr_text = text
        sec = int(frame_idx / fps)
        timeline[sec] = current_ocr_text
    return timeline


def _process_cam_for_person(args):
    (
        cam,
        video_path,
        shard_db,
        person_id,
        target_feat,
        ocr_timeline,
        reid_interval,
        action_interval,
        detector_weights,
        detector_imgsz,
        detector_conf,
        action_model_path,
        action_window_sec,
        action_frames,
        gpu_id,
    ) = args

    if not video_path:
        print(f"[Track] {cam}: no video")
        return cam, 0
    db = VideoDB(shard_db, fast_write=True)
    try:
        tracker = BoxMOTTracker(
            gpu_id=gpu_id,
            tracker_type="deepocsort",
            detector_weights=detector_weights,
            detector_imgsz=int(detector_imgsz),
            detector_conf=float(detector_conf),
        )
    except TypeError:
        tracker = BoxMOTTracker(
            gpu_id=gpu_id,
            tracker_type="deepocsort",
            detector_weights=detector_weights,
        )
    action_rec = ActionRecognizer(device=f"cuda:{gpu_id}", model_path=action_model_path)
    reid_extractor = ReIDExtractor(device=f"cuda:{gpu_id}")

    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps() or 25.0
    window_frames = max(1, int(fps * float(action_window_sec)))
    target_frames = max(1, int(action_frames))
    stride = max(1, window_frames // target_frames)

    last_reid_features = {}
    target_track_id = None
    clip_buffer = []
    written = 0
    saved_check = False

    for frame_idx, frame in enumerate(vr):
        if frame_idx % int(action_interval) != 0:
            continue
        frame_rgb = frame.asnumpy()
        tracks = tracker.process_frame(frame_rgb[:, :, ::-1])
        if not tracks:
            continue
        active_ids = [t["track_id"] for t in tracks]
        if target_track_id not in active_ids or frame_idx % int(reid_interval) == 0:
            crops = []
            indices = []
            for idx, t in enumerate(tracks):
                x1, y1, x2, y2 = map(int, t["bbox"])
                crop = frame_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size > 0:
                    crops.append(crop)
                    indices.append(idx)
            if crops:
                feats = reid_extractor.extract(crops, input_color="rgb")
                best_idx = None
                best_score = -1.0
                for i, feat in enumerate(feats):
                    score = float(np.dot(feat, target_feat))
                    tid = tracks[indices[i]]["track_id"]
                    last_reid_features[tid] = feat
                    if score > best_score:
                        best_score = score
                        best_idx = indices[i]
                if best_idx is not None and best_score >= 0.35:
                    target_track_id = tracks[best_idx]["track_id"]

        if target_track_id not in active_ids:
            continue

        t = next(x for x in tracks if x["track_id"] == target_track_id)
        x1, y1, x2, y2 = map(int, t["bbox"])
        crop = frame_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue
        if frame_idx % stride == 0:
            clip_buffer.append(Image.fromarray(crop[:, :, ::-1]))
        if frame_idx % window_frames != 0 or frame_idx == 0:
            continue
        if len(clip_buffer) < target_frames:
            clip_buffer = []
            continue
        action_label = action_rec.predict_clip(clip_buffer)
        clip_buffer = []
        if action_label in ("buffering", "Buffering", "Uncertain", "Unknown", "", None):
            continue

        sec = int(frame_idx / fps)
        ocr_time = ocr_timeline.get(sec, "")
        entry = {
            "video_name": os.path.basename(video_path),
            "frame_idx": frame_idx,
            "timestamp": round(frame_idx / fps, 4),
            "ocr_time": ocr_time,
            "person_id": person_id,
            "action": action_label,
            "bbox": np.array(t["bbox"]),
            "keypoints": np.array(t["keypoints"]),
            "reid_feature": last_reid_features.get(target_track_id, target_feat),
        }
        db.add_entry(entry)
        written += 1
        if not saved_check:
            try:
                os.makedirs("check_buffer", exist_ok=True)
                img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(img)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                draw.text((x1 + 4, y1 + 4), f"{cam} ID:{person_id}", fill=(255, 0, 0))
                img.save(os.path.join("check_buffer", f"{cam}_frame{frame_idx:06d}_ID{person_id}.jpg"))
                saved_check = True
            except Exception:
                pass
        print(f"[Action] {cam} t={sec}s action={action_label}")
        if written % 50 == 0:
            db.commit()

    db.commit()
    db.close()
    return cam, written


def _get_detector(weights):
    model = _DETECTOR_CACHE.get(weights)
    if model is None:
        model = YOLO(weights)
        _DETECTOR_CACHE[weights] = model
    return model


def list_videos(root_dir):
    exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return sorted(files)


def list_recordings(root_dir):
    if not root_dir or not os.path.isdir(root_dir):
        return []
    entries = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.startswith("recording_"):
            entries.append(full)
    return sorted(entries)


def parse_date_from_recording(recording_dir):
    # recording_2019_06_22_9_20_am -> 2019-06-22
    base = os.path.basename(recording_dir)
    parts = base.split("_")
    if len(parts) >= 4:
        y, m, d = parts[1], parts[2], parts[3]
        if len(y) == 4 and len(m) == 2 and len(d) == 2:
            return f"{y}-{m}-{d}"
    return None


def first_video_in_cam(recording_dir, cam_name):
    cam_dir = os.path.join(recording_dir, cam_name)
    if not os.path.isdir(cam_dir):
        return None
    vids = list_videos(cam_dir)
    return vids[0] if vids else None


def get_recording_cam_videos(recording_dir):
    cams = ["cam_10", "cam_11", "cam_12", "cam_13"]
    out = {}
    for cam in cams:
        out[cam] = first_video_in_cam(recording_dir, cam)
    return out


def find_db_for_recording(recording_dir):
    if not recording_dir:
        return None
    db_root = DB_ROOT_DEFAULT
    date_str = parse_date_from_recording(recording_dir)
    candidates = [
        os.path.join(db_root, f"{os.path.basename(recording_dir)}.db"),
        os.path.join(db_root, f"{date_str}.db") if date_str else None,
        os.path.join(recording_dir, "person.db"),
        os.path.join(recording_dir, "video_data.db"),
    ]
    for path in candidates:
        if not path:
            continue
        if os.path.exists(path):
            return path
    return None


def resolve_db_path(recording_dir, db_path):
    detected = find_db_for_recording(recording_dir)
    if detected:
        return detected
    date_str = parse_date_from_recording(recording_dir) if recording_dir else None
    if date_str:
        return os.path.join(DB_ROOT_DEFAULT, f"{date_str}.db")
    if recording_dir:
        return os.path.join(DB_ROOT_DEFAULT, f"{os.path.basename(recording_dir)}.db")
    return db_path


def read_frame(video_path, frame_idx=None):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total == 0:
            return None
        if frame_idx is None:
            frame_idx = max(0, total // 3)
        frame_idx = min(max(0, frame_idx), total - 1)
        try:
            return vr[frame_idx].asnumpy()  # RGB
        except Exception:
            try:
                return vr[0].asnumpy()
            except Exception:
                return _read_first_frame_imageio(video_path)
    except Exception:
        frame = _read_frame_imageio(video_path, frame_idx)
        return frame if frame is not None else _read_first_frame_imageio(video_path)


def _read_frame_imageio(video_path, frame_idx=None):
    try:
        import imageio.v3 as iio
    except Exception:
        return None
    try:
        if frame_idx is None:
            frame_idx = 0
        frame = iio.imread(video_path, index=frame_idx)
        if frame is None:
            return None
        return frame
    except Exception:
        return None


def _read_first_frame_imageio(video_path):
    try:
        import imageio.v3 as iio
    except Exception:
        return None
    try:
        it = iio.imiter(video_path)
        frame = next(it, None)
        return frame
    except Exception:
        return None


def draw_boxes(frame_rgb, bboxes):
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, y1 + 4), str(i), fill=(0, 255, 0))
    return np.array(img)


def _clip_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), max(0, width - 1)))
    y1 = max(0, min(int(round(y1)), max(0, height - 1)))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return [x1, y1, x2, y2]


def _refine_preview_bbox(
    frame_rgb,
    bbox,
    keypoints=None,
    pad_x=0.14,
    pad_top=0.2,
    pad_bottom=0.28,
    target_ratio=0.48,
):
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)

    if keypoints is not None:
        points = np.asarray(keypoints)
        if points.ndim == 2 and points.shape[1] >= 2:
            if points.shape[1] >= 3:
                visible = points[:, 2] > 0.2
                points = points[visible]
            if len(points) > 0:
                px = points[:, 0]
                py = points[:, 1]
                x1 = min(x1, float(np.min(px)))
                y1 = min(y1, float(np.min(py)))
                x2 = max(x2, float(np.max(px)))
                y2 = max(y2, float(np.max(py)))

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    refined = [
        x1 - box_w * pad_x,
        y1 - box_h * pad_top,
        x2 + box_w * pad_x,
        y2 + box_h * pad_bottom,
    ]
    refined_w = max(1.0, refined[2] - refined[0])
    refined_h = max(1.0, refined[3] - refined[1])
    if refined_w / refined_h > target_ratio:
        desired_h = refined_w / target_ratio
        extra_h = desired_h - refined_h
        refined[1] -= extra_h * 0.22
        refined[3] += extra_h * 0.78
    return _clip_bbox(refined, width, height)


def detect_bboxes(
    frame_rgb,
    detector_weights=PREVIEW_DETECTOR_WEIGHTS_DEFAULT,
    detector_imgsz=PREVIEW_DETECTOR_IMGSZ_DEFAULT,
    detector_conf=0.35,
):
    # Use detector only for selection to avoid tracker init cost
    model = _get_detector(detector_weights)
    results = model.predict(
        frame_rgb,
        classes=[0],
        conf=float(detector_conf),
        imgsz=int(detector_imgsz),
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []
    raw_boxes = results[0].boxes.xyxy.cpu().numpy()
    raw_keypoints = None
    if results[0].keypoints is not None and getattr(results[0].keypoints, "data", None) is not None:
        raw_keypoints = results[0].keypoints.data.cpu().numpy()

    boxes = []
    for idx, bbox in enumerate(raw_boxes):
        keypoints = raw_keypoints[idx] if raw_keypoints is not None and idx < len(raw_keypoints) else None
        boxes.append(_refine_preview_bbox(frame_rgb, bbox, keypoints=keypoints))
    return boxes


def select_bbox_by_click(bboxes, x, y):
    if not bboxes:
        return None
    best_idx = None
    best_score = -1e9
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        inside = x1 <= x <= x2 and y1 <= y <= y2
        if inside:
            area = (x2 - x1) * (y2 - y1)
            score = area
        else:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = (x - cx) ** 2 + (y - cy) ** 2
            score = -dist
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def extract_reid_feature(frame_rgb, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reid = ReIDExtractor(device=device)
    feats = reid.extract([crop], input_color="rgb")
    if feats.size == 0:
        return None
    return feats[0]


def extract_reid_feature_multi(video_path, frame_idx, bbox, offsets):
    feats = []
    for off in offsets:
        idx = frame_idx + off if frame_idx is not None else None
        frame = read_frame(video_path, idx)
        if frame is None:
            continue
        feat = extract_reid_feature(frame, bbox)
        if feat is not None:
            feats.append(feat)
    if not feats:
        return None
    feat = np.mean(np.stack(feats, axis=0), axis=0)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def load_video_state(
    recording_dir,
    frame_idx,
    detector_weights=PREVIEW_DETECTOR_WEIGHTS_DEFAULT,
    detector_imgsz=PREVIEW_DETECTOR_IMGSZ_DEFAULT,
    detector_conf=0.35,
):
    if not recording_dir:
        return None, None, None, "Choose a recording first."
    cam_videos = get_recording_cam_videos(recording_dir)
    cam10 = cam_videos.get("cam_10")
    if not cam10:
        return None, None, None, "No video found under cam_10."
    frame = read_frame(cam10, frame_idx)
    if frame is None:
        return None, None, None, "Failed to read the preview frame."
    bboxes = detect_bboxes(frame, detector_weights=detector_weights, detector_imgsz=detector_imgsz, detector_conf=detector_conf)
    vis = draw_boxes(frame, bboxes) if bboxes else frame
    return frame, bboxes, vis, f"cam_10 preview detected {len(bboxes)} people."


def on_image_click(evt: gr.SelectData, frame, bboxes):
    if frame is None or not bboxes:
        return None, None, "No person was detected in the preview."
    x, y = evt.index
    idx = select_bbox_by_click(bboxes, x, y)
    if idx is None:
        return None, None, "No person was selected."
    bbox = bboxes[idx]
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    return bbox, crop, f"Selected bounding box #{idx}."


def choose_person_id(db_path, frame, bbox, text_prompt, recording_dir, frame_idx, cam_videos):
    db_path = resolve_db_path(recording_dir, db_path)
    if not db_path or not recording_dir:
        return None, "Recording is missing or the DB path is invalid."
    db = VideoDB(db_path)
    if bbox is None:
        db.close()
        return None, "Select a person from the preview image first."
    cam10 = cam_videos.get("cam_10") if cam_videos else None
    if cam10:
        offsets = [0, 5, -5, 10, -10]
        feat = extract_reid_feature_multi(cam10, frame_idx or 0, bbox, offsets)
    else:
        feat = extract_reid_feature(frame, bbox)
    if feat is None:
        db.close()
        return None, "Failed to extract the ReID feature."
    video_names = []
    if cam_videos:
        for v in cam_videos.values():
            if v:
                video_names.append(os.path.basename(v))
    candidates = db.search_by_image_feature_filtered(feat, top_k=5, threshold=0.35, video_names=video_names)
    if candidates:
        best_id = candidates[0]["person_id"]
        db.close()
        return best_id, f"Matched an existing person_id={best_id}."
    # no match: create new id
    db.cursor.execute("SELECT MAX(person_id) FROM frames")
    row = db.cursor.fetchone()
    max_id = row[0] if row and row[0] is not None else 0
    new_id = int(max_id) + 1
    db.close()
    return new_id, f"No existing match found. Created person_id={new_id}."


def generate_log(
    db_path,
    person_id,
    date_filter,
    cam_videos,
):
    if person_id is None:
        return "Select a person first."
    log = build_dense_log(
        person_id,
        db_path=db_path,
        date_filter=date_filter,
        video_names=cam_videos,
        max_lines=None,
        step=1,
    )
    if not log:
        return "No valid activity log was found."
    thinking, summary = query_qwen_summary(person_id, log)
    if not summary:
        return "Qwen did not return a valid summary."
    return summary


def build_dense_log(
    person_id,
    db_path,
    date_filter=None,
    video_names=None,
    max_lines=None,
    step=1,
):
    """
    Build raw_data_sequence from DB: ALL timestamps + actions for this person.
    No label filtering, no subsampling, no dedup.
    """
    if not db_path or not os.path.exists(db_path):
        return None
    # keep all records for this person, but filter invalid actions
    invalid_labels = {"buffering", "None", "Unknown", "Buffering", "Uncertain", ""}

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ocr_time, timestamp, action, video_name
            FROM frames
            WHERE person_id = ?
            ORDER BY video_name, frame_idx ASC
            """,
            (person_id,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    valid_video_names = None
    if video_names:
        valid_video_names = set()
        for name in video_names:
            if not name:
                continue
            valid_video_names.add(str(name))
            valid_video_names.add(Path(str(name)).stem)

    normalized_filter = _normalize_date(date_filter) if date_filter else None

    lines = []
    for (ocr_time, timestamp, action, video_name) in rows:
        if action in invalid_labels:
            continue
        if normalized_filter:
            current_date = _normalize_date(ocr_time) or ocr_time
            if current_date != normalized_filter:
                continue
        if valid_video_names and video_name not in valid_video_names and Path(video_name).stem not in valid_video_names:
            continue
        label = ocr_time.strip() if isinstance(ocr_time, str) and ocr_time.strip() else f"T+{float(timestamp):.1f}s"
        lines.append(f"[{label}] {action}")

    return "\n".join(lines) if lines else None


def build_app(video_root):
    recordings = list_recordings(video_root)

    with gr.Blocks(css=APP_CSS, title="Person Activity Summary System") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="hero">
                  <h1>Person Activity Summary System</h1>
                  <p>
                    Select a person from a recording, track identity across cameras, and generate a long-form
                    evidence-grounded activity summary from the recorded timeline.
                  </p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=7):
                    with gr.Group(elem_classes=["panel", "section-card"]):
                        gr.HTML('<div class="section-title">Session Setup</div>')
                        with gr.Row():
                            recording_dropdown = gr.Dropdown(
                                choices=recordings,
                                label="Recording",
                                info="Choose a processed recording folder.",
                            )
                            date_filter = gr.Textbox(
                                label="Date Hint",
                                placeholder="2019-06-22",
                            )
                        with gr.Row():
                            frame_idx = gr.Number(value=None, precision=0, label="Frame Index")
                            db_path = gr.Textbox(value="person.db", label="Database Path")
                            load_btn = gr.Button("Load Preview", elem_classes=["primary-action"])
                        status = gr.Textbox(label="Status", interactive=False)

                    with gr.Group(elem_classes=["panel", "section-card"]):
                        gr.HTML('<div class="section-title">Visual Selection</div>')
                        gr.Markdown(
                            "<div class='compact-note'>Load the preview, click a person in the left frame, and confirm the identity on the right. Summary generation uses the full DB timeline for the selected person_id.</div>"
                        )
                        with gr.Row():
                            image = gr.Image(label="Preview Frame", type="numpy", height=520)
                            crop_preview = gr.Image(label="Selected Crop", type="numpy", height=520)

                    with gr.Group(elem_classes=["panel", "section-card"]):
                        gr.HTML('<div class="section-title">Outputs</div>')
                        with gr.Tabs():
                            with gr.Tab("Summary Report"):
                                log_all = gr.Textbox(
                                    label="Summary Report",
                                    lines=18,
                                    elem_classes=["output-panel"],
                                )
                            with gr.Tab("Action Timeline"):
                                timeline_all = gr.Textbox(
                                    label="Action Timeline",
                                    lines=18,
                                    elem_classes=["output-panel"],
                                )

                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["panel", "section-card"]):
                        gr.HTML('<div class="section-title">Identity</div>')
                        text_prompt = gr.Textbox(
                            label="Description",
                            placeholder="Optional free-form note about the target person.",
                        )
                        result = gr.Textbox(label="Match Result", interactive=False)
                        select_btn = gr.Button("Confirm Identity", elem_classes=["primary-action"])
                        with gr.Row():
                            log_btn = gr.Button("Generate Summary", elem_classes=["secondary-action"])
                            track_btn = gr.Button("Track And Write To DB", elem_classes=["primary-action"])

                    with gr.Accordion("Tracking Controls", open=False, elem_classes=["panel"]):
                        with gr.Row():
                            reid_interval = gr.Number(value=10, precision=0, label="ReID Interval")
                            action_interval = gr.Number(value=6, precision=0, label="Action Interval")
                            ocr_interval_sec = gr.Number(value=1.0, label="OCR Interval (sec)")
                        with gr.Row():
                            action_window_sec = gr.Number(value=5.0, label="Action Window (sec)")
                            action_frames = gr.Number(value=32, precision=0, label="Action Frames")
                            motion_trigger_threshold = gr.Number(value=0.0008, label="Motion Trigger")

                    with gr.Accordion("Model And Detection Settings", open=False, elem_classes=["panel"]):
                        with gr.Row():
                            detector_weights = gr.Textbox(value=PREVIEW_DETECTOR_WEIGHTS_DEFAULT, label="Detector Weights")
                            action_model_path = gr.Textbox(value="OpenGVLab/InternVL2_5-1B", label="Action Model")
                        with gr.Row():
                            detector_imgsz = gr.Number(value=PREVIEW_DETECTOR_IMGSZ_DEFAULT, precision=0, label="Detector Image Size")
                            detector_conf = gr.Number(value=0.35, label="Detector Confidence")

        frame_state = gr.State()
        bboxes_state = gr.State()
        bbox_state = gr.State()
        feature_state = gr.State()
        person_id_state = gr.State()
        cam_videos_state = gr.State()

        def _load(recording_dir, frame_idx, db_path, detector_weights, detector_imgsz, detector_conf):
            frame, bboxes, vis, msg = load_video_state(
                recording_dir,
                frame_idx,
                detector_weights=detector_weights,
                detector_imgsz=detector_imgsz,
                detector_conf=detector_conf,
            )
            cam_videos = get_recording_cam_videos(recording_dir) if recording_dir else {}
            date_guess = parse_date_from_recording(recording_dir) if recording_dir else None
            detected_db = resolve_db_path(recording_dir, db_path)
            db_msg = ""
            if detected_db and os.path.exists(detected_db):
                db_path = detected_db
                db_msg = f"Detected DB: {detected_db}"
            else:
                db_root = DB_ROOT_DEFAULT
                date_str = parse_date_from_recording(recording_dir) if recording_dir else None
                db_default = (
                    os.path.join(db_root, f"{date_str}.db")
                    if date_str
                    else os.path.join(db_root, f"{os.path.basename(recording_dir)}.db")
                )
                db_msg = (
                    "No DB was found for this recording.\n"
                    f"Suggested DB path: {db_default}\n"
                    "A new DB will be created automatically when needed."
                )
            status_msg = f"{msg} | {db_msg}"
            return frame, bboxes, vis, status_msg, cam_videos, (date_guess or ""), db_path

        load_btn.click(
            _load,
            inputs=[recording_dropdown, frame_idx, db_path, detector_weights, detector_imgsz, detector_conf],
            outputs=[frame_state, bboxes_state, image, status, cam_videos_state, date_filter, db_path],
        )

        def _on_click(frame, bboxes, detector_weights, detector_imgsz, detector_conf, evt: gr.SelectData):
            if (bboxes is None or len(bboxes) == 0) and frame is not None:
                bboxes = detect_bboxes(
                    frame,
                    detector_weights=detector_weights,
                    detector_imgsz=detector_imgsz,
                    detector_conf=detector_conf,
                )
            if evt is None or not hasattr(evt, "index"):
                return None, None, None, "No click event was received."
            bbox, crop, msg = on_image_click(evt, frame, bboxes)
            if bbox is None or crop is None:
                return bbox, crop, None, msg
            feat = extract_reid_feature(frame, bbox)
            return bbox, crop, feat, msg

        image.select(
            _on_click,
            inputs=[frame_state, bboxes_state, detector_weights, detector_imgsz, detector_conf],
            outputs=[bbox_state, crop_preview, feature_state, status],
        )

        select_btn.click(
            choose_person_id,
            inputs=[db_path, frame_state, bbox_state, text_prompt, recording_dropdown, frame_idx, cam_videos_state],
            outputs=[person_id_state, result],
        )

        def _gen_all(db_path, person_id, date_filter, cam_videos):
            if not cam_videos:
                return generate_log(db_path, person_id, date_filter, [])
            video_names = []
            for cam in ["cam_10", "cam_11", "cam_12", "cam_13"]:
                video_path = cam_videos.get(cam)
                if not video_path:
                    continue
                base = os.path.basename(video_path)
                video_names.extend([base, os.path.splitext(base)[0]])
            return generate_log(db_path, person_id, date_filter, video_names)

        log_btn.click(
            _gen_all,
            inputs=[db_path, person_id_state, date_filter, cam_videos_state],
            outputs=[log_all],
        )

        def _track_person(
            recording_dir,
            db_path,
            person_id,
            target_feat,
            cam_videos,
            reid_interval,
            action_interval,
            ocr_interval_sec,
            detector_weights,
            detector_imgsz,
            detector_conf,
            action_model_path,
            action_window_sec,
            action_frames,
            motion_trigger_threshold,
        ):
            if not recording_dir:
                yield "No recording selected.", "No valid action timeline."
                return
            if target_feat is None:
                yield "No person selected.", "No valid action timeline."
                return

            db_path = resolve_db_path(recording_dir, db_path)
            db_dir = os.path.dirname(db_path) if db_path else ""
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            db = VideoDB(db_path, fast_write=True)

            if person_id is None:
                candidates = db.search_by_image_feature(target_feat, top_k=1, threshold=0.35)
                if candidates:
                    person_id = candidates[0]["person_id"]
                else:
                    db.cursor.execute("SELECT MAX(person_id) FROM frames")
                    row = db.cursor.fetchone()
                    max_id = row[0] if row and row[0] is not None else 0
                    person_id = int(max_id) + 1

            try:
                tracker = BoxMOTTracker(
                    gpu_id=0,
                    tracker_type="deepocsort",
                    detector_weights=detector_weights,
                    detector_imgsz=int(detector_imgsz),
                    detector_conf=float(detector_conf),
                )
            except TypeError:
                tracker = BoxMOTTracker(
                    gpu_id=0,
                    tracker_type="deepocsort",
                    detector_weights=detector_weights,
                )
            action_rec = ActionRecognizer(device="cuda:0", model_path=action_model_path)
            reid_extractor = ReIDExtractor(device="cuda:0")
            ocr_reader = TimeOCR(use_gpu=True)

            timeline_lines = []
            cams = ["cam_10", "cam_11", "cam_12", "cam_13"]
            cam_videos_map = {}
            for cam in cams:
                cam_dir = os.path.join(recording_dir, cam)
                cam_videos_map[cam] = list_videos(cam_dir) if os.path.isdir(cam_dir) else []

            print("[Track] Search order:", cams)
            for cam in cams:
                print(f"[Track] {cam} has {len(cam_videos_map[cam])} videos")

            def _process_video(video_path, cam):
                nonlocal timeline_lines
                written = 0
                if hasattr(tracker, "tracker") and hasattr(tracker.tracker, "reset"):
                    try:
                        tracker.tracker.reset()
                    except Exception:
                        pass
                try:
                    vr = VideoReader(video_path, ctx=cpu(0))
                except Exception:
                    print(f"[Track] Failed to open {video_path}")
                    return written

                fps = vr.get_avg_fps() or 25.0
                yield f"Processing {cam} {os.path.basename(video_path)}", ""
                ocr_interval = max(1, int(fps * float(ocr_interval_sec)))
                window_frames = max(1, int(fps * float(action_window_sec)))
                target_frames = max(1, int(action_frames))
                stride = max(1, window_frames // target_frames)
                current_ocr_text = ""
                prev_ocr_text = ""
                last_reid_features = {}
                target_track_id = None
                clip_buffer = []
                last_bbox = None
                last_action_label = None
                last_target_feat = None
                last_action_frame = -10**9
                saved_check = False

                for frame_idx, frame in enumerate(vr):
                    if frame_idx % int(action_interval) != 0:
                        continue
                    frame_rgb = frame.asnumpy()
                    if frame_idx % ocr_interval == 0:
                        current_ocr_text = ocr_reader.recognize(
                            frame_rgb[:, :, ::-1],
                            roi_bbox=[0, 0, 600, 100],
                        ) or current_ocr_text

                    tracks = tracker.process_frame(frame_rgb[:, :, ::-1])
                    if not tracks:
                        continue

                    active_ids = [t["track_id"] for t in tracks]
                    if target_track_id not in active_ids or frame_idx % int(reid_interval) == 0:
                        crops = []
                        indices = []
                        for idx, t in enumerate(tracks):
                            x1, y1, x2, y2 = map(int, t["bbox"])
                            crop = frame_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                            if crop.size > 0:
                                crops.append(crop)
                                indices.append(idx)
                        if crops:
                            feats = reid_extractor.extract(crops, input_color="rgb")
                            best_idx = None
                            best_score = -1.0
                            for i, feat in enumerate(feats):
                                score = float(np.dot(feat, target_feat))
                                tid = tracks[indices[i]]["track_id"]
                                last_reid_features[tid] = feat
                                if score > best_score:
                                    best_score = score
                                    best_idx = indices[i]
                            if best_idx is not None and best_score >= 0.35:
                                target_track_id = tracks[best_idx]["track_id"]

                    if target_track_id not in active_ids:
                        continue

                    t = next(x for x in tracks if x["track_id"] == target_track_id)
                    x1, y1, x2, y2 = map(int, t["bbox"])
                    crop = frame_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if crop.size == 0:
                        continue
                    # update target appearance feature when available
                    if last_reid_features.get(target_track_id) is not None:
                        last_target_feat = last_reid_features.get(target_track_id)
                    motion_score = 0.0
                    if last_bbox is not None:
                        px1, py1, px2, py2 = last_bbox
                        pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
                        motion_score = ((cx - pcx) / pw) ** 2 + ((cy - pcy) / ph) ** 2
                    last_bbox = (x1, y1, x2, y2)

                    # information-gain adaptive sampling (no training):
                    # proxy IG = wm*Δmotion + wa*Δappearance + wo*Δocr
                    appearance_change = 1.0
                    if last_target_feat is not None:
                        cur_feat = last_reid_features.get(target_track_id, last_target_feat)
                        appearance_change = 1.0 - float(np.dot(cur_feat, last_target_feat))
                    ocr_change = 1.0 if (current_ocr_text and current_ocr_text != prev_ocr_text) else 0.0
                    if current_ocr_text and current_ocr_text != prev_ocr_text:
                        prev_ocr_text = current_ocr_text

                    ig_wm, ig_wa, ig_wo = 1.0, 1.0, 0.5
                    ig_tau = 0.25
                    info_gain = ig_wm * float(motion_score) + ig_wa * float(appearance_change) + ig_wo * float(ocr_change)
                    if info_gain < ig_tau:
                        continue

                    if frame_idx % stride == 0:
                        clip_buffer.append((motion_score, frame_idx, Image.fromarray(crop[:, :, ::-1])))

                    if frame_idx % window_frames != 0 or frame_idx == 0:
                        continue

                    if not clip_buffer:
                        continue

                    avg_motion = float(np.mean([m for m, _, _ in clip_buffer])) if clip_buffer else 0.0
                    if avg_motion < float(motion_trigger_threshold) and last_action_label is not None:
                        action_label = last_action_label
                    else:
                        clip_buffer.sort(key=lambda x: x[0], reverse=True)
                        selected = clip_buffer[:target_frames]
                        selected.sort(key=lambda x: x[1])
                        frames = [img for _, _, img in selected]
                        action_label = action_rec.predict_clip(frames)
                        last_action_label = action_label

                    clip_buffer = []
                    if action_label in ("buffering", "Buffering", "Uncertain", "Unknown", "", None):
                        continue
                    last_action_frame = frame_idx

                    entry = {
                        "video_name": os.path.basename(video_path),
                        "frame_idx": frame_idx,
                        "timestamp": round(frame_idx / fps, 4),
                        "ocr_time": current_ocr_text,
                        "person_id": person_id,
                        "action": action_label,
                        "bbox": np.array(t["bbox"]),
                        "keypoints": np.array(t["keypoints"]),
                        "reid_feature": last_reid_features.get(target_track_id, target_feat),
                    }
                    db.add_entry(entry)
                    written += 1
                    if not saved_check:
                        try:
                            os.makedirs("check_buffer", exist_ok=True)
                            img = Image.fromarray(frame_rgb)
                            draw = ImageDraw.Draw(img)
                            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                            draw.text((x1 + 4, y1 + 4), f"{cam} ID:{person_id}", fill=(255, 0, 0))
                            img.save(os.path.join("check_buffer", f"{cam}_frame{frame_idx:06d}_ID{person_id}.jpg"))
                            saved_check = True
                        except Exception:
                            pass

                    ts_label = current_ocr_text if current_ocr_text else f"T+{int(frame_idx / fps):04d}s"
                    timeline_lines.append(f"[{cam} {ts_label}] {action_label}")
                    print(f"[Action] {cam} {ts_label} {action_label}")
                    if len(timeline_lines) % 10 == 0:
                        db.commit()
                        yield f"Processing {cam} ...", "\n".join(timeline_lines[-200:])

                db.commit()
                yield (
                    f"Finished {cam} {os.path.basename(video_path)} wrote={written}",
                    "\n".join(timeline_lines[-50:]) if timeline_lines else "",
                )
                return written

            max_segments = max(len(v) for v in cam_videos_map.values()) if cam_videos_map else 0
            if max_segments == 0:
                db.close()
                yield "No videos were found.", "No valid action timeline."
                return

            active_cam = "cam_10"
            consecutive_miss = 0
            found_any = False
            seg_idx = 0

            while seg_idx < max_segments:
                videos = cam_videos_map.get(active_cam, [])
                if seg_idx >= len(videos):
                    consecutive_miss += 1
                    if consecutive_miss >= 8:
                        print("[Track] Stop: 8 consecutive missing segments")
                        break
                    active_cam = cams[(cams.index(active_cam) + 1) % len(cams)]
                    continue

                video_path = videos[seg_idx]
                print(f"[Track] Try {active_cam} video {seg_idx + 1}/{len(videos)}: {os.path.basename(video_path)}")
                written = yield from _process_video(video_path, active_cam)
                if written is None:
                    written = 0

                if written > 0:
                    found_any = True
                    consecutive_miss = 0
                    seg_idx += 1
                    continue

                consecutive_miss += 1
                if consecutive_miss >= 8:
                    print("[Track] Stop: 8 consecutive missing segments")
                    break
                active_cam = cams[(cams.index(active_cam) + 1) % len(cams)]

            db.commit()
            db.close()
            if found_any:
                yield f"Completed: person_id={person_id}", "\n".join(timeline_lines[-500:])
            else:
                yield "The target person was not found in any camera.", "No valid action timeline."

        track_btn.click(
            _track_person,
            inputs=[
                recording_dropdown,
                db_path,
                person_id_state,
                feature_state,
                cam_videos_state,
                reid_interval,
                action_interval,
                ocr_interval_sec,
                detector_weights,
                detector_imgsz,
                detector_conf,
                action_model_path,
                action_window_sec,
                action_frames,
                motion_trigger_threshold,
            ],
            outputs=[status, timeline_all],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default=VIDEO_ROOT_DEFAULT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app(args.video_root)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
