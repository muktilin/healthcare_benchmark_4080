import argparse
import glob
import os
import sqlite3
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw
from ultralytics import YOLO

from tool.action import ActionRecognizer
from tool.dbmanager import VideoDB
from tool.generate_summary import _normalize_date
from tool.generate_summary_on_chip import (
    ON_CHIP_MODEL_NAME,
    empty_summary_cards,
    generate_on_chip_summary,
    is_valid_on_chip_response,
    render_on_chip_summary,
)
from tool.ocr import TimeOCR
from tool.reid import ReIDExtractor
from tool.track import BoxMOTTracker


VIDEO_ROOT_DEFAULT = "/data/lllidy/dataset/healthcare/videos_processed"
PROJECT_DB_DIR_DEFAULT = str(Path(__file__).resolve().parent / "db")
PREVIEW_DETECTOR_WEIGHTS_DEFAULT = "yolo11l-pose.pt"
PREVIEW_DETECTOR_IMGSZ_DEFAULT = 960
AVAILABLE_SUMMARY_MODELS = [
    ON_CHIP_MODEL_NAME,
]
DEFAULT_SUMMARY_MODEL = AVAILABLE_SUMMARY_MODELS[0]

_DETECTOR_CACHE = {}

APP_CSS = """
:root {
  --bg: #eef8f6;
  --panel: rgba(255, 255, 255, 0.88);
  --panel-strong: rgba(255, 255, 255, 0.96);
  --panel-soft: rgba(15, 118, 110, 0.1);
  --ink: #183130;
  --muted: #607675;
  --line: rgba(15, 118, 110, 0.2);
  --accent: #0f766e;
  --accent-2: #e85d4f;
  --accent-3: #5b8c00;
  --field: #f8fffd;
  --field-strong: #e8f7f3;
  --shadow: 0 18px 44px rgba(20, 69, 65, 0.14);
  --glow: 0 0 0 1px rgba(15, 118, 110, 0.14), 0 18px 56px rgba(15, 118, 110, 0.14);
}

html,
body,
#root {
  width: 100%;
  min-height: 100%;
  margin: 0;
}

.gradio-container {
  width: 100vw !important;
  max-width: none !important;
  min-height: 100vh;
  padding: 0 !important;
  background:
    linear-gradient(rgba(15, 118, 110, 0.07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15, 118, 110, 0.07) 1px, transparent 1px),
    linear-gradient(135deg, #e9fbf7 0%, #f7fbff 42%, #fff1ec 100%);
  background-size: 34px 34px, 34px 34px, auto;
  color: var(--ink);
  font-family: "Avenir Next", "Helvetica Neue", sans-serif;
  font-size: 16px;
}

.gradio-container .contain,
.gradio-container main,
.gradio-container > div {
  max-width: none !important;
}

.app-shell {
  width: calc(100vw - 20px);
  max-width: none;
  margin: 0;
  padding: 10px;
  height: 100vh;
  overflow: auto;
  gap: 10px !important;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  min-height: 64px;
  padding: 14px 18px;
  background:
    linear-gradient(90deg, rgba(15, 118, 110, 0.18), transparent 36%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(235, 255, 250, 0.94)),
    var(--panel-strong);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow), var(--glow);
}

.app-header h1 {
  margin: 0;
  font-size: 34px;
  letter-spacing: 0;
  font-weight: 800;
  color: var(--ink);
}

.app-header p {
  margin: 2px 0 0;
  color: var(--muted);
  font-size: 16px;
  line-height: 1.35;
}

.app-badge {
  background: linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(232, 93, 79, 0.1));
  border: 1px solid rgba(15, 118, 110, 0.24);
  border-radius: 8px;
  color: var(--accent);
  font-size: 14px;
  font-weight: 700;
  padding: 10px 14px;
  white-space: nowrap;
  box-shadow: inset 0 0 18px rgba(15, 118, 110, 0.08);
}

.panel,
.gr-group,
.gr-box,
.gr-accordion {
  background: var(--panel);
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  box-shadow: var(--shadow);
  color: var(--ink);
  backdrop-filter: blur(14px);
}

.gradio-container .form,
.gradio-container .block,
.gradio-container .block.padded,
.gradio-container .gr-form {
  background: transparent !important;
  border-color: rgba(15, 118, 110, 0.14) !important;
}

.gradio-container .panel,
.gradio-container .gr-group {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(244, 255, 252, 0.88)) !important;
}

.compact-panel {
  padding: 8px !important;
}

.setup-row,
.work-row {
  gap: 10px !important;
}

.work-row {
  min-height: 0;
  flex: 1 1 auto;
  overflow: visible;
  align-items: stretch !important;
}

.work-row > .column:first-child {
  flex: 1.05 1 0 !important;
}

.work-row > .column:nth-child(2) {
  flex: 0 0 330px !important;
  max-width: 330px !important;
}

.work-row > .column:nth-child(3) {
  flex: 1.75 1 0 !important;
}

.section-title {
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 0;
  color: var(--accent);
  margin: 0 0 8px;
  font-weight: 800;
  padding: 2px 0 0;
}

.section-card {
  padding: 6px;
}

button.primary-action,
button.secondary-action {
  border-radius: 8px !important;
  min-height: 52px !important;
  font-size: 16px !important;
  font-weight: 700 !important;
  letter-spacing: 0;
  box-shadow: 0 12px 24px rgba(15, 118, 110, 0.18) !important;
}

button.primary-action {
  background: linear-gradient(135deg, #0f766e, #12a594) !important;
  color: #ffffff !important;
  border: none !important;
}

button.secondary-action {
  background: linear-gradient(135deg, #e85d4f, #ff8a66) !important;
  color: #ffffff !important;
  border: none !important;
}

.compact-note {
  color: var(--muted);
  font-size: 14px;
  margin: -2px 0 8px;
}

.visual-stack,
.control-stack,
.output-stack {
  min-height: 0;
  overflow: visible;
}

.visual-stack {
  flex: 1.05 1 0 !important;
}

.control-stack {
  flex: 0 0 330px !important;
  max-width: 330px !important;
}

.output-stack {
  flex: 1.75 1 0 !important;
}

.visual-stack .compact-panel,
.output-stack .compact-panel {
  min-height: 720px;
}

.visual-stack .compact-panel {
  height: 720px;
}

.preview-col {
  flex: 7 1 0 !important;
}

.crop-col {
  flex: 3 1 0 !important;
}

.visual-stack .gr-image,
.visual-stack .image-container {
  border-radius: 8px !important;
}

.visual-stack .gr-image,
.output-stack textarea {
  background: var(--field) !important;
  border-color: rgba(15, 118, 110, 0.18) !important;
}

.identity-panel textarea,
.identity-panel input,
.setup-row textarea,
.setup-row input {
  min-height: 42px !important;
}

.output-grid {
  gap: 10px !important;
}

.output-panel textarea,
.output-panel pre,
.output-panel .scroll-hide {
  font-family: "IBM Plex Mono", "Menlo", monospace !important;
  font-size: 16px !important;
  line-height: 1.5 !important;
  color: var(--ink) !important;
}

.output-panel textarea {
  min-height: 600px !important;
  box-shadow: inset 0 0 26px rgba(15, 118, 110, 0.06);
}

.summary-panel {
  min-height: 600px;
  background: var(--field);
  border: 1px solid rgba(15, 118, 110, 0.18);
  border-radius: 8px;
  padding: 12px;
  box-shadow: inset 0 0 26px rgba(15, 118, 110, 0.06);
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.summary-item {
  display: grid;
  grid-template-columns: 44px 1fr;
  gap: 12px;
  align-items: flex-start;
  min-height: 118px;
  padding: 14px;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(15, 118, 110, 0.16);
  border-left: 5px solid var(--accent);
  border-radius: 8px;
}

.summary-item.summary {
  border-left-color: #2563eb;
}

.summary-item.actions {
  border-left-color: var(--accent);
}

.summary-item.risk {
  border-left-color: #c2410c;
}

.summary-item.anomaly {
  border-left-color: var(--accent-2);
}

.summary-item.advice {
  border-left-color: var(--accent-3);
  grid-column: 1 / -1;
}

.summary-icon {
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--field-strong);
  border: 1px solid rgba(15, 118, 110, 0.14);
  border-radius: 8px;
  font-size: 23px;
}

.summary-name {
  margin: 0 0 6px;
  color: var(--accent);
  font-size: 14px;
  font-weight: 800;
}

.summary-value {
  color: var(--ink);
  font-size: 16px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.summary-empty {
  min-height: 574px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  border: 1px dashed rgba(15, 118, 110, 0.28);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.46);
  font-size: 16px;
  text-align: center;
}

.output-stack {
  min-width: 0;
}

label,
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .info {
  color: var(--muted) !important;
  font-size: 15px !important;
  line-height: 1.35 !important;
}

input,
textarea,
select,
.gradio-container .wrap {
  color: var(--ink) !important;
  font-size: 16px !important;
}

.gradio-container .prose,
.gradio-container p,
.gradio-container span {
  font-size: 15px;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: var(--field) !important;
  border-color: rgba(15, 118, 110, 0.18) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: #8aa09f !important;
}

.gradio-container .block-title,
.gradio-container .accordion {
  color: var(--ink) !important;
}

.gr-form,
.gr-block {
  border-radius: 8px !important;
}

footer {
  display: none !important;
}

@media (max-height: 860px) {
  .app-header {
    min-height: 54px;
    padding: 8px 12px;
  }
  .app-header h1 {
    font-size: 24px;
  }
  .output-panel textarea {
    min-height: 430px !important;
  }
}

@media (max-width: 1280px) {
  .app-shell {
    height: auto;
    overflow: auto;
  }
  .output-stack {
    min-width: 0;
  }
}
"""


def _get_detector(weights):
    """Return a cached YOLO detector instance for the given weights path.

    Args:
        weights: Model identifier or local weights path accepted by Ultralytics.

    Returns:
        A reusable ``YOLO`` detector instance.
    """
    model = _DETECTOR_CACHE.get(weights)
    if model is None:
        model = YOLO(weights)
        _DETECTOR_CACHE[weights] = model
    return model


def list_videos(root_dir):
    """Recursively collect video files under a root directory.

    Args:
        root_dir: Directory that may contain videos in nested subfolders.

    Returns:
        A sorted list of absolute video file paths.
    """
    exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return sorted(files)


def list_video_choices(video_root):
    """Build UI-friendly relative video choices for the dropdown.

    Args:
        video_root: Root directory used by the single-video app.

    Returns:
        A list of paths relative to ``video_root``. Returns an empty list if the
        root does not exist.
    """
    if not video_root or not os.path.isdir(video_root):
        return []
    return [os.path.relpath(path, video_root) for path in list_videos(video_root)]


def resolve_video_path(video_root, selected_video):
    """Resolve the selected dropdown value into a filesystem path.

    Args:
        video_root: Root directory that stores all candidate videos.
        selected_video: Relative path from the dropdown or an absolute path.

    Returns:
        An absolute normalized video path, or ``None`` if the inputs are empty.
    """
    if not video_root or not selected_video:
        return None
    if os.path.isabs(selected_video):
        return selected_video
    return os.path.normpath(os.path.join(video_root, selected_video))


def resolve_project_db_path(video_path, project_db_dir=PROJECT_DB_DIR_DEFAULT):
    """Map a video file to its project-local SQLite database path.

    Args:
        video_path: Absolute path of the selected video file.
        project_db_dir: Directory where per-video databases are stored.

    Returns:
        The database path ``<project_db_dir>/<video_stem>.db``. The directory is
        created if needed.
    """
    os.makedirs(project_db_dir, exist_ok=True)
    video_name = Path(video_path).stem
    return os.path.join(project_db_dir, f"{video_name}.db")


def read_frame(video_path, frame_idx=None):
    """Read one RGB frame from a video with multiple fallbacks.

    Args:
        video_path: Video file path.
        frame_idx: Optional target frame index. If ``None``, a frame around the
            first third of the video is used.

    Returns:
        A numpy RGB image array, or ``None`` if no frame can be decoded.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        if total == 0:
            return None
        if frame_idx is None:
            frame_idx = max(0, total // 3)
        frame_idx = min(max(0, frame_idx), total - 1)
        try:
            return vr[frame_idx].asnumpy()
        except Exception:
            try:
                return vr[0].asnumpy()
            except Exception:
                return _read_first_frame_imageio(video_path)
    except Exception:
        frame = _read_frame_imageio(video_path, frame_idx)
        return frame if frame is not None else _read_first_frame_imageio(video_path)


def _read_frame_imageio(video_path, frame_idx=None):
    """Read a frame with ``imageio`` as a decord fallback.

    Args:
        video_path: Video file path.
        frame_idx: Optional frame index. Defaults to the first frame.

    Returns:
        A numpy image array, or ``None`` if decoding fails.
    """
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
    """Read the first decodable frame with ``imageio`` iteration.

    Args:
        video_path: Video file path.

    Returns:
        A numpy image array for the first frame, or ``None`` on failure.
    """
    try:
        import imageio.v3 as iio
    except Exception:
        return None
    try:
        it = iio.imiter(video_path)
        return next(it, None)
    except Exception:
        return None


def draw_boxes(frame_rgb, bboxes):
    """Draw indexed green bounding boxes on an RGB frame.

    Args:
        frame_rgb: Source RGB frame as a numpy array.
        bboxes: Iterable of ``[x1, y1, x2, y2]`` boxes.

    Returns:
        A new numpy RGB image array with rectangles and box indices rendered.
    """
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, y1 + 4), str(i), fill=(0, 255, 0))
    return np.array(img)


def _clip_bbox(bbox, width, height):
    """Clamp a bounding box so it stays inside image bounds.

    Args:
        bbox: Box-like sequence ``[x1, y1, x2, y2]``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A clipped integer box ``[x1, y1, x2, y2]``.
    """
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
    """Expand and regularize a detector box for preview selection.

    Args:
        frame_rgb: Source RGB frame.
        bbox: Raw detector box.
        keypoints: Optional pose keypoints used to enlarge the box.
        pad_x: Horizontal expansion factor.
        pad_top: Upper vertical expansion factor.
        pad_bottom: Lower vertical expansion factor.
        target_ratio: Desired width/height ratio for the preview crop.

    Returns:
        A clipped integer box suitable for preview and ReID cropping.
    """
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)

    if keypoints is not None:
        points = np.asarray(keypoints)
        if points.ndim == 2 and points.shape[1] >= 2:
            if points.shape[1] >= 3:
                points = points[points[:, 2] > 0.2]
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
    """Detect person boxes on a single RGB frame.

    Args:
        frame_rgb: RGB image array used for detection.
        detector_weights: Ultralytics weights name or path.
        detector_imgsz: Inference image size.
        detector_conf: Detection confidence threshold.

    Returns:
        A list of refined person boxes in ``[x1, y1, x2, y2]`` format.
    """
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
    """Resolve a click location to the most relevant detected box.

    Args:
        bboxes: Candidate person boxes.
        x: Click x-coordinate.
        y: Click y-coordinate.

    Returns:
        The selected box index, or ``None`` if no candidates are available.
    """
    if not bboxes:
        return None
    best_idx = None
    best_score = -1e9
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        inside = x1 <= x <= x2 and y1 <= y <= y2
        if inside:
            score = (x2 - x1) * (y2 - y1)
        else:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            score = -((x - cx) ** 2 + (y - cy) ** 2)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def extract_reid_feature(frame_rgb, bbox):
    """Extract one normalized ReID feature from a frame crop.

    Args:
        frame_rgb: RGB frame as a numpy array.
        bbox: Crop box in ``[x1, y1, x2, y2]`` format.

    Returns:
        A single feature vector, or ``None`` when extraction fails.
    """
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
    """Average ReID features from several nearby frames around one location.

    Args:
        video_path: Video file path.
        frame_idx: Anchor frame index.
        bbox: Target crop box.
        offsets: Relative frame offsets sampled around ``frame_idx``.

    Returns:
        An averaged and normalized ReID feature vector, or ``None`` if all
        attempts fail.
    """
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


def build_dense_log(person_id, db_path, date_filter=None, video_names=None):
    """Build a dense action timeline from the per-video SQLite database.

    Args:
        person_id: Target person identifier.
        db_path: SQLite database path for the selected video.
        date_filter: Optional date string used to keep matching OCR timestamps.
        video_names: Optional list of accepted video names or stems.

    Returns:
        A newline-joined action log string, or ``None`` when no valid records
        are found.
    """
    if not db_path or not os.path.exists(db_path):
        return None

    invalid_labels = {"buffering", "None", "Unknown", "Buffering", "Uncertain", ""}
    valid_video_names = set()
    if video_names:
        for name in video_names:
            if not name:
                continue
            valid_video_names.add(str(name))
            valid_video_names.add(Path(str(name)).stem)

    normalized_filter = _normalize_date(date_filter) if date_filter else None

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

    lines = []
    for ocr_time, timestamp, action, video_name in rows:
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


def generate_log(db_path, person_id, date_filter, video_path, summary_model_name):
    """Generate a textual summary for one person in one selected video.

    Args:
        db_path: SQLite database path.
        person_id: Selected person identifier.
        date_filter: Optional date filter forwarded to log construction.
        video_path: Selected video file path.
        summary_model_name: Name of the summary backend to use.

    Returns:
        A summary string or a user-facing error message.
    """
    if person_id is None:
        return empty_summary_cards("Select a person first.")
    video_names = []
    if video_path:
        base = os.path.basename(video_path)
        video_names.extend([base, os.path.splitext(base)[0]])
    log = build_dense_log(
        person_id,
        db_path=db_path,
        date_filter=date_filter,
        video_names=video_names,
    )
    if not log:
        return empty_summary_cards("No valid activity log was found.")
    if summary_model_name not in AVAILABLE_SUMMARY_MODELS:
        return empty_summary_cards(f"Unsupported summary model: {summary_model_name}")
    summary = generate_on_chip_summary(person_id, log, model_name=summary_model_name)
    if not is_valid_on_chip_response(summary):
        return empty_summary_cards(summary or "Summary model did not return a valid summary.")
    return render_on_chip_summary(summary)


def load_video_state(
    video_path,
    frame_idx,
    detector_weights=PREVIEW_DETECTOR_WEIGHTS_DEFAULT,
    detector_imgsz=PREVIEW_DETECTOR_IMGSZ_DEFAULT,
    detector_conf=0.35,
):
    """Load one preview frame and detect clickable person boxes.

    Args:
        video_path: Selected video file path.
        frame_idx: Optional frame index for preview loading.
        detector_weights: Ultralytics weights name or path.
        detector_imgsz: Inference image size.
        detector_conf: Detection confidence threshold.

    Returns:
        A tuple ``(frame, bboxes, visualization, message)`` where ``frame`` and
        ``visualization`` are numpy RGB arrays and ``bboxes`` is a list of boxes.
    """
    if not video_path:
        return None, None, None, "Choose a video first."
    frame = read_frame(video_path, frame_idx)
    if frame is None:
        return None, None, None, "Failed to read the preview frame."
    bboxes = detect_bboxes(
        frame,
        detector_weights=detector_weights,
        detector_imgsz=detector_imgsz,
        detector_conf=detector_conf,
    )
    vis = draw_boxes(frame, bboxes) if bboxes else frame
    return frame, bboxes, vis, f"{os.path.basename(video_path)} preview detected {len(bboxes)} people."


def on_image_click(evt: gr.SelectData, frame, bboxes):
    """Convert a preview click into a selected box and crop.

    Args:
        evt: Gradio selection event carrying click coordinates.
        frame: The current preview frame as a numpy array.
        bboxes: Detected boxes for the preview frame.

    Returns:
        A tuple ``(bbox, crop, message)`` where ``bbox`` is the selected box and
        ``crop`` is the selected person image.
    """
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


def choose_person_id(db_path, frame, bbox, text_prompt, video_path, frame_idx):
    """Match the selected person to an existing ID or create a new one.

    Args:
        db_path: SQLite database path for the selected video.
        frame: Current preview frame.
        bbox: Selected person box.
        text_prompt: Optional free-form description from the UI.
        video_path: Selected video file path.
        frame_idx: Preview frame index used as the ReID anchor.

    Returns:
        A tuple ``(person_id, message)`` describing the resolved identity.
    """
    if not db_path or not video_path:
        return None, "Video path or DB path is invalid."
    db = VideoDB(db_path)
    if bbox is None:
        db.close()
        return None, "Select a person from the preview image first."

    offsets = [0, 5, -5, 10, -10]
    feat = extract_reid_feature_multi(video_path, frame_idx or 0, bbox, offsets)
    if feat is None:
        feat = extract_reid_feature(frame, bbox)
    if feat is None:
        db.close()
        return None, "Failed to extract the ReID feature."

    base = os.path.basename(video_path)
    candidates = db.search_by_image_feature_filtered(
        feat,
        top_k=5,
        threshold=0.35,
        video_names=[base, os.path.splitext(base)[0]],
    )
    if candidates:
        best_id = candidates[0]["person_id"]
        db.close()
        return best_id, f"Matched existing person_id={best_id} from {base}."

    db.cursor.execute("SELECT MAX(person_id) FROM frames")
    row = db.cursor.fetchone()
    max_id = row[0] if row and row[0] is not None else 0
    new_id = int(max_id) + 1
    db.close()
    return new_id, f"No existing match found in {base}. Created person_id={new_id}."


def build_app(video_root, project_db_dir):
    """Build the Gradio single-video application.

    Args:
        video_root: Root directory that contains processed videos.
        project_db_dir: Directory where per-video databases are stored.

    Returns:
        A configured ``gr.Blocks`` application instance.
    """
    videos = list_video_choices(video_root)

    with gr.Blocks(css=APP_CSS, title="Single Video Activity Summary") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="app-header">
                  <div>
                    <h1>Single Video Activity Summary</h1>
                    <p>Choose one video, confirm the target person, then generate the report and action timeline side by side.</p>
                  </div>
                  <div class="app-badge">Single-video workflow</div>
                </div>
                """
            )

            with gr.Row(elem_classes=["setup-row"]):
                with gr.Group(elem_classes=["panel", "compact-panel"]):
                    gr.HTML('<div class="section-title">Session Setup</div>')
                    with gr.Row():
                        video_dropdown = gr.Dropdown(
                            choices=videos,
                            label="Video",
                            info="Pick exactly one processed video.",
                            scale=5,
                        )
                        date_filter = gr.Textbox(
                            label="Date Hint",
                            placeholder="2019-06-22",
                            scale=1,
                        )
                        frame_idx = gr.Number(value=None, precision=0, label="Frame Index", scale=1)
                        load_btn = gr.Button("Load Preview", elem_classes=["primary-action"], scale=1)
                    with gr.Row():
                        db_path = gr.Textbox(label="DB Path", interactive=False, scale=3)
                        status = gr.Textbox(label="Status", interactive=False, scale=2)

            with gr.Row(elem_classes=["work-row"]):
                with gr.Column(scale=6, elem_classes=["visual-stack"]):
                    with gr.Group(elem_classes=["panel", "compact-panel"]):
                        gr.HTML('<div class="section-title">Visual Selection</div>')
                        gr.Markdown(
                            "<div class='compact-note'>Load a preview frame, click the target person, and keep all downstream work scoped to this video.</div>"
                        )
                        with gr.Row():
                            with gr.Column(scale=7, elem_classes=["preview-col"]):
                                image = gr.Image(label="Preview Frame", type="numpy", height=635)
                            with gr.Column(scale=3, elem_classes=["crop-col"]):
                                crop_preview = gr.Image(label="Selected Crop", type="numpy", height=635)

                with gr.Column(scale=3, elem_classes=["control-stack"]):
                    with gr.Group(elem_classes=["panel", "compact-panel", "identity-panel"]):
                        gr.HTML('<div class="section-title">Identity And Run</div>')
                        text_prompt = gr.Textbox(
                            label="Description",
                            placeholder="Optional note about the selected person.",
                            lines=1,
                        )
                        result = gr.Textbox(label="Match Result", interactive=False, lines=2)
                        with gr.Row():
                            select_btn = gr.Button("Confirm Identity", elem_classes=["primary-action"])
                            log_btn = gr.Button("Generate Summary", elem_classes=["secondary-action"])
                            track_btn = gr.Button("Track And Write To DB", elem_classes=["primary-action"])

                    with gr.Group(elem_classes=["panel", "compact-panel"]):
                        gr.HTML('<div class="section-title">Summary Model</div>')
                        summary_model = gr.Dropdown(
                            choices=AVAILABLE_SUMMARY_MODELS,
                            value=DEFAULT_SUMMARY_MODEL,
                            label="Summary Generation Model",
                            info="DLER uses a shorter prompt because the U250 deployment only supports about 600 total tokens.",
                        )

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

                with gr.Column(scale=11, elem_classes=["output-stack"]):
                    with gr.Group(elem_classes=["panel", "compact-panel"]):
                        gr.HTML('<div class="section-title">Outputs</div>')
                        gr.Markdown(
                            "<div class='compact-note'>Summary and action timeline stay visible together, so there is no tab switching during review.</div>"
                        )
                        with gr.Row(elem_classes=["output-grid"]):
                            log_all = gr.HTML(value=empty_summary_cards())
                            timeline_all = gr.Textbox(
                                label="Action Timeline",
                                lines=28,
                                elem_classes=["output-panel"],
                            )

        frame_state = gr.State()
        bboxes_state = gr.State()
        bbox_state = gr.State()
        feature_state = gr.State()
        person_id_state = gr.State()
        video_path_state = gr.State()

        def _resolve_session(video_relpath):
            """Resolve the selected dropdown item into runtime session paths.

            Args:
                video_relpath: Relative path selected from the video dropdown.

            Returns:
                A tuple ``(video_path, db_path, message)`` used by the UI state.
            """
            video_path = resolve_video_path(video_root, video_relpath)
            if not video_path or not os.path.exists(video_path):
                return None, "", "Choose a valid video."
            db_path = resolve_project_db_path(video_path, project_db_dir)
            if os.path.exists(db_path):
                msg = f"Using existing DB: {db_path}"
            else:
                msg = f"DB will be created on demand: {db_path}"
            return video_path, db_path, msg

        def _load(video_relpath, frame_idx, detector_weights, detector_imgsz, detector_conf):
            """Load preview data and session metadata for the selected video.

            Args:
                video_relpath: Relative video path from the dropdown.
                frame_idx: Optional preview frame index.
                detector_weights: Detector weights name or path.
                detector_imgsz: Detector image size.
                detector_conf: Detector confidence threshold.

            Returns:
                A tuple containing frame state, box state, preview image, status
                text, resolved video path, and resolved database path.
            """
            video_path, db_path, session_msg = _resolve_session(video_relpath)
            if not video_path:
                return None, None, None, session_msg, None, ""
            frame, bboxes, vis, msg = load_video_state(
                video_path,
                frame_idx,
                detector_weights=detector_weights,
                detector_imgsz=detector_imgsz,
                detector_conf=detector_conf,
            )
            status_msg = f"{msg} | {session_msg}"
            return frame, bboxes, vis, status_msg, video_path, db_path

        load_btn.click(
            _load,
            inputs=[video_dropdown, frame_idx, detector_weights, detector_imgsz, detector_conf],
            outputs=[frame_state, bboxes_state, image, status, video_path_state, db_path],
        )

        def _on_click(frame, bboxes, detector_weights, detector_imgsz, detector_conf, evt: gr.SelectData):
            """Handle a click on the preview image and extract a ReID feature.

            Args:
                frame: Current preview frame.
                bboxes: Current list of detected boxes.
                detector_weights: Detector weights name or path.
                detector_imgsz: Detector image size.
                detector_conf: Detector confidence threshold.
                evt: Gradio selection event.

            Returns:
                A tuple ``(bbox, crop, feature, message)`` for UI state updates.
            """
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
            inputs=[db_path, frame_state, bbox_state, text_prompt, video_path_state, frame_idx],
            outputs=[person_id_state, result],
        )

        log_btn.click(
            generate_log,
            inputs=[db_path, person_id_state, date_filter, video_path_state, summary_model],
            outputs=[log_all],
        )

        def _track_person(
            video_path,
            db_path,
            person_id,
            target_feat,
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
            """Track one selected person through a single video and write results.

            Args:
                video_path: Selected video path.
                db_path: Target SQLite database path.
                person_id: Existing or newly assigned person identifier.
                target_feat: ReID feature for the selected person.
                reid_interval: Frame interval for ReID refresh.
                action_interval: Frame interval for tracking/action updates.
                ocr_interval_sec: OCR sampling interval in seconds.
                detector_weights: Detector weights name or path.
                detector_imgsz: Detector image size.
                detector_conf: Detector confidence threshold.
                action_model_path: Vision-language action model identifier.
                action_window_sec: Action recognition window size in seconds.
                action_frames: Number of frames passed to the action model.
                motion_trigger_threshold: Threshold for label reuse vs. new action
                    prediction.

            Yields:
                Tuples ``(status_message, timeline_text)`` for streaming UI
                updates during tracking.
            """
            if not video_path:
                yield "No video selected.", "No valid action timeline."
                return
            if target_feat is None:
                yield "No person selected.", "No valid action timeline."
                return

            db = VideoDB(db_path, fast_write=True)
            if person_id is None:
                base = os.path.basename(video_path)
                candidates = db.search_by_image_feature_filtered(
                    target_feat,
                    top_k=1,
                    threshold=0.35,
                    video_names=[base, os.path.splitext(base)[0]],
                )
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
            saved_check = False
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
            except Exception:
                db.close()
                yield f"Failed to open {video_path}", "No valid action timeline."
                return

            fps = vr.get_avg_fps() or 25.0
            ocr_interval = max(1, int(fps * float(ocr_interval_sec)))
            window_frames = max(1, int(fps * float(action_window_sec)))
            target_frames = max(1, int(action_frames))
            stride = max(1, window_frames // target_frames)
            current_ocr_text = ""
            last_reid_features = {}
            target_track_id = None
            clip_buffer = []
            last_bbox = None
            last_action_label = None

            yield f"Processing {os.path.basename(video_path)}", ""

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
                motion_score = 0.0
                if last_bbox is not None:
                    px1, py1, px2, py2 = last_bbox
                    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
                    motion_score = ((cx - pcx) / pw) ** 2 + ((cy - pcy) / ph) ** 2
                last_bbox = (x1, y1, x2, y2)

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
                    action_label = action_rec.predict_clip([img for _, _, img in selected])
                    last_action_label = action_label

                clip_buffer = []
                if action_label in ("buffering", "Buffering", "Uncertain", "Unknown", "", None):
                    continue

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

                if not saved_check:
                    try:
                        os.makedirs("check_buffer", exist_ok=True)
                        img = Image.fromarray(frame_rgb)
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                        draw.text((x1 + 4, y1 + 4), f"ID:{person_id}", fill=(255, 0, 0))
                        img.save(os.path.join("check_buffer", f"{Path(video_path).stem}_frame{frame_idx:06d}_ID{person_id}.jpg"))
                        saved_check = True
                    except Exception:
                        pass

                ts_label = current_ocr_text if current_ocr_text else f"T+{int(frame_idx / fps):04d}s"
                timeline_lines.append(f"[{ts_label}] {action_label}")
                if len(timeline_lines) % 10 == 0:
                    db.commit()
                    yield f"Processing {os.path.basename(video_path)} ...", "\n".join(timeline_lines[-200:])

            db.commit()
            db.close()
            if timeline_lines:
                yield f"Completed: person_id={person_id}", "\n".join(timeline_lines[-500:])
            else:
                yield "The target person was not found in the selected video.", "No valid action timeline."

        track_btn.click(
            _track_person,
            inputs=[
                video_path_state,
                db_path,
                person_id_state,
                feature_state,
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
    """Run the single-video Gradio app from the command line.

    Returns:
        ``None``. This function launches the web UI server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default=VIDEO_ROOT_DEFAULT)
    parser.add_argument("--db_dir", default=PROJECT_DB_DIR_DEFAULT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    app = build_app(args.video_root, args.db_dir)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
