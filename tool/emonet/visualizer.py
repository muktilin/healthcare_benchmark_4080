from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from tool.emonet.models import EmoNet


EMOTION_CLASSES_8 = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
}

LANDMARK_GROUPS = [
    ("jaw", [(i, i + 1) for i in range(0, 16)], (105, 220, 235)),
    ("brow", [(i, i + 1) for i in range(17, 21)] + [(i, i + 1) for i in range(22, 26)], (248, 196, 87)),
    ("nose", [(i, i + 1) for i in range(27, 30)] + [(i, i + 1) for i in range(31, 35)] + [(30, 35)], (126, 177, 255)),
    ("eyes", [(i, i + 1) for i in range(36, 41)] + [(41, 36)] + [(i, i + 1) for i in range(42, 47)] + [(47, 42)], (111, 238, 179)),
    ("mouth", [(i, i + 1) for i in range(48, 59)] + [(59, 48)] + [(i, i + 1) for i in range(60, 67)] + [(67, 60)], (255, 132, 132)),
]


class EmoNetTrackerVisualizer:
    def __init__(self, device=None, n_expression=8, image_size=256):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_expression = int(n_expression)
        self.image_size = int(image_size)
        self.root = Path(__file__).resolve().parent
        self.model = self._load_model()
        self.circumplex = cv2.imread(str(self.root / "images" / "circumplex.png"))

    def _load_model(self):
        state_path = self.root / "pretrained" / f"emonet_{self.n_expression}.pth"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing EmoNet weights: {state_path}")
        state_dict = torch.load(str(state_path), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model = EmoNet(n_expression=self.n_expression).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def predict(self, face_crop_rgb):
        if face_crop_rgb is None or face_crop_rgb.size == 0:
            return None
        resized = cv2.resize(face_crop_rgb, (self.image_size, self.image_size))
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(self.device) / 255.0
        with torch.no_grad():
            return self.model(image_tensor.unsqueeze(0))

    def visualize(self, frame_rgb, person_bbox, keypoints=None, label_prefix=None):
        h, w = frame_rgb.shape[:2]
        face_bbox = estimate_face_bbox(person_bbox, keypoints, w, h)
        x1, y1, x2, y2 = face_bbox
        face_crop = frame_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            return frame_rgb

        prediction = self.predict(face_crop.copy())
        if prediction is None:
            return frame_rgb

        expression_idx = int(
            torch.argmax(nn.functional.softmax(prediction["expression"], dim=1)).cpu().item()
        )
        expression = EMOTION_CLASSES_8.get(expression_idx, str(expression_idx))
        valence = float(prediction["valence"].detach().cpu().clamp(-1.0, 1.0).item())
        arousal = float(prediction["arousal"].detach().cpu().clamp(-1.0, 1.0).item())

        annotated = frame_rgb.copy()
        px1, py1, px2, py2 = [int(v) for v in person_bbox]
        draw_box(annotated, (px1, py1, px2, py2), (245, 78, 78), "target")
        draw_box(annotated, (x1, y1, x2, y2), (250, 202, 82), "face")

        text = f"{expression} V:{valence:+.2f} A:{arousal:+.2f}"
        if label_prefix:
            text = f"{label_prefix} | {text}"
        cv2.putText(
            annotated,
            text,
            (max(6, px1), max(24, py1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 64, 64),
            2,
            cv2.LINE_AA,
        )

        heatmap = torch.nn.functional.interpolate(
            prediction["heatmap"],
            (face_crop.shape[0], face_crop.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        landmark_crop = face_crop.copy()
        landmarks = heatmap_to_landmarks(heatmap)
        draw_landmark_mesh(landmark_crop, landmarks)
        draw_landmark_mesh(annotated, [(x1 + lx, y1 + ly) for lx, ly in landmarks])

        side_panel = self._make_side_panel(
            landmark_crop,
            expression,
            valence,
            arousal,
            annotated.shape[0],
        )

        return np.concatenate([annotated, side_panel], axis=1)

    def _plot_valence_arousal(self, valence, arousal, size):
        return make_valence_arousal_chart(size, valence, arousal)

    def _make_side_panel(self, landmark_crop, expression, valence, arousal, frame_height):
        width = min(max(760, int(frame_height * 1.65)), 900)
        panel = np.zeros((frame_height, width, 3), dtype=np.uint8)
        panel[:] = (10, 18, 21)

        pad = max(12, width // 24)
        title_h = 26
        tile = max(180, min((width - pad * 3) // 2, frame_height - pad * 3 - title_h))
        content_height = tile + pad + title_h
        y = max(pad, (frame_height - content_height) // 2)
        face_x = pad
        chart_x = pad * 2 + tile

        put_panel_title(panel, "FACIAL LANDMARKS", (face_x, y + 18), (154, 238, 229))
        put_panel_title(
            panel,
            f"{expression}  V {valence:+.2f}  A {arousal:+.2f}",
            (chart_x, y + 18),
            (255, 214, 128),
        )
        y += title_h + pad
        face_panel = make_image_tile(landmark_crop, tile)
        chart_panel = self._plot_valence_arousal(valence, arousal, tile)
        panel[y : y + tile, face_x : face_x + tile] = face_panel
        panel[y : y + tile, chart_x : chart_x + tile] = chart_panel
        return panel


def heatmap_to_landmarks(heatmap):
    landmarks = []
    heatmap_cpu = heatmap.detach().cpu()
    for landmark_idx in range(heatmap_cpu.shape[1]):
        channel = heatmap_cpu[0, landmark_idx]
        flat_idx = int(torch.argmax(channel).item())
        y, x = divmod(flat_idx, channel.shape[1])
        landmarks.append((int(x), int(y)))
    return landmarks


def draw_landmark_mesh(image, landmarks):
    if len(landmarks) < 68:
        for lx, ly in landmarks:
            cv2.circle(image, (lx, ly), 2, (154, 238, 229), -1, cv2.LINE_AA)
        return
    overlay = image.copy()
    for _, connections, color in LANDMARK_GROUPS:
        for start, end in connections:
            cv2.line(overlay, landmarks[start], landmarks[end], color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.76, image, 0.24, 0, image)
    for _, connections, color in LANDMARK_GROUPS:
        point_ids = sorted({idx for pair in connections for idx in pair})
        for idx in point_ids:
            cv2.circle(image, landmarks[idx], 2, color, -1, cv2.LINE_AA)
            cv2.circle(image, landmarks[idx], 3, (8, 14, 16), 1, cv2.LINE_AA)


def draw_box(image, bbox, color, label):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    tx2 = x1 + label_size[0] + 10
    if y1 > label_size[1] + 12:
        ty1, ty2 = y1 - label_size[1] - 10, y1
        text_y = y1 - 5
    else:
        ty1, ty2 = y1, y1 + label_size[1] + 10
        text_y = y1 + label_size[1] + 5
    cv2.rectangle(image, (x1, ty1), (tx2, ty2), color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (8, 14, 16), 1, cv2.LINE_AA)


def put_panel_title(image, text, pos, color):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def make_image_tile(image, size):
    tile = np.zeros((size, size, 3), dtype=np.uint8)
    tile[:] = (16, 24, 27)
    h, w = image.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=interpolation,
    )
    blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
    resized = cv2.addWeighted(resized, 1.35, blurred, -0.35, 0)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (46, 78, 82), 1, cv2.LINE_AA)
    return tile


def make_valence_arousal_chart(size, valence, arousal):
    chart = np.zeros((size, size, 3), dtype=np.uint8)
    chart[:] = (10, 18, 21)
    center = size // 2
    radius = int(size * 0.39)

    line_thickness = max(1, size // 180)
    for r, alpha in [(radius, 0.34), (int(radius * 0.66), 0.22), (int(radius * 0.33), 0.16)]:
        overlay = chart.copy()
        cv2.circle(overlay, (center, center), r, (40, 94, 101), line_thickness, cv2.LINE_AA)
        chart = cv2.addWeighted(overlay, alpha, chart, 1 - alpha, 0)

    cv2.line(chart, (center - radius, center), (center + radius, center), (85, 136, 142), line_thickness, cv2.LINE_AA)
    cv2.line(chart, (center, center - radius), (center, center + radius), (85, 136, 142), line_thickness, cv2.LINE_AA)

    labels = [
        ("Arousal", (center - 34, center - radius - 10), (255, 214, 128)),
        ("Calm", (center - 22, center + radius + 22), (128, 203, 255)),
        ("Negative", (center - radius - 4, center - 10), (255, 132, 132)),
        ("Positive", (center + radius - 60, center - 10), (111, 238, 179)),
        ("Happy", (center + int(radius * 0.46), center - int(radius * 0.35)), (226, 240, 236)),
        ("Sad", (center - int(radius * 0.78), center + int(radius * 0.50)), (226, 240, 236)),
        ("Fear", (center - int(radius * 0.55), center - int(radius * 0.55)), (226, 240, 236)),
        ("Relaxed", (center + int(radius * 0.35), center + int(radius * 0.55)), (226, 240, 236)),
    ]
    font_scale = max(0.46, size / 620.0)
    text_thickness = max(1, size // 260)
    for text, pos, color in labels:
        cv2.putText(chart, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)

    x = int(center + valence * radius)
    y = int(center - arousal * radius)
    glow = chart.copy()
    cv2.circle(glow, (x, y), max(12, size // 18), (255, 83, 83), -1, cv2.LINE_AA)
    chart = cv2.addWeighted(glow, 0.22, chart, 0.78, 0)
    cv2.circle(chart, (x, y), max(7, size // 34), (255, 83, 83), -1, cv2.LINE_AA)
    cv2.circle(chart, (x, y), max(7, size // 34), (255, 226, 226), 1, cv2.LINE_AA)
    cv2.rectangle(chart, (0, 0), (size - 1, size - 1), (46, 78, 82), 1, cv2.LINE_AA)
    return chart


def estimate_face_bbox(person_bbox, keypoints, width, height):
    x1, y1, x2, y2 = [float(v) for v in person_bbox]
    person_w = max(1.0, x2 - x1)
    person_h = max(1.0, y2 - y1)
    points = _visible_face_keypoints(keypoints)

    if len(points) >= 2:
        px = points[:, 0]
        py = points[:, 1]
        fx1, fy1, fx2, fy2 = float(px.min()), float(py.min()), float(px.max()), float(py.max())
        span = max(fx2 - fx1, fy2 - fy1, person_w * 0.14, person_h * 0.08)
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        size = span * 2.25
        face = [cx - size / 2, cy - size * 0.58, cx + size / 2, cy + size * 0.42]
    else:
        face = [
            x1 + person_w * 0.18,
            y1,
            x2 - person_w * 0.18,
            y1 + person_h * 0.42,
        ]

    return _clip_bbox(face, width, height)


def _visible_face_keypoints(keypoints):
    if keypoints is None:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 5 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    face = arr[:5]
    if arr.shape[1] >= 3:
        face = face[face[:, 2] > 0.2]
    return face[:, :2] if len(face) else np.empty((0, 2), dtype=np.float32)


def _clip_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), max(0, width - 2)))
    y1 = max(0, min(int(round(y1)), max(0, height - 2)))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return [x1, y1, x2, y2]
