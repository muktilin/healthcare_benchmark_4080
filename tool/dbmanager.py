import io
import json
import sqlite3
from typing import Iterable, List, Optional, Sequence

import numpy as np


class VideoDB:
    """
    SQLite wrapper for video frame metadata + ReID features.
    """

    def __init__(self, db_path: str = "video_data.db", fast_write: bool = False):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        if fast_write:
            self.cursor.execute("PRAGMA synchronous=OFF")
            self.cursor.execute("PRAGMA journal_mode=OFF")
            self.cursor.execute("PRAGMA temp_store=MEMORY")
        self._create_table()

    def _create_table(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT,
                frame_idx INTEGER,
                timestamp REAL,
                ocr_time TEXT,
                person_id INTEGER,
                action TEXT,
                bbox TEXT,
                keypoints BLOB,
                reid_feature BLOB
            )
            """
        )
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_person_id ON frames(person_id)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_action ON frames(action)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_name ON frames(video_name)")
        self.conn.commit()

    def _adapt_array(self, arr: Optional[np.ndarray]) -> Optional[bytes]:
        if arr is None:
            return None
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return out.read()

    def _convert_array(self, blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        out = io.BytesIO(blob)
        return np.load(out)

    def add_entry(self, entry: dict) -> None:
        bbox = entry["bbox"]
        bbox_json = json.dumps(bbox.tolist() if isinstance(bbox, np.ndarray) else bbox)
        kpts_blob = self._adapt_array(entry.get("keypoints"))
        reid_blob = self._adapt_array(entry.get("reid_feature"))

        self.cursor.execute(
            """
            INSERT INTO frames (video_name, frame_idx, timestamp, ocr_time, person_id, action, bbox, keypoints, reid_feature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["video_name"],
                entry["frame_idx"],
                entry["timestamp"],
                entry.get("ocr_time", ""),
                entry["person_id"],
                entry["action"],
                bbox_json,
                kpts_blob,
                reid_blob,
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # =====================
    # Retrieval
    # =====================

    def get_trajectory(self, person_id: int) -> List[dict]:
        self.cursor.execute(
            """
            SELECT frame_idx, bbox, action
            FROM frames
            WHERE person_id = ?
            ORDER BY frame_idx
            """,
            (person_id,),
        )
        results = []
        for row in self.cursor.fetchall():
            results.append({"frame": row[0], "bbox": json.loads(row[1]), "action": row[2]})
        return results

    def search_by_action(self, action_query: str) -> list:
        self.cursor.execute(
            """
            SELECT DISTINCT person_id, video_name, min(timestamp), max(timestamp)
            FROM frames
            WHERE action = ?
            GROUP BY person_id, video_name
            """,
            (action_query,),
        )
        return self.cursor.fetchall()

    def _fetch_reid_rows(self, video_names: Optional[Sequence[str]] = None) -> list:
        if video_names:
            placeholders = ",".join(["?"] * len(video_names))
            sql = f"""
                SELECT person_id, reid_feature
                FROM frames
                WHERE reid_feature IS NOT NULL AND video_name IN ({placeholders})
            """
            self.cursor.execute(sql, tuple(video_names))
            return self.cursor.fetchall()
        self.cursor.execute("SELECT person_id, reid_feature FROM frames WHERE reid_feature IS NOT NULL")
        return self.cursor.fetchall()

    def _score_reid(self, query_feature: np.ndarray, rows: list) -> list:
        db_ids = []
        db_feats = []
        for pid, blob in rows:
            feat = self._convert_array(blob)
            if feat is not None and feat.shape == query_feature.shape:
                db_ids.append(pid)
                db_feats.append(feat)
        if not db_feats:
            return []
        db_feats_mat = np.stack(db_feats)
        scores = np.dot(db_feats_mat, query_feature)
        return list(zip(db_ids, scores))

    def _topk_unique(self, scored: list, top_k: int, threshold: float) -> list:
        scored.sort(key=lambda x: x[1], reverse=True)
        unique = []
        seen = set()
        for pid, score in scored:
            if score < threshold:
                break
            if pid in seen:
                continue
            unique.append({"person_id": int(pid), "score": float(score)})
            seen.add(pid)
            if len(unique) >= top_k:
                break
        return unique

    def search_by_image_feature(
        self, query_feature: np.ndarray, top_k: int = 5, threshold: float = 0.5
    ) -> list:
        rows = self._fetch_reid_rows()
        if not rows:
            return []
        scored = self._score_reid(query_feature, rows)
        return self._topk_unique(scored, top_k, threshold)

    def search_by_image_feature_filtered(
        self,
        query_feature: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.5,
        video_names: Optional[Sequence[str]] = None,
    ) -> list:
        rows = self._fetch_reid_rows(video_names=video_names)
        if not rows:
            return []
        scored = self._score_reid(query_feature, rows)
        return self._topk_unique(scored, top_k, threshold)


if __name__ == "__main__":
    db = VideoDB("test_db.sqlite")
    dummy_feat = np.random.rand(512).astype(np.float32)
    dummy_feat /= np.linalg.norm(dummy_feat)

    entry = {
        "video_name": "test.mp4",
        "frame_idx": 100,
        "timestamp": 3.5,
        "ocr_time": "2019-06-22",
        "person_id": 1,
        "action": "walking",
        "bbox": np.array([100, 100, 200, 200]),
        "keypoints": np.zeros((17, 3)),
        "reid_feature": dummy_feat,
    }

    db.add_entry(entry)
    db.commit()
    print("Inserted dummy data.")
    print("Search Action 'walking':", db.search_by_action("walking"))
    print("Search by Feature:", db.search_by_image_feature(dummy_feat))
    db.close()
