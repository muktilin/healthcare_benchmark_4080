import io
import sqlite3
from collections import defaultdict

import numpy as np


def _load_feature(blob):
    if blob is None:
        return None
    buf = io.BytesIO(blob)
    return np.load(buf)


def reconcile_ids(db_path, threshold=0.75, min_samples=5):
    """
    Post-merge global ID reconciliation by clustering person_id centroids.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT person_id, reid_feature FROM frames WHERE reid_feature IS NOT NULL")
    rows = cur.fetchall()
    if not rows:
        conn.close()
        return

    sums = {}
    counts = defaultdict(int)
    for pid, blob in rows:
        feat = _load_feature(blob)
        if feat is None:
            continue
        if pid not in sums:
            sums[pid] = feat.astype(np.float32)
        else:
            sums[pid] += feat
        counts[pid] += 1

    # build centroids
    pids = [pid for pid in sums.keys() if counts[pid] >= min_samples]
    if not pids:
        conn.close()
        return
    centroids = {}
    for pid in pids:
        c = sums[pid] / max(1, counts[pid])
        norm = np.linalg.norm(c)
        centroids[pid] = c / norm if norm > 0 else c

    # greedy clustering by cosine threshold
    assigned = set()
    clusters = []
    pid_list = list(centroids.keys())
    for i, pid in enumerate(pid_list):
        if pid in assigned:
            continue
        cluster = [pid]
        assigned.add(pid)
        ci = centroids[pid]
        for j in range(i + 1, len(pid_list)):
            other = pid_list[j]
            if other in assigned:
                continue
            score = float(np.dot(ci, centroids[other]))
            if score >= threshold:
                cluster.append(other)
                assigned.add(other)
        clusters.append(cluster)

    # remap IDs to smallest in cluster
    updates = []
    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        canonical = min(cluster)
        for pid in cluster:
            if pid != canonical:
                updates.append((canonical, pid))

    if updates:
        cur.executemany("UPDATE frames SET person_id = ? WHERE person_id = ?", updates)
        conn.commit()

    conn.close()
