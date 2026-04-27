"""
Clean ByteTrack implementation: bbox-based tracking with motion inertia.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box" (ECCV 2022, arXiv:2110.06864).

Deviations from the paper:
- No Kalman filter. Motion uses a constant-velocity model on (cx, cy, w, h)
  with EMA-smoothed velocity and drag on lost tracks. Fewer knobs, less
  drift on noisy detections.
- Match cost blends IoU and normalized centroid distance so the ordering
  stays meaningful when boxes don't overlap (brief occlusions, fast motion).
- Per-class matching by default — a deer can't inherit a vehicle's track ID.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # xyxy, absolute pixels
    score: float
    cls_id: int


class TrackState(IntEnum):
    TRACKED = 0  # matched this frame
    LOST = 1  # missed recent frames, still within track_buffer
    REMOVED = 2  # permanently gone


def _xyxy_to_cxywh(bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    return np.array([x1 + w / 2, y1 + h / 2, w, h], dtype=np.float64)


def _cxywh_to_xyxy(cxywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = cxywh
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU for xyxy boxes of shape [A, 4] and [B, 4] → [A, B]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    ax = a[:, None, :]
    bx = b[None, :, :]

    inter_x1 = np.maximum(ax[..., 0], bx[..., 0])
    inter_y1 = np.maximum(ax[..., 1], bx[..., 1])
    inter_x2 = np.minimum(ax[..., 2], bx[..., 2])
    inter_y2 = np.minimum(ax[..., 3], bx[..., 3])

    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax[..., 2] - ax[..., 0]) * (ax[..., 3] - ax[..., 1])
    area_b = (bx[..., 2] - bx[..., 0]) * (bx[..., 3] - bx[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def _pairwise_centroid_dist(a: np.ndarray, b: np.ndarray, diag: float) -> np.ndarray:
    """L2 centroid distance in [0, 1], normalized by image diagonal."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ca = np.stack([(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2], axis=1)
    cb = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    diff = ca[:, None, :] - cb[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1)) / max(diag, 1e-6)
    return np.clip(dist, 0.0, 1.0)


class _Track:
    __slots__ = (
        "track_id",
        "cls_id",
        "score",
        "mean",
        "velocity",
        "state",
        "age",
        "hits",
        "time_since_update",
    )

    def __init__(self, track_id: int, bbox, score: float, cls_id: int):
        self.track_id = track_id
        self.cls_id = int(cls_id)
        self.score = float(score)
        self.mean = _xyxy_to_cxywh(bbox)  # last observed (cx, cy, w, h)
        self.velocity = np.zeros(4, dtype=np.float64)

        self.state = TrackState.TRACKED
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

    def predicted_mean(self, drag: float) -> np.ndarray:
        """Linear extrapolation from last observation, damped while lost."""
        if self.state == TrackState.LOST and self.time_since_update > 1:
            damp = drag ** (self.time_since_update - 1)
        else:
            damp = 1.0
        predicted = self.mean + self.velocity * self.time_since_update * damp
        predicted[2] = max(predicted[2], 1.0)
        predicted[3] = max(predicted[3], 1.0)
        return predicted

    def predicted_xyxy(self, drag: float) -> np.ndarray:
        return _cxywh_to_xyxy(self.predicted_mean(drag))

    def predict(self):
        """Age the track one frame. mean/velocity stay at the last observation
        and are extrapolated on demand via predicted_mean()."""
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, score: float, velocity_alpha: float):
        new_mean = _xyxy_to_cxywh(bbox)
        gap = max(self.time_since_update, 1)
        observed_v = (new_mean - self.mean) / gap

        if self.hits < 2:
            # On the first real update, seed velocity directly — EMA blending
            # from zeros would under-shoot and lose inertia for fast movers.
            self.velocity = observed_v
        else:
            self.velocity = velocity_alpha * self.velocity + (1 - velocity_alpha) * observed_v

        self.mean = new_mean
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED

    def mark_lost(self):
        if self.state == TrackState.TRACKED:
            self.state = TrackState.LOST


class ByteTrack:
    """Two-stage association tracker with constant-velocity motion.

    Args:
        track_thresh: Score dividing the 'high' and 'low' detection pools. High-
            pool dets drive the first association; low-pool dets recover tracks
            in the second.
        unmatched_thresh: Min score for an unmatched high-pool detection to spawn
            a new track. Higher → fewer spurious IDs.
        detrack_thresh: Hard floor — detections below this are dropped entirely.
            Set below track_thresh to enable the second (low-pool) association.
        tracking_thresh: Max match cost allowed. Pairs above this cost are
            rejected, so Hungarian can't force a bad match. With iou_weight=1,
            cost = 1 - IoU, so tracking_thresh = 0.6 ≈ min IoU 0.4.
        track_buffer: Frames a LOST track is kept alive before removal.
        max_age: Absolute cap on total track age in frames (0 = disabled).
        min_hits: Min consecutive/total matched frames before a track is emitted.
        class_aware: If True, detections only match tracks of the same class.
        iou_weight: Cost mix — iou_weight*(1-IoU) + (1-iou_weight)*centroid.
        drag: Per-frame velocity damping while a track is LOST (0-1).
        velocity_alpha: EMA weight for the stored velocity on update.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        unmatched_thresh: float = 0.7,
        detrack_thresh: float = 0.1,
        tracking_thresh: float = 0.7,
        track_buffer: int = 30,
        max_age: int = 0,
        min_hits: int = 1,
        class_aware: bool = True,
        iou_weight: float = 0.75,
        drag: float = 0.85,
        velocity_alpha: float = 0.6,
    ):
        self.track_thresh = track_thresh
        self.unmatched_thresh = unmatched_thresh
        self.detrack_thresh = detrack_thresh
        self.tracking_thresh = tracking_thresh
        self.track_buffer = track_buffer
        self.max_age = max_age
        self.min_hits = min_hits
        self.class_aware = class_aware
        self.iou_weight = iou_weight
        self.drag = drag
        self.velocity_alpha = velocity_alpha

        self.tracks: List[_Track] = []
        self.frame_idx = 0
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self.frame_idx = 0
        self._next_id = 1

    def update(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
    ) -> List[Tuple[int, int, Tuple[float, float, float, float], float]]:
        """Run one frame of tracking.

        Args:
            detections: detections for the current frame (xyxy, absolute pixels).
            frame_shape: (height, width), used to normalize the centroid cost.

        Returns:
            list of (track_id, cls_id, (x1, y1, x2, y2), score) for tracks
            matched this frame with hits >= min_hits.
        """
        self.frame_idx += 1
        H, W = frame_shape
        diag = float(np.hypot(H, W))

        # ---- 0. Pre-filter detections and split into high/low pools ----
        dets = [d for d in detections if d.score >= self.detrack_thresh]
        high = [d for d in dets if d.score >= self.track_thresh]
        low = [d for d in dets if d.score < self.track_thresh]

        # ---- 1. Predict every non-removed track forward ----
        for t in self.tracks:
            t.predict()

        pool = [t for t in self.tracks if t.state != TrackState.REMOVED]

        # ---- 2. First association: all live tracks vs high-pool detections ----
        matches_hi, unmatched_tracks, unmatched_high = self._associate(
            pool, high, diag, gate=self.tracking_thresh
        )
        for t_i, d_i in matches_hi:
            pool[t_i].update(high[d_i].bbox, high[d_i].score, self.velocity_alpha)

        # ---- 3. Second association: still-TRACKED unmatched tracks vs low pool ----
        # Lost tracks intentionally skipped here — the paper shows that pairing
        # lost tracks with low-confidence dets hurts more than it helps.
        second_pool_local = [i for i in unmatched_tracks if pool[i].state == TrackState.TRACKED]
        second_tracks = [pool[i] for i in second_pool_local]

        matches_lo, _, _ = self._associate(second_tracks, low, diag, gate=self.tracking_thresh)
        matched_in_second = set()
        for local_i, d_i in matches_lo:
            global_i = second_pool_local[local_i]
            pool[global_i].update(low[d_i].bbox, low[d_i].score, self.velocity_alpha)
            matched_in_second.add(global_i)

        # ---- 4. Mark unmatched tracks as LOST ----
        for i in unmatched_tracks:
            if i in matched_in_second:
                continue
            pool[i].mark_lost()

        # ---- 5. Spawn new tracks from unmatched high-conf detections ----
        for d_i in unmatched_high:
            det = high[d_i]
            if det.score >= self.unmatched_thresh:
                self.tracks.append(_Track(self._next_id, det.bbox, det.score, det.cls_id))
                self._next_id += 1

        # ---- 6. Drop stale tracks ----
        kept: List[_Track] = []
        for t in self.tracks:
            if t.state == TrackState.REMOVED:
                continue
            if t.time_since_update > self.track_buffer:
                continue
            if self.max_age > 0 and t.age > self.max_age:
                continue
            kept.append(t)
        self.tracks = kept

        # ---- 7. Emit tracks matched this frame ----
        output = []
        for t in self.tracks:
            if t.state != TrackState.TRACKED or t.time_since_update != 0:
                continue
            if t.hits < self.min_hits:
                continue
            x1, y1, x2, y2 = _cxywh_to_xyxy(t.mean)
            x1 = float(np.clip(x1, 0, W - 1))
            y1 = float(np.clip(y1, 0, H - 1))
            x2 = float(np.clip(x2, 0, W - 1))
            y2 = float(np.clip(y2, 0, H - 1))
            output.append((t.track_id, t.cls_id, (x1, y1, x2, y2), float(t.score)))
        return output

    def _associate(
        self,
        tracks: List[_Track],
        dets: List[Detection],
        diag: float,
        gate: float,
    ):
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        track_boxes = np.stack([t.predicted_xyxy(self.drag) for t in tracks])
        det_boxes = np.array([d.bbox for d in dets], dtype=np.float64)

        iou = _pairwise_iou(track_boxes, det_boxes)
        cent = _pairwise_centroid_dist(track_boxes, det_boxes, diag)
        cost = self.iou_weight * (1 - iou) + (1 - self.iou_weight) * cent

        if self.class_aware:
            track_cls = np.array([t.cls_id for t in tracks])
            det_cls = np.array([d.cls_id for d in dets])
            cost = np.where(track_cls[:, None] != det_cls[None, :], 1e6, cost)

        # Hard gate: no pair above the threshold cost is ever accepted.
        cost = np.where(cost > gate, 1e6, cost)

        row_ind, col_ind = linear_sum_assignment(cost)

        matches: List[Tuple[int, int]] = []
        matched_rows = set()
        matched_cols = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e6:
                continue
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))

        unmatched_t = [i for i in range(len(tracks)) if i not in matched_rows]
        unmatched_d = [i for i in range(len(dets)) if i not in matched_cols]
        return matches, unmatched_t, unmatched_d
