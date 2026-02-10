from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
try:
    import supervision as sv
except ModuleNotFoundError:  # optional dependency
    sv = None
import numpy as np
import cv2


@dataclass(frozen=False, slots=True)
class DetectedObject:
    xyxy: np.ndarray      # (4,) float [x1,y1,x2,y2]
    conf: float
    class_id: int #TODO: maybe an enum is better i need to know if there is a class_id for the possessor

    @property
    def x1(self): return float(self.xyxy[0])
    @property
    def y1(self): return float(self.xyxy[1])
    @property
    def x2(self): return float(self.xyxy[2])
    @property
    def y2(self): return float(self.xyxy[3])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        # punto "a terra" (utile nei campi)
        return ((self.x1 + self.x2) / 2, self.y2)
    
    def as_int_tuple(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = np.rint(self.xyxy).astype(int)
        return (int(x1), int(y1), int(x2), int(y2))

class Ball(DetectedObject):
    pass

@dataclass(frozen=False, slots=True)
class Player(DetectedObject):
    track_id: Optional[int]      # ID persistente (ByteTrack/SORT). None se non assegnato
    
    def get_dominant_jersey_color(self, frame_bgr) -> tuple[int, int, int]:
        """
        Estimate the dominant jersey color inside the *torso* region of the player.
        This version is more robust than a full-bbox dominant color because it:
          - uses an upper-body ROI (so it avoids floor/shorts)
          - filters low-saturation / low-value pixels in HSV (so it avoids shadows)
          - uses a median color (more stable under lighting changes)

        Returns:
            tuple: Dominant color in BGR format (B, G, R)
        """
        x1, y1, x2, y2 = self.as_int_tuple()
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # anti-crash: bbox too small
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return (0, 0, 0)

        bw = x2 - x1
        bh = y2 - y1

        # --- Upper-body ROI (torso) ---
        # Goal: avoid face/legs/floor as much as possible.
        # y: ~18% -> ~60% of bbox height
        ry1 = int(y1 + bh * 0.18)
        ry2 = int(y1 + bh * 0.60)

        # x: crop inside to reduce arms/background bleed
        pad = int(bw * 0.15)
        rx1 = x1 + pad
        rx2 = x2 - pad

        ry1 = max(y1, min(y2 - 1, ry1))
        ry2 = max(y1 + 1, min(y2, ry2))
        rx1 = max(x1, min(x2 - 1, rx1))
        rx2 = max(x1 + 1, min(x2, rx2))

        roi = frame_bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return (0, 0, 0)

        # --- HSV filtering ---
        # We want jersey pixels, not skin/floor.
        # 1) Require some saturation/value to avoid shadows/gray background.
        # 2) Remove typical skin-tone hues (which often dominate bbox).
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # Keep colorful enough pixels (good for colored jerseys)
        mask_color = (s_chan >= 55) & (v_chan >= 45)

        # Also keep very bright / low-saturation pixels (good for WHITE / light jerseys)
        # White tends to have low S but high V.
        mask_white = (s_chan <= 45) & (v_chan >= 160)

        # Skin mask (approx) in OpenCV HSV:
        # - Light/medium skin often sits around H ~ [0..25]
        # - Under some lighting it can wrap close to 180, so also remove [160..180]
        skin1 = (h_chan >= 0) & (h_chan <= 25)
        skin2 = (h_chan >= 160) & (h_chan <= 180)
        skin = (skin1 | skin2) & (s_chan >= 20) & (s_chan <= 200) & (v_chan >= 60)

        mask = (mask_color | mask_white) & (~skin)
        pixels = roi[mask]

        # Fallbacks:
        # - If the strict mask removes too much, relax saturation a bit (still removing skin)
        if pixels.shape[0] < 80:
            mask_relaxed_color = (s_chan >= 35) & (v_chan >= 35) & (~skin)
            mask_relaxed_white = (s_chan <= 60) & (v_chan >= 140) & (~skin)
            pixels = roi[(mask_relaxed_color | mask_relaxed_white)]

        # - Final fallback: use full ROI
        if pixels.shape[0] < 60:
            pixels = roi.reshape(-1, 3)

        # sample for speed
        if pixels.shape[0] > 2500:
            idx = np.random.choice(pixels.shape[0], 2500, replace=False)
            pixels = pixels[idx]

        med = np.median(pixels.astype(np.int32), axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))

    def get_dominant_jersey_color_roi(self, frame_bgr) -> tuple[int, int, int]:
        """
        Optional ROI-based jersey color sampler.
        For compatibility with DrawWindow; currently aliases get_dominant_jersey_color().
        """
        return self.get_dominant_jersey_color(frame_bgr)


def detections_to_players(dets: sv.Detections) -> list[Player]:
    if sv is None:
        raise ModuleNotFoundError("The 'supervision' package is required for this conversion function. Install it (pip install supervision) or avoid calling this helper.")

    """
    Convert a supervision Detections object to a list of Player instances.
    Args:
        dets (sv.Detections): Supervision Detections object containing detection data.
    Returns:
        list: List of Player instances.
    """
    if dets is None or len(dets) == 0:
        return []

    out: list[Player] = []
    tids = dets.tracker_id if dets.tracker_id is not None else [None] * len(dets)

    for xyxy, conf, cid, tid in zip(dets.xyxy, dets.confidence, dets.class_id, tids):
        out.append(
            Player(
                track_id=None if tid is None else int(tid),
                xyxy=xyxy.copy(),
                conf=float(conf),
                class_id=int(cid),
            )
        )
    return out


def players_to_detections(players: list[Player]) -> sv.Detections:
    if sv is None:
        raise ModuleNotFoundError("The 'supervision' package is required for this conversion function. Install it (pip install supervision) or avoid calling this helper.")

    """
    Convert a list of Player instances to a supervision Detections object.
    Args:
        players (list): List of Player instances.
    Returns:
        sv.Detections: Supervision Detections object containing the players' data.
    """
    if not players:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int64),
            tracker_id=None,
        )

    xyxy = np.stack([p.xyxy for p in players]).astype(np.float32)          # (N,4)
    confidence = np.array([p.conf for p in players], dtype=np.float32)     # (N,)
    class_id = np.array([p.class_id for p in players], dtype=np.int64)     # (N,)

    tids = [p.track_id for p in players]
    tracker_id = None if all(t is None for t in tids) else np.array(
        [-1 if t is None else int(t) for t in tids], dtype=np.int64
    )

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )
    
def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    """
    Convert BGR color tuple to hex string.
    Args:
        bgr: Color in BGR format (B, G, R)
    Returns:
        str: Hex color string in format '#RRGGBB'
    """
    b, g, r = bgr
    return f'#{r:02x}{g:02x}{b:02x}'