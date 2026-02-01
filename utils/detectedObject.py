from dataclasses import dataclass
from typing import Optional
import supervision as sv
import numpy as np
import cv2
from .bbox_utils import get_foot_position # Usiamo la nuova logica centralizzata

@dataclass(frozen=False, slots=True)
class DetectedObject:
    xyxy: np.ndarray      # (4,) float [x1,y1,x2,y2]
    conf: float
    class_id: int

    @property
    def x1(self): return float(self.xyxy[0])
    @property
    def y1(self): return float(self.xyxy[1])
    @property
    def x2(self): return float(self.xyxy[2])
    @property
    def y2(self): return float(self.xyxy[3])

    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        # Usa la funzione in bbox_utils per coerenza
        return get_foot_position(self.xyxy)
    
    def as_int_tuple(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = np.rint(self.xyxy).astype(int)
        return (int(x1), int(y1), int(x2), int(y2))

class Ball(DetectedObject):
    pass

@dataclass(frozen=False, slots=True)
class Player(DetectedObject):
    track_id: Optional[int]      

    def get_dominant_jersey_color(self, frame_bgr) -> tuple[int, int, int]:
        # Logica colore invariata per ora (funziona, anche se può essere lenta)
        x1, y1, x2, y2 = self.as_int_tuple()
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        # Validazione anti-crash: se il box è troppo piccolo o nullo
        if x2 <= x1 + 2 or y2 <= y1 + 2: 
            return (0, 0, 0) # Ritorna nero se invalido

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return (0, 0, 0)

        # Sampling più aggressivo per velocità (solo centro maglia)
        crop = crop[: max(1, int(crop.shape[0] * 0.6)), :] # Prendiamo il 60% superiore (torso)
        crop = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_NEAREST)
        pixels = crop.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten(), minlength=2)
            bgr = centers[int(np.argmax(counts))].astype(int)
            return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        except Exception:
            return (0,0,0) # Fallback

# Le funzioni di conversione detections_to_players e viceversa rimangono identiche,
# basta assicurarsi che importino Player da qui.
def detections_to_players(dets: sv.Detections) -> list[Player]:
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
    # (Copia pure la tua versione precedente qui, era corretta)
    if not players:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int64),
            tracker_id=None,
        )

    xyxy = np.stack([p.xyxy for p in players]).astype(np.float32)
    confidence = np.array([p.conf for p in players], dtype=np.float32)
    class_id = np.array([p.class_id for p in players], dtype=np.int64)
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
    b, g, r = bgr
    return f'#{r:02x}{g:02x}{b:02x}'