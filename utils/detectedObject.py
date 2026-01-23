from dataclasses import dataclass
from typing import Optional
import supervision as sv
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
        Estimate the dominant jersey color within the bounding box of the detected object.
        Args:
            frame_bgr (np.ndarray): The BGR image frame from which to extract the color
        Returns:
            tuple: Dominant color in BGR format (B, G, R)
        """
        x1, y1, x2, y2 = self.as_int_tuple()
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return (0, 200, 255)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return (0, 200, 255)

        crop = crop[: max(1, crop.shape[0] // 2), :]
        crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
        pixels = crop.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=2)
        bgr = centers[int(np.argmax(counts))].astype(int)
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    

def detections_to_players(dets: sv.Detections) -> list[Player]:
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