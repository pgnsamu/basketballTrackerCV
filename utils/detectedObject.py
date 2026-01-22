from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
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
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        # punto "a terra" (utile nei campi da calcio)
        return ((self.x1 + self.x2) / 2, self.y2)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = np.rint(self.xyxy).astype(int)
        return (int(x1), int(y1), int(x2), int(y2))
