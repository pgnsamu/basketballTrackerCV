import cv2
import numpy as np

class PointDrawer:
    def __init__(self, point_color: str = "#FF0000", point_radius: int = 5):
        self.point_color = point_color
        self.point_radius = point_radius

    def drawPoints(self, frame, points: np.ndarray, color: str = None):
        """
        Draw points on frames using OpenCV.

        Args:
            frame: BGR image (np.ndarray)
            points: np.ndarray, shape (N,2), float32
                    (0,0) means not detected

        Returns:
            annotated frame
        """
        
        
        oneTimeColor = self.point_color        
        if color is not None:
            oneTimeColor = color

        # colore in BGR
        color = tuple(int(oneTimeColor[i:i+2], 16) for i in (1, 3, 5))
        color = color[::-1]  # RGB -> BGR

        annotated = frame.copy()

        if points is None:
            output_frames.append(annotated)
            return

        points = np.asarray(points, dtype=np.float32)

        for i, (x, y) in enumerate(points):
            if x < 0 or y < 0:
                continue

            x_i, y_i = int(round(x)), int(round(y))

            # disegna punto
            cv2.circle(
                annotated,
                (x_i, y_i),
                self.point_radius,
                color,
                -1
            )
            cv2.putText(
                annotated,
                f"{i}",
                (x_i + 5, y_i - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )

        return annotated
    
    def drawSpecifiedPoint(self, x: float, y: float, frame, color: str = None):
        """
        Draw a specified point on the frame using OpenCV.

        Args:
            x: x-coordinate of the point
            y: y-coordinate of the point
            frame: BGR image (np.ndarray)

        Returns:
            annotated frame
        """
        
        oneTimeColor = self.point_color        
        if color is not None:
            oneTimeColor = color

        # colore in BGR
        color = tuple(int(oneTimeColor[i:i+2], 16) for i in (1, 3, 5))
        color = color[::-1]  # RGB -> BGR

        #annotated = frame.copy()

        if x <= 0 or y <= 0:
            return frame

        x_i, y_i = int(round(x)), int(round(y))

        # disegna punto
        cv2.circle(
            frame,
            (x_i, y_i),
            self.point_radius,
            color,
            -1
        )

        return frame