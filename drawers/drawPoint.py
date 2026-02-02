import cv2
import numpy as np

class PointDrawer:
    def __init__(self, point_color: str = "#FF0000", point_radius: int = 5):
        self.point_color = point_color
        self.point_radius = point_radius

    def _parse_color(self, color_input):
        """
        Convert color from hex string or tuple to BGR tuple.
        
        Args:
            color_input: Either hex string (e.g., "#FF0000") or BGR tuple
            
        Returns:
            BGR tuple (B, G, R)
        """
        if isinstance(color_input, tuple):
            return color_input
        # Assume it's a hex string
        color_bgr = tuple(int(color_input[i:i+2], 16) for i in (1, 3, 5))
        return color_bgr[::-1]  # RGB -> BGR

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
        color_bgr = self._parse_color(oneTimeColor)

        annotated = frame.copy()

        if points is None:
            return annotated

        for i, (x, y) in enumerate(points):
            if x < 0 or y < 0:
                continue

            x_i, y_i = int(round(x)), int(round(y))

            # disegna punto
            cv2.circle(
                annotated,
                (x_i, y_i),
                self.point_radius,
                color_bgr,
                -1
            )
            cv2.putText(
                annotated,
                f"{i}",
                (x_i + 5, y_i - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_bgr,
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
            color: hex color string (e.g., "#FF0000") or BGR tuple

        Returns:
            annotated frame
        """
        oneTimeColor = self.point_color        
        if color is not None:
            oneTimeColor = color

        # colore in BGR
        color_bgr = self._parse_color(oneTimeColor)

        if x <= 0 or y <= 0:
            return frame

        x_i, y_i = int(round(x)), int(round(y))

        # disegna punto
        cv2.circle(
            frame,
            (x_i, y_i),
            self.point_radius,
            color_bgr,
            -1
        )

        return frame