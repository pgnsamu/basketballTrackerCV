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
    
    def drawSpecifiedPoint(self, x: float, y: float, frame, color: str = None, label: str = None):
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
        if label is not None:
            text = str(label)
            
            # --- IMPOSTAZIONI FONT MIGLIORATO ---
            # DUPLEX è più "pieno" e leggibile del SIMPLEX
            font_face = cv2.FONT_HERSHEY_DUPLEX 
            font_scale = 0.5  # Piccolo ma leggibile grazie al font DUPLEX
            thickness = 1
            
            # Calcola dimensioni testo
            (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
            
            # Coordinate per centrare il testo sopra il punto
            # Offset verticale: raggio + 4 pixel
            text_x = x_i - (text_w // 2)
            text_y = y_i - self.point_radius - 4 

            # --- SFONDO AD ALTO CONTRASTO (Opzionale ma consigliato) ---
            # Disegna un rettangolino nero (o grigio scuro) dietro il numero
            box_coords_1 = (text_x - 2, text_y - text_h - 2)
            box_coords_2 = (text_x + text_w + 2, text_y + 2)
            cv2.rectangle(frame, box_coords_1, box_coords_2, (0, 0, 0), -1)
            
            # --- TESTO BIANCO PURO ---
            # Il bianco su sfondo nero è il massimo contrasto possibile
            cv2.putText(frame, text, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return frame