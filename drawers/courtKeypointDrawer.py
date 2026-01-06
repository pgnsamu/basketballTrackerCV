import supervision as sv
import numpy as np
import cv2

class CourtKeypointDrawer:
    """
    A drawer class responsible for drawing court keypoints on a sequence of frames.

    Attributes:
        keypoint_color (str): Hex color value for the keypoints.
    """
    def __init__(self):
        self.keypoint_color = '#ff2c2c'

    def draw(self, frames, court_keypoints):
        """
        Draws court keypoints on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            court_keypoints (list): A corresponding list of lists where each sub-list contains
                the (x, y) coordinates of court keypoints for that frame.

        Returns:
            list: A list of frames with keypoints drawn on them.
        """
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8)
        
        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )
        
        output_frames = []
        for index,frame in enumerate(frames):
            annotated_frame = frame.copy()

            keypoints = court_keypoints[index]
            # Draw dots
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints)
            # Draw labels
            # Convert PyTorch tensor to numpy array
            keypoints_numpy = keypoints.cpu().numpy()
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_numpy)

            output_frames.append(annotated_frame)

        return output_frames
    
    def draw(self, frames: list[np.ndarray], court_keypoints: list[np.ndarray], cv) -> list[np.ndarray]:
        """
        Draw court keypoints on frames using OpenCV.

        Args:
            frames: list of BGR images (np.ndarray)
            court_keypoints: list of np.ndarray, each shape (18,2), float32
                            (0,0) means not detected

        Returns:
            list of annotated frames
        """

        output_frames = []

        # colore in BGR
        color = tuple(int(self.keypoint_color[i:i+2], 16) for i in (1, 3, 5))
        color = color[::-1]  # RGB -> BGR

        for idx, frame in enumerate(frames):
            annotated = frame.copy()
            kps = court_keypoints[idx]

            if kps is None:
                output_frames.append(annotated)
                continue

            kps = np.asarray(kps, dtype=np.float32)

            for kp_idx, (x, y) in enumerate(kps):
                if x <= 0 or y <= 0:
                    continue

                x_i, y_i = int(round(x)), int(round(y))

                # disegna punto
                cv2.circle(
                    annotated,
                    center=(x_i, y_i),
                    radius=6,
                    color=color,
                    thickness=-1
                )

                # disegna label
                cv2.putText(
                    annotated,
                    text=str(kp_idx),
                    org=(x_i + 5, y_i - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

            output_frames.append(annotated)

        return output_frames