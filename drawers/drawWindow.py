import cv2
import numpy as np
from .drawPoint import PointDrawer
from homography.homography import Homography

class DrawWindow:
    def __init__(self, window_name: str, homography: Homography = None):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.clicked_point = (0, 0)
        self.point_drawer = PointDrawer(point_color="#FF0000", point_radius=7)
        self.homography = homography
        self.picture_in_picture_section = None
        self.other_point = None
        self.scaleBig = 0.25
        self.scaleSmall = 1.0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            if self.picture_in_picture_section is not None and self.homography is not None:
                y1, y2, x1, x2 = self.picture_in_picture_section
                width = (x2 - x1)/self.scaleSmall
                print("width PIP:", width)
                height = (y2 - y1)/self.scaleSmall
                print("height PIP:", height)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    
                    xFinal, yFinal = width * (x - x1) / (x2-x1), height * (y - y1) / (y2-y1)
                    point = np.array([[xFinal, yFinal]], dtype='float32')
                    
                    print("Clicked inside PIP at:", (x, y))
                    print("Clicked inside PIP at:", (xFinal, yFinal))
                    self.other_point = self.homography.transform_points(point, inverse=False)[0]
                else:
                    point = np.array([[[x, y]]], dtype=np.float32)
                    tactical_pt = self.homography.transform_points(point.reshape(1, 2), inverse=True)[0]  # (tx,ty)

                    tx, ty = float(tactical_pt[0]), float(tactical_pt[1])

                    # TACTICAL original -> coordinate DENTRO PiP (quindi sul big frame)
                    pip_x = x1 + tx * (width*self.scaleSmall / width)
                    pip_y = y1 + ty * (height*self.scaleSmall / height)

                    self.other_point = (pip_x, pip_y)
                      
            
    def composeFrame(self, big, small, pos=(0,0), scale=0.25):
        """
        picture-in-picture composition of two frames
        """
        # self.scale = scale
        
        # getting dimension of big frame
        h, w = big.shape[:2]
        hsmall, wsmall = small.shape[:2]
        self.scaleBig = scale
        
        # resizing small frame
        #sh, sw = int(h*scale), int(w*scale)
        sh, sw = 161, 300
        small = cv2.resize(small, (sw, sh))
        self.scaleSmall = sw / wsmall
        
        # setting position
        x, y = pos
        
        # ritaglio un rettangolo e lo sostituisco con la small
        big[y:y+sh, x:x+sw] = small
        self.picture_in_picture_section = (y,y+sh,x,x+sw)
        return big
    
    def drawOnFrame(self, frame, points: np.ndarray = None):
        """
        Draw points on the frame.

        Args:
            frame: BGR image (np.ndarray)
            points: np.ndarray, shape (N,2), float32
                    (0,0) means not detected

        Returns:
            annotated frame
        """
        annotated = frame.copy()
        
        print(points)
        
        # Draw other points
        if points is not None:
            annotated = self.point_drawer.drawPoints(annotated, points)

        return annotated

    def realtimeDisplaying(self, frame):
        """
        Display the frame in a window and allow user to click on it to get coordinates.

        Args:
            frame: BGR image (np.ndarray)
        """
        while True:
            display = frame.copy()
            #display = cv2.resize(display, (width, height))

            # Draw the clicked point
            display = self.point_drawer.drawSpecifiedPoint(self.clicked_point[0], self.clicked_point[1], display, color="#00FF00")
            if self.other_point is not None:
                display = self.point_drawer.drawSpecifiedPoint(self.other_point[0], self.other_point[1], display, color="#FF0095")
            
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyWindow(self.window_name)
        