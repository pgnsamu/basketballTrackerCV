import cv2
import numpy as np
from .drawPoint import PointDrawer
from homography.homography import Homography

class DrawWindow:
    
    homography: Homography = None
    
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
        
    def setHomography(self, homography: Homography):
        self.homography = homography

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # store clicked point
            self.clicked_point = (x, y)
            
            if self.picture_in_picture_section is not None and self.homography is not None:
                # frame coordinates of PiP section
                minY, maxY, minX, maxX = self.picture_in_picture_section
                
                # width and height of original photo before resizing to picture-in-picture size
                width = (maxX - minX)/self.scaleSmall
                height = (maxY - minY)/self.scaleSmall
                
                # check if clicked point is inside PiP
                if minX <= x <= maxX and minY <= y <= maxY:
                    # conversion to original photo coordinates as it was full-screen
                    xFinal, yFinal = width * (x - minX) / (maxX - minX), height * (y - minY) / (maxY - minY)
                    point = np.array([[xFinal, yFinal]], dtype='float32')
                    
                    self.other_point = self.homography.transform_points(point, inverse=False)[0]
                else:
                    point = np.array([[[x, y]]], dtype=np.float32)
                    tactical_pt = self.homography.transform_points(point.reshape(1, 2), inverse=True)[0]  # (tx,ty)
                    # convert to original tactical image coordinates
                    tacticalPointFoundX, tacticalPointFoundY = float(tactical_pt[0]), float(tactical_pt[1])

                    # resizing back to picture-in-picture scale
                    pip_x = minX + tacticalPointFoundX * self.scaleSmall
                    pip_y = minY + tacticalPointFoundY * self.scaleSmall

                    self.other_point = (pip_x, pip_y)
                      
            
    def composeFrame(self, big, small, pos=(0,0), scale=0.25):
        """
        picture-in-picture composition of two frames
        """
        
        # getting dimension of big frame
        hsmall, wsmall = small.shape[:2]
        self.scaleBig = scale
        
        # resizing small frame
        # removed because using this method the scale was different in width and height 
        #h, w = big.shape[:2]
        #sh, sw = int(h*scale), int(w*scale)
        
        sh, sw = 161, 300
        small = cv2.resize(small, (sw, sh))
        self.scaleSmall = sw / wsmall
        
        # setting position
        x, y = pos
        
        # ritaglio un rettangolo e lo sostituisco con la small
        big[y:y+sh, x:x+sw] = small
        # coordinates of picture-in-picture section in big frame
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
            
            # Draw the other point on the other image
            if self.other_point is not None:
                display = self.point_drawer.drawSpecifiedPoint(self.other_point[0], self.other_point[1], display, color="#FF0095")
            
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyWindow(self.window_name)
        
    def drawAllFrames(self, frames: list[np.ndarray], small, point_per_small: list[np.ndarray], points_per_frame: list[np.ndarray], homography: Homography = None ) -> list[np.ndarray]:
        """
        Draw all frames with points transformed by the homography.

        Args:
            frames: list of BGR images (np.ndarray)
            small: BGR image (np.ndarray) for picture-in-picture
            point_per_small: np.ndarray, shape (N,2), float32
            points_per_frame: list of np.ndarray, each of shape (N,2), float32
            homography: Homography object (optional)
        """
        # container for video output
        frames_out = []
        
        # draw small once
        frameImg = cv2.imread("images/basketball_court.png")
        frameImg = self.drawOnFrame(frameImg, point_per_small)
        
        for frame_idx, frame in enumerate(frames):
            # make homography for each frame
            homography = Homography(
                source_points=point_per_small,
                destination_points=points_per_frame[frame_idx]
            )
            self.setHomography(homography)
            # taking frame from video and draw points took from yolo model
            frameSpec = frame.copy()
            frameSpec = self.drawOnFrame(frameSpec, points_per_frame[frame_idx])
            
            # compose with picture-in-picture
            frame = self.composeFrame(frameSpec, frameImg, pos=(10,10), scale=0.3)
            
            frames_out.append(frame)
        return frames_out
        
        