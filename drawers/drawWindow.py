import cv2
import numpy as np
from .drawPoint import PointDrawer
from utils import Player, Ball, bgr_to_hex

class DrawWindow:
    
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.point_drawer = PointDrawer(point_color="#FF0000", point_radius=7)
        self.picture_in_picture_section = None
        self.other_point = None
        self.scaleBig = 0.25
        self.scaleSmall = 1.0
        
        self.jersey_colors_cache = {}                      
            
    def composeFrame(self, big, small, pos=(0,0), scale=0.25) -> np.ndarray:
        """
        picture-in-picture composition of two frames
        Args:
            big: BGR image (np.ndarray)
            small: BGR image (np.ndarray)
            pos: tuple (x,y) position of small frame in big frame
            scale: float, scaling factor for small frame
        Returns:
            composed frame
        """
        
        # getting dimension of big frame
        hsmall, wsmall = small.shape[:2]
        self.scaleBig = scale
        
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
    
    def drawPointsOnFrame(self, frame, points: np.ndarray = None) -> np.ndarray:
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

    def drawBoxOnFrame(self, frame, box: np.ndarray, color: str = "#00FF00", thickness: int = 2) -> np.ndarray:
        """
        Draw a bounding box on the frame.

        Args:
            frame: BGR image (np.ndarray)
            box: np.ndarray, shape (4,), float [x1,y1,x2,y2]
            color: str, hex color code
            thickness: int, thickness of the box lines

        Returns:
            annotated frame
        """
        annotated = frame.copy()
        
        # colore in BGR
        color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        color_bgr = color_bgr[::-1]  # RGB -> BGR

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)
        
        return annotated

    def realtimeDisplaying(self, frame: np.ndarray, frame_idx: int = None) -> None:
        """
        Display the frame in a window and allow user to click on it to get coordinates.

        Args:
            frame: BGR image (np.ndarray)
        """
        while True:
            display = frame.copy()
            #display = cv2.resize(display, (width, height))

            cv2.imshow(str(frame_idx) if frame_idx is not None else self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyWindow(self.window_name)
    
    def draw_player_ellipse(frame: np.ndarray, player: Player, color: tuple[int, int, int], is_possessor: bool):
        x1, y1, x2, y2 = player.as_int_tuple()
        number = player.track_id if player.track_id is not None else 0
        cx, cy = int(player.center[0]), int(player.center[1])
        foot_x, foot_y = int(player.foot[0]), int(player.foot[1])

        w = max(12, x2 - x1)
        ax1 = int(w * 0.55)
        ax2 = int(w * 0.22)

        # black outline to keep white jerseys visible
        cv2.ellipse(frame, (cx, foot_y), (ax1, ax2), 0, 0, 360, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, foot_y), (ax1, ax2), 0, 0, 360, color, 3, cv2.LINE_AA)

        if is_possessor:
            cv2.ellipse(frame, (cx, foot_y), (ax1 + 10, ax2 + 8), 0, 0, 360, (0, 255, 255), 4, cv2.LINE_AA)

        label = str(number)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx = cx - tw // 2
        ly = max(th + 10, y1 - 10)
        
        cv2.rectangle(frame, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), (0,0,0), -1)
        cv2.rectangle(frame, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 5), color, 1)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    
    def drawAllFrames(
            self, frames: list[np.ndarray], 
            small: np.ndarray,
            point_per_small: list[np.ndarray], 
            points_per_frame: list[np.ndarray], 
            players_per_frame: list[list[Player]] = None,
            tactical_players_per_frame: list[dict[int, list[float, float]]] = None,
            ball_per_frame: list[Ball] = None,
        ) -> list[np.ndarray]:
        """
        Draw all frames with points and boxes.

        Args:
            frames: list of BGR images (np.ndarray)
            small: BGR image (np.ndarray) for picture-in-picture
            point_per_small: np.ndarray, shape (N,2), float32
            points_per_frame: list of np.ndarray, each of shape (N,2), float32
            players_per_frame: list of list of Player  per frame
            tactical_players_per_frame: list of dict of player_id to (x,y) coordinates
            ball_per_frame: objects of Ball per frame
        Returns:
            list of annotated frames
        """
        # container for video output
        frames_out = []
        
        # draw small first
        frameImg = cv2.imread("images/basketball_court.png")
        frameImg = self.drawPointsOnFrame(frameImg, point_per_small)
        
        frame = self.composeFrame(frames[0].copy(), frameImg, pos=(10,10), scale=0.3)
        
        # TODO: to be removed from here, the logic of tracking players feature in the drawing class is soooooo wrong
        # playerNumber is the same of track_id so why recalculating it again?
        
        
        for frame_idx, frame in enumerate(frames):
            # taking frame from video and draw points took from yolo model
            
            if frame_idx % 50 == 0: print(f"Rendering {frame_idx}/{len(frames)}")
            
            frameTactical = frameImg.copy()
            frameSpec = frame.copy()
            frameSpec = self.drawPointsOnFrame(frameSpec, points_per_frame[frame_idx])
            
            
            # can be divided in more functions to have a better readability
            if players_per_frame[frame_idx] is not None:
                for player in players_per_frame[frame_idx]:
                    
                    if player.track_id not in self.jersey_colors_cache:
                        self.jersey_colors_cache[player.track_id] = player.get_dominant_jersey_color(frame)
                    
                    color = self.jersey_colors_cache[player.track_id]
                    #print(color)
                    is_possessor = (player.class_id == 99)
                    
                    DrawWindow.draw_player_ellipse(frameSpec, player, color, is_possessor=is_possessor)
                    
                    tactical_data = tactical_players_per_frame[frame_idx]
                        
                    if player.track_id in tactical_data:
                            coord = tactical_data[player.track_id] # Recupera (x, y) specifico per questo player
                            # Disegna il punto sul frame tattico usando lo stesso colore della maglia
                            frameTactical = self.point_drawer.drawSpecifiedPoint(coord[0], coord[1], frameTactical, color=color)
                    
                    '''
                    #print("Drawing player box with class_id:", player)
                    if player.class_id == 99:
                        print("Drawing possessor box:", player)
                    if player.class_id == 99:  # Possessor
                        frameSpec = self.drawBoxOnFrame(frameSpec, player.xyxy, color="#FF0000", thickness=3)
                    else:
                        if player.track_id not in self.jersey_colors_cache:
                            jersery_colors[player.track_id] = player.get_dominant_jersey_color(frameSpec)
                        #frameSpec = self.drawBoxOnFrame(frameSpec, player.xyxy, color=jersery_colors[player.track_id], thickness=2)
                        DrawWindow.draw_player_ellipse(frameSpec, player, jersery_colors[player.track_id], is_possessor=is_possessor)
                    '''
                #for k, tactical_player_coord in tactical_players_per_frame[frame_idx].items():
                #    frameTactical = self.point_drawer.drawSpecifiedPoint(tactical_player_coord[0], tactical_player_coord[1], frameTactical, color=color)

                if ball_per_frame is not None and ball_per_frame[frame_idx] is not None:
                    ball = ball_per_frame[frame_idx]
                    frameSpec = self.drawBoxOnFrame(frameSpec, ball.xyxy, color="#0000FF", thickness=2)
                    
            
            # compose with picture-in-picture
            frame = self.composeFrame(frameSpec, frameTactical, pos=(10,10), scale=0.3)
            # self.realtimeDisplaying(frame, frame_idx)
            frames_out.append(frame)
        return frames_out
        
        