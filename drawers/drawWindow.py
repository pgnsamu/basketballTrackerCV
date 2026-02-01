import cv2
import numpy as np
from .drawPoint import PointDrawer

class DrawWindow:
    
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.point_drawer = PointDrawer(point_color="#FF0000", point_radius=7)
        self.jersey_colors_cache = {} 

        # --- CONFIGURAZIONE VISIVA NUOVA ---
        self.MINIMAP_WIDTH = 400  # RIDOTTO (era 600)
        self.MINIMAP_HEIGHT = 215 # Proporzionale (~321 * 400/598)
        
    def composeFrame(self, big, small, pos=(0,0)) -> np.ndarray:
        """Sovrappone la minimappa rendendola quasi opaca."""
        # 1. Resize
        if small.shape[1] != self.MINIMAP_WIDTH:
            small = cv2.resize(small, (self.MINIMAP_WIDTH, self.MINIMAP_HEIGHT))
        
        x_offset, y_offset = pos
        y1, y2 = y_offset, y_offset + small.shape[0]
        x1, x2 = x_offset, x_offset + small.shape[1]

        if y2 > big.shape[0] or x2 > big.shape[1]: return big 

        roi = big[y1:y2, x1:x2]
        
        # 2. Bordo nero alla minimappa
        cv2.rectangle(small, (0,0), (small.shape[1]-1, small.shape[0]-1), (0,0,0), 4)

        # 3. Blending "Solido" (Opaco)
        # alpha=0.05 (pochissimo video sotto), beta=0.95 (tutta mappa)
        # Questo toglie l'effetto "fantasma" trasparente
        blended = cv2.addWeighted(roi, 0.05, small, 0.95, 0)
        
        big[y1:y2, x1:x2] = blended
        return big
    
    def drawPointsOnFrame(self, frame, points: np.ndarray = None) -> np.ndarray:
        if points is not None:
            return self.point_drawer.drawPoints(frame.copy(), points)
        return frame.copy()

    @staticmethod
    def draw_player_ellipse(frame: np.ndarray, player, color: tuple, is_possessor: bool):
        """Disegna ellisse e numero sul campo (invariato ma pulito)"""
        x1, y1, x2, y2 = player.as_int_tuple()
        number = player.track_id if player.track_id is not None else 0
        cx, cy = int(player.center[0]), int(player.center[1])
        foot_x, foot_y = int(player.foot[0]), int(player.foot[1])

        w = max(12, x2 - x1)
        ax1 = int(w * 0.55)
        ax2 = int(w * 0.22)

        cv2.ellipse(frame, (cx, foot_y), (ax1, ax2), 0, 0, 360, (0, 0, 0), 6, cv2.LINE_AA) 
        cv2.ellipse(frame, (cx, foot_y), (ax1, ax2), 0, 0, 360, color, 3, cv2.LINE_AA)     

        if is_possessor:
            cv2.ellipse(frame, (cx, foot_y), (ax1 + 10, ax2 + 8), 0, 0, 360, (0, 255, 255), 4, cv2.LINE_AA)

        label = str(number)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx = cx - tw // 2
        ly = max(th + 10, y1 - 15)
        
        cv2.rectangle(frame, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), (0,0,0), -1)
        cv2.rectangle(frame, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), color, 1)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def drawAllFrames(self, frames, small, point_per_small, points_per_frame, 
                      players_per_frame=None, tactical_players_per_frame=None, 
                      ball_per_frame=None, tactical_ball_per_frame=None):
        
        frames_out = []
        base_court = cv2.imread("images/basketball_court.png")
        if base_court is None: base_court = np.zeros((321, 598, 3), dtype=np.uint8)
        
        print(f"Rendering finale {len(frames)} frames...")
        
        for frame_idx, frame in enumerate(frames):
            if frame_idx % 50 == 0: print(f"Rendering {frame_idx}/{len(frames)}")

            frameSpec = frame.copy()
            current_tactical = base_court.copy()

            # --- GIOCATORI ---
            if players_per_frame and players_per_frame[frame_idx]:
                for player in players_per_frame[frame_idx]:
                    if player.track_id not in self.jersey_colors_cache:
                        self.jersey_colors_cache[player.track_id] = player.get_dominant_jersey_color(frame)
                    
                    color = self.jersey_colors_cache[player.track_id]
                    is_possessor = (player.class_id == 99)
                    
                    # Video
                    self.draw_player_ellipse(frameSpec, player, color, is_possessor)

                    # Minimappa
                    if tactical_players_per_frame and frame_idx < len(tactical_players_per_frame):
                        t_players = tactical_players_per_frame[frame_idx]
                        if player.track_id in t_players:
                            coord = t_players[player.track_id]
                            tx, ty = int(coord[0]), int(coord[1])
                            
                            # Pallino Giocatore
                            cv2.circle(current_tactical, (tx, ty), 10, (0,0,0), -1)
                            cv2.circle(current_tactical, (tx, ty), 8, color, -1)
                            
                            # Numero
                            pid_str = str(player.track_id)
                            (tw, th), _ = cv2.getTextSize(pid_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.putText(current_tactical, pid_str, (tx - tw//2, ty + th//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

            # --- PALLA (VIDEO PRINCIPALE) ---
            if ball_per_frame and ball_per_frame[frame_idx]:
                ball = ball_per_frame[frame_idx]
                if ball and hasattr(ball, 'xyxy'):
                    x1, y1, x2, y2 = map(int, ball.xyxy)
                    bx = int((x1+x2)/2)
                    by = y1 - 10
                    # Triangolo Arancione Palla
                    pts = np.array([[bx, by], [bx-6, by-12], [bx+6, by-12]], np.int32)
                    cv2.fillPoly(frameSpec, [pts], (0, 140, 255)) 

            # --- PALLA (MINIMAPPA) ---
            if tactical_ball_per_frame and frame_idx < len(tactical_ball_per_frame):
                t_ball = tactical_ball_per_frame[frame_idx]
                if t_ball:
                    tx, ty = int(t_ball[0]), int(t_ball[1])
                    # Pallino Arancione per la Palla
                    cv2.circle(current_tactical, (tx, ty), 8, (0,0,0), -1) # Bordo
                    cv2.circle(current_tactical, (tx, ty), 6, (0, 140, 255), -1) # Arancione Basket

            # --- COMPOSIZIONE ---
            final_frame = self.composeFrame(frameSpec, current_tactical, pos=(20, 20))
            frames_out.append(final_frame)

        return frames_out