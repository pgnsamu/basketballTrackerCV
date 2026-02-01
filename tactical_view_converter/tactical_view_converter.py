import cv2
import numpy as np
from copy import deepcopy
from homography.homography import Homography

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.court_img = cv2.imread(court_image_path)
        
        self.width = 598
        self.height = 321
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15 

        # Keypoints Standard (FIBA/NBA court model)
        self.key_points = [
            (0,0), (0, int((0.91/15)*321)), (0, int((5.18/15)*321)), (0, int((10/15)*321)), (0, int((14.1/15)*321)), (0, 321), # 0-5
            (int(598/2), 321), (int(598/2), 0), # 6-7
            (int((5.79/28)*598), int((5.18/15)*321)), (int((5.79/28)*598), int((10/15)*321)), # 8-9
            (598, 321), (598, int((14.1/15)*321)), (598, int((10/15)*321)), (598, int((5.18/15)*321)), (598, int((0.91/15)*321)), (598, 0), # 10-15
            (int(((28-5.79)/28)*598), int((5.18/15)*321)), (int(((28-5.79)/28)*598), int((10/15)*321)) # 16-17
        ]
        
        self.current_side = "left" 
        self.frame_width = 0 

    def getKeypointsForOpencv(self) -> np.ndarray:
        return np.array(self.key_points, dtype=np.float32)

    def determine_court_side(self, keypoints, frame_width):
        """Determina il lato del campo basandosi sulla linea centrale"""
        if frame_width == 0: return self.current_side
        
        mid_line_indices = [6, 7]
        detected_mids = [kp for i, kp in enumerate(keypoints) if i in mid_line_indices and kp[0] > 0]

        if not detected_mids: return self.current_side 

        avg_x = np.mean([kp[0] for kp in detected_mids])
        
        if avg_x > frame_width * 0.6: return "left"
        elif avg_x < frame_width * 0.4: return "right"
        
        return self.current_side

    def correct_keypoint_ids(self, keypoints, side):
        """Scambia ID se rilevato lato errato (Simmetria)"""
        corrected_kps = deepcopy(keypoints)
        
        left_to_right = { 0: 15, 1: 14, 2: 13, 3: 12, 4: 11, 5: 10, 8: 17, 9: 16 }
        right_to_left = {v: k for k, v in left_to_right.items()}

        detected_indices = [i for i, kp in enumerate(keypoints) if kp[0] > 0]
        
        if side == "right":
            left_count = sum(1 for i in detected_indices if i in left_to_right)
            if left_count > 2:
                new_kps = np.zeros_like(corrected_kps)
                for i, kp in enumerate(corrected_kps):
                    if i in left_to_right and kp[0] > 0: new_kps[left_to_right[i]] = kp
                    elif i in [6, 7]: new_kps[i] = kp
                return new_kps

        elif side == "left":
            right_count = sum(1 for i in detected_indices if i in right_to_left)
            if right_count > 2:
                new_kps = np.zeros_like(corrected_kps)
                for i, kp in enumerate(corrected_kps):
                    if i in right_to_left and kp[0] > 0: new_kps[right_to_left[i]] = kp
                    elif i in [6, 7]: new_kps[i] = kp
                return new_kps

        return corrected_kps

    def validate_keypoints(self, keypoints_list):
        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, players_list, ball_list=None):
        """
        Trasforma Giocatori E Palla nella vista tattica.
        Returns:
            tuple: (tactical_players_list, tactical_ball_list)
        """
        tactical_positions_list = []
        tactical_ball_list = []
        
        frame_w = 1280 

        for i, (frame_kps, frame_players) in enumerate(zip(keypoints_list, players_list)):
            frame_ball = ball_list[i] if ball_list and i < len(ball_list) else None
            
            # Default empty outputs
            frame_tactical_data = {}
            frame_tactical_ball = None

            if frame_kps is None: 
                tactical_positions_list.append(frame_tactical_data)
                tactical_ball_list.append(frame_tactical_ball)
                continue

            # 1. Lato e Correzione
            current_side = self.determine_court_side(frame_kps, frame_w)
            self.current_side = current_side
            corrected_kps = self.correct_keypoint_ids(frame_kps, current_side)
            
            # 2. Omografia
            valid_indices = [idx for idx, kp in enumerate(corrected_kps) if kp[0] > 0]
            if len(valid_indices) < 4:
                tactical_positions_list.append(frame_tactical_data)
                tactical_ball_list.append(frame_tactical_ball)
                continue
                
            src_pts = np.array([corrected_kps[idx] for idx in valid_indices], dtype=np.float32)
            dst_pts = np.array([self.key_points[idx] for idx in valid_indices], dtype=np.float32)
            
            try:
                homography = Homography(src_pts, dst_pts)
                
                # --- TRASFORMAZIONE GIOCATORI ---
                for player in frame_players:
                    if player.track_id is None: continue
                    foot_point = np.array([player.foot], dtype=np.float32).reshape(1, 1, 2)
                    transformed = cv2.perspectiveTransform(foot_point, homography.homography_matrix)
                    tx, ty = transformed[0][0]
                    
                    if 0 <= tx <= self.width and 0 <= ty <= self.height:
                        frame_tactical_data[player.track_id] = [tx, ty]
                
                # --- TRASFORMAZIONE PALLA ---
                if frame_ball is not None:
                    # Usiamo il centro della palla o il piede? Per la palla in aria, meglio il centro proiettato
                    # Ma bbox_utils.get_foot_position va bene anche per la palla (base del box)
                    ball_pt = np.array([frame_ball.foot], dtype=np.float32).reshape(1, 1, 2)
                    transformed_ball = cv2.perspectiveTransform(ball_pt, homography.homography_matrix)
                    bx, by = transformed_ball[0][0]
                    if 0 <= bx <= self.width and 0 <= by <= self.height:
                        frame_tactical_ball = [bx, by]

            except Exception:
                pass

            tactical_positions_list.append(frame_tactical_data)
            tactical_ball_list.append(frame_tactical_ball)

        return tactical_positions_list, tactical_ball_list