import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from homography.homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import get_foot_position, measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 598
        self.height= 321

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15 

        self.key_points = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            # Middle line
            (int(self.width/2),self.height),
            (int(self.width/2),0),
            
            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            # right edge
            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]
    def getKeypointsForOpencv(self) -> np.ndarray:
        """
        Returns the tactical view keypoints as a numpy array suitable for OpenCV functions.
        
        Returns:
            np.ndarray: An array of shape (18, 2) containing the tactical view keypoints.
        """
        return np.array(self.key_points, dtype=np.float32)  # (18,2)

    def validate_keypoints(self, keypoints_list: list[np.ndarray]) -> list[np.ndarray]:
        """
        Args:
            keypoints_list: list of np.ndarray, each of shape (18,2), dtype float32
                            (0,0) means not detected
        Returns:
            list[np.ndarray]: validated keypoints, same structure
        """

        keypoints_list = deepcopy(keypoints_list)
        tactical_pts = np.array(self.key_points, dtype=np.float32)  # (18,2)

        for frame_idx, frame_kps in enumerate(keypoints_list):

            if frame_kps is None:
                continue

            # sicurezza
            frame_kps = np.asarray(frame_kps, dtype=np.float32)

            # indici dei keypoint rilevati
            detected_points = [
                point for point, keypoint_coordinates in enumerate(frame_kps)
                if keypoint_coordinates[0] > 0 and keypoint_coordinates[1] > 0
            ]

            # servono almeno 3 punti
            if len(detected_points) < 3:
                continue

            invalid_keypoints = []

            for p0 in detected_points:

                if p0 in invalid_keypoints:
                    continue

                if frame_kps[p0][0] == 0 and frame_kps[p0][1] == 0:
                    continue

                other_indices = [
                    idx for idx in detected_points
                    if idx != p0 and idx not in invalid_keypoints
                ]

                if len(other_indices) < 2:
                    continue

                p1, p2 = other_indices[0], other_indices[1]

                # distanze nel frame
                distance_p0_p1 = measure_distance(frame_kps[p0], frame_kps[p1])
                distance_p0_p2 = measure_distance(frame_kps[p0], frame_kps[p2])
                if distance_p0_p2 == 0:
                    continue

                # distanze nella vista tattica
                distance_p0_p1_tactic = measure_distance(tactical_pts[p0], tactical_pts[p1])
                distance_p0_p2_tactic = measure_distance(tactical_pts[p0], tactical_pts[p2])

                if distance_p0_p2_tactic == 0:
                    continue

                prop_detected = distance_p0_p1 / distance_p0_p2
                prop_tactical = distance_p0_p1_tactic / distance_p0_p2_tactic
                
                error = abs(prop_detected - prop_tactical) / abs(prop_tactical)

                if error > 0.8:  # 80% di errore
                    frame_kps[p0] = (0.0, 0.0)
                    invalid_keypoints.append(p0)

            keypoints_list[frame_idx] = frame_kps

        return keypoints_list
'''
    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.
        
        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            player_tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        
        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            frame_keypoints = frame_keypoints.xy.tolist()[0]

            # Skip frames with insufficient keypoints
            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Get detected keypoints for this frame
            detected_keypoints = frame_keypoints
            
            # Filter out undetected keypoints (those with coordinates (0,0))
            valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]
            
            # Need at least 4 points for a reliable homography
            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Create source and target point arrays for homography
            source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32)
            target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
            
            try:
                # Create homography transformer
                homography = Homography(source_points, target_points)
                
                # Transform each player's position
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    # Use bottom center of bounding box as player position
                    player_position = np.array([get_foot_position(bbox)])
                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(player_position)

                    # If tactical position is not in the tactical view, skip
                    if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height:
                        continue

                    tactical_positions[player_id] = tactical_position[0].tolist()
                    
            except (ValueError, cv2.error) as e:
                # If homography fails, continue with empty dictionary
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions
'''
