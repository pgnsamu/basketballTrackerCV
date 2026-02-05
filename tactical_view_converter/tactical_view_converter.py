import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from homography.homography import Homography
from utils import Player

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import measure_distance, check_side

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

        left_ids  = [0,  1,  2,  3,  4,  5,  8,  9]
        right_ids = [15, 14, 13, 12, 11, 10, 16, 17]

        cache_keypoints = []
        # improvement: when a point disappear for a few frames, and the others dont change too much, then we can assume that the point is still there and interpolate it.
        multi_cache_keypoints = []

        first_impostor_keypoints = []
        second_impostor_keypoints = []

        for frame_idx, frame_kps in enumerate(keypoints_list):

            if frame_kps is None:
                multi_cache_keypoints.append(cache_keypoints)
                cache_keypoints = []
                continue

            # sicurezza
            frame_kps = np.asarray(frame_kps, dtype=np.float32)

            # indici dei keypoint rilevati
            detected_points = [
                point for point, keypoint_coordinates in enumerate(frame_kps)
                if keypoint_coordinates[0] > 0 or keypoint_coordinates[1] > 0
            ]

            # servono almeno 3 punti
            if len(detected_points) < 3:
                multi_cache_keypoints.append(cache_keypoints)
                cache_keypoints = []
                continue

            invalid_keypoints = []
            # finding invalid points based on the symmetry on the minimap
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
                
                if (6 in detected_points or 7 in detected_points) and p0 != 6 and p0 !=7:
                    side_result = check_side(frame_kps[p0], frame_kps[7 if 7 in detected_points else 6], p0, frame_idx)
                    if side_result == 0:
                        frame_kps[p0] = (0.0, 0.0)
                        invalid_keypoints.append(p0)


                # serve per vedere se un punto ha vicino un suo simmetrico nello stesso frame
                if p0 in left_ids:
                    p_right = right_ids[left_ids.index(p0)]
                    distance_p0_p_right = measure_distance(frame_kps[p0], frame_kps[p_right])
                    if distance_p0_p_right < 15 and distance_p0_p_right > 0:
                        # getting percentage of points of each side and then delete the point which has less points
                        percentage_left = len([p for p in left_ids if frame_kps[p][0] > 0 or frame_kps[p][1] > 0]) / len(left_ids)
                        percentage_right = len([p for p in right_ids if frame_kps[p][0] > 0 or frame_kps[p][1] > 0]) / len(right_ids)
                        if percentage_left < percentage_right:
                            frame_kps[p0] = (0.0, 0.0)
                            invalid_keypoints.append(p0)
                        else:
                            frame_kps[p_right] = (0.0, 0.0)
                            invalid_keypoints.append(p_right)
                    
                elif p0 in right_ids:
                    p_left = left_ids[right_ids.index(p0)]
                    distance_p0_p_left = measure_distance(frame_kps[p0], frame_kps[p_left])
                    if distance_p0_p_left < 15 and distance_p0_p_left > 0:
                        # getting percentage of points of each side and then delete the point which has less points
                        percentage_left = len([p for p in left_ids if frame_kps[p][0] > 0 or frame_kps[p][1] > 0]) / len(left_ids)
                        percentage_right = len([p for p in right_ids if frame_kps[p][0] > 0 or frame_kps[p][1] > 0]) / len(right_ids)
                        if percentage_left < percentage_right:
                            frame_kps[p_left] = (0.0, 0.0)
                            invalid_keypoints.append(p_left)
                        else:
                            frame_kps[p0] = (0.0, 0.0)
                            invalid_keypoints.append(p0)
            
            
            
            keypoints_list[frame_idx] = frame_kps
            cache_keypoints = frame_kps
            multi_cache_keypoints.append(frame_kps)


        print("first_impostor_keypoints", first_impostor_keypoints)
        print("second_impostor_keypoints", second_impostor_keypoints)
        return keypoints_list            
    
    def transform_players_to_tactical_view(self, keypoints_list, players: list[list[Player]]) -> list[dict[int, list[float, float]]]:
        """
        Transform player positions from video frame coordinates to tactical view coordinates.
        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            players (list): List of lists containing detected player objects for each frame.        
        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []
        
        for frame_idx, _ in enumerate(keypoints_list):
            frame_keypoints = keypoints_list[frame_idx]
            players_in_frame = players[frame_idx]
            # Initialize empty dictionary for this frame
            tactical_positions = {}

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
                # print("Creating homography for frame:", frame_idx, "with valid keypoints:", valid_indices)
                # Create homography transformer
                homography = Homography(source_points, target_points)
                
                for player in players_in_frame:
                    # TODO: check if everytime the player has track_id
                    if player.track_id is None:
                        continue
                    player_id = int(player.track_id)
                    bbox = player.xyxy
                    # Use bottom center of bounding box as player position
                    player_position = np.array([player.foot])
                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(player_position.reshape(1, 2), inverse=False)
                    # If tactical position is not in the tactical view, skip
                    if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height:
                        #print("|||||||||||||||||||||||") # TODO: errore probabilmente qui 
                        continue
                    tactical_positions[player_id] = tactical_position[0].tolist()
                    
                    
            except (ValueError, cv2.error) as e:
                # If homography fails, continue with empty dictionary
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions

    
