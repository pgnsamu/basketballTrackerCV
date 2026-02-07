import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from homography.homography import Homography
from utils import Player
from itertools import combinations

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import measure_distance, check_side

class TacticalViewConverter:
    def __init__(self, court_image_path, video_width, video_height):
        self.court_image_path = court_image_path
        self.width = 598
        self.height= 321
        

        self.frame_height = video_height
        self.frame_width = video_width

        # TODO: calculate this value based on the video resolution maybe a 5%/10% of the width
        # self.THRESHOLD_DISTANCE = self.frame_width * 0.05 
        self.THRESHOLD_DISTANCE = 70

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
        sides ={ 0: left_ids, 1: right_ids }

        cache_keypoints = []
        # improvement: when a point disappear for a few frames, and the others dont change too much, then we can assume that the point is still there and interpolate it.
        multi_cache_keypoints = []

        first_impostor_keypoints = []
        second_impostor_keypoints = []
        third_impostor_keypoints = []

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

            if frame_idx > 0:
                # TODO: in teoria lo switch dei punti dovrebbe rimanere finche non si rilevano i punti centrali e quelli nuovi vengano messi dall'altra parte della linea
                # in questo caso si fa fede alla prima detection valida
                for index, left_index  in enumerate(left_ids):   
                    # left new right cached
                    point = frame_kps[left_ids[index]]
                    

                    if len(cache_keypoints) > 0 and (cache_keypoints[right_ids[index]][0] > 0 or cache_keypoints[right_ids[index]][1] > 0):
                        point2 = cache_keypoints[right_ids[index]]
                    else:
                        point2 = (0.0, 0.0)
                        for i in range(2, 11):
                            if frame_idx - i >= 0 and frame_idx - i < len(multi_cache_keypoints) and len(multi_cache_keypoints[frame_idx - i]) > 0:
                                prev_point = multi_cache_keypoints[frame_idx - i][right_ids[index]]
                                if prev_point[0] > 0 or prev_point[1] > 0:
                                    point2 = prev_point
                                    break
                
                    distance1 = measure_distance(point, point2)
                    if distance1 < self.THRESHOLD_DISTANCE and distance1 > 0:  
                        # inverti punti
                        frame_kps[right_ids[index]] = frame_kps[left_ids[index]]
                        frame_kps[left_ids[index]] = (0.0, 0.0)
                        detected_points.remove(left_ids[index])
                        detected_points.append(right_ids[index])

                        #point2
                        first_impostor_keypoints.append((frame_idx, (left_ids[index], right_ids[index])))
                        continue

                    # right new left cached    
                    point = frame_kps[right_ids[index]]
                    
                    after_check = False
                    if len(cache_keypoints) > 0 and (cache_keypoints[left_ids[index]][0] > 0 or cache_keypoints[left_ids[index]][1] > 0):
                        point2 = cache_keypoints[left_ids[index]]
                    else:
                        point2 = (0.0, 0.0)
                        for i in range(2, 11):
                            if frame_idx - i >= 0 and frame_idx - i < len(multi_cache_keypoints) and len(multi_cache_keypoints[frame_idx - i]) > 0:
                                prev_point = multi_cache_keypoints[frame_idx - i][left_ids[index]]
                                if prev_point[0] > 0 or prev_point[1] > 0:
                                    point2 = prev_point
                                    break

                    distance = measure_distance(point, point2)            
                    if distance < self.THRESHOLD_DISTANCE and distance > 0: 
                        # inverti punti
                        frame_kps[left_ids[index]] = frame_kps[right_ids[index]]
                        frame_kps[right_ids[index]] = (0.0, 0.0)
                        detected_points.remove(right_ids[index])
                        detected_points.append(left_ids[index])

                        #point1
                        second_impostor_keypoints.append((frame_idx, (right_ids[index], left_ids[index])))
                        continue
            
            detected_points_copy = detected_points.copy()
            for point, point2 in combinations(detected_points, 2):
                if point == point2:
                    detected_points_copy.remove(point)
                    continue
                    
            


            for point in detected_points_copy:
                #better side
                if (6 in detected_points_copy or 7 in detected_points_copy) and point != 6 and point !=7:
                    side_result, _ = check_side(frame_kps[point], frame_kps[7 if 7 in detected_points_copy else 6], point, frame_idx)
                    if side_result == 0:
                        frame_kps[point] = (0.0, 0.0)
                        detected_points_copy.remove(point)
                        continue
                side, percentage = self.get_side_of_court(frame_kps)
                if point in left_ids:
                    if side == 0:
                        #godi
                        pass
                    else:
                        # scambia da sinistra a destra
                        sus_index = left_ids.index(point)
                        frame_kps[right_ids[sus_index]] = frame_kps[left_ids[sus_index]]
                        frame_kps[left_ids[sus_index]] = (0.0, 0.0)
                        detected_points_copy.remove(left_ids[sus_index])
                        detected_points_copy.append(right_ids[sus_index])

                        third_impostor_keypoints.append((frame_idx, (left_ids[sus_index], right_ids[sus_index])))
                        
                elif point in right_ids:
                    if side == 1:
                        #godi
                        pass
                    else:
                        # scambia da destra a sinistra
                        sus_index = right_ids.index(point)
                        frame_kps[left_ids[sus_index]] = frame_kps[right_ids[sus_index]]
                        frame_kps[right_ids[sus_index]] = (0.0, 0.0)
                        detected_points_copy.remove(right_ids[sus_index])
                        detected_points_copy.append(left_ids[sus_index])

                        third_impostor_keypoints.append((frame_idx, (right_ids[sus_index], left_ids[sus_index])))
                        
                    
                
            detected_points = detected_points_copy      
                     

            invalid_keypoints = []
            # finding invalid points based on the symmetry on the minimap
            for p0 in detected_points:

                if p0 in invalid_keypoints:
                    continue

                if frame_kps[p0][0] == 0 and frame_kps[p0][1] == 0:
                    continue

                # TODO: pensare in quel caso all'inversione
                if (6 in detected_points or 7 in detected_points) and p0 != 6 and p0 !=7 and p0 not in invalid_keypoints:
                    side_result, side = check_side(frame_kps[p0], frame_kps[7 if 7 in detected_points else 6], p0, frame_idx)
                    if side_result == 0:
                        frame_kps[p0] = (0.0, 0.0)
                        invalid_keypoints.append(p0)
                
                other_values = []
                near_indices = []
                
                for idx, num in enumerate(detected_points):
                    if num != p0 and num not in invalid_keypoints:
                        other_values.append(num)
                    elif num == p0 and num not in invalid_keypoints:
                        # left side
                        if idx > 0:
                            near_indices.append(idx-1)
                        elif idx == 0:
                            near_indices.append(idx+1)
                        # right side
                        if idx < len(detected_points) - 1:
                            near_indices.append(idx)
                        elif idx == len(detected_points) - 1:
                            near_indices.append(idx-2)
                 

                if len(other_values) < 3:
                    continue

                max_error = 0
                # capire quanto aumenta di tempo questo doppio for contando che massimo ci possno essere 8 punti a schermo
                # togliamo p0 quindi 7 punti togliamo un punto a iterazione perché non ci possono essere due punti uguali
                # quindi 7 * 6 = 42 / 2  = 21 eliminando le combinazioni invertite 
                # 168 iterazioni a frame se ci sono esattamente 8 punti n * (n(n-1)/2) AAAAAAAAAAAAAAAAAAAAAAA

                #for p1, p2 in combinations(other_values, 2):
                
                #get p1 and p2 from the same side

                
                if other_values[0] == 6 or other_values[0] == 7 or other_values[1] == 6 or other_values[1] == 7:
                    p1 = other_values[-1]
                    p2 = other_values[-2]
                else:
                    p1, p2 = other_values[0], other_values[1]
                #p3, p4 = other_values[near_indices[0]], other_values[near_indices[1]]

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

                #if error > max_error:
                #    max_error = error


                if error >= 0.7:  # 70% di errore
                    frame_kps[p0] = (0.0, 0.0)
                    invalid_keypoints.append(p0)


                # percentuale di side
                percentage_left = len([p for p in left_ids if frame_kps[p][0] > 0 or frame_kps[p][1] > 0]) / len(left_ids)
                percentage_right = len([p for p in right_ids if frame_kps[p][0] > 0 or frame_kps[p][1] > 0]) / len(right_ids)
                if percentage_left <= 0.250 and p0 in left_ids:
                    frame_kps[p0] = (0.0, 0.0)
                    invalid_keypoints.append(p0)
                elif percentage_right <= 0.250 and p0 in right_ids:
                    frame_kps[p0] = (0.0, 0.0)
                    invalid_keypoints.append(p0)
                
                if p0 not in [6, 7, 8, 9, 16, 17]:  # central points
                    is_valid = self.validate_side_between_points(frame_kps, p0, frame_idx)
                    if not is_valid:
                        frame_kps[p0] = (0.0, 0.0)
                        invalid_keypoints.append(p0)


                # serve per vedere se un punto ha vicino un suo simmetrico nello stesso frame
                if p0 in left_ids:
                    p_right = right_ids[left_ids.index(p0)]
                    distance_p0_p_right = measure_distance(frame_kps[p0], frame_kps[p_right])
                    if distance_p0_p_right < self.THRESHOLD_DISTANCE and distance_p0_p_right > 0:
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
                    if distance_p0_p_left < self.THRESHOLD_DISTANCE and distance_p0_p_left > 0:
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
    
    def validate_side_between_points(self, frame_kps: list[tuple[float, float]], p0: int, frame_idx: int) -> int:
        '''
        
        '''
        left_ids  = [0,  1,  2,  3,  4,  5,  8,  9]
        right_ids = [15, 14, 13, 12, 11, 10, 16, 17]
        sides = {0: left_ids, 1: right_ids}
        free_throw_line = {0: [8,9], 1: [16,17]}

        p1 = frame_kps[p0]

        side_result, percentage = self.get_side_of_court(frame_kps)
        side_ids = sides[side_result]

        central_points = free_throw_line[side_result]

        result = 0
        for pc in central_points:
            if frame_kps[pc][0] == 0.0 and frame_kps[pc][1] == 0.0:
                continue
            if p1[0] > frame_kps[pc][0] and side_result == 1:          # a destra
                # godi
                result = 1
            elif p1[0] < frame_kps[pc][0] and side_result == 0:          # a sinistra
                # godi
                result = 1
            else:                      
                # non godi
                result = 0

        return result
    
    def get_side_of_court(self, frame_kps: list[tuple[float, float]]):
        left_ids  = [0,  1,  2,  3,  4,  5,  8,  9]
        right_ids = [15, 14, 13, 12, 11, 10, 16, 17]

        percentage_left = len([p for p in left_ids if frame_kps[p][0] > 0 or frame_kps[p][1] > 0]) / len(left_ids)
        percentage_right = len([p for p in right_ids if frame_kps[p][0] > 0 or frame_kps[p][1] > 0]) / len(right_ids)

        if percentage_left > percentage_right:
            return 0, percentage_left
        else:
            return 1, percentage_right
        

    
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

    
#TODO: last error between 2800 and 2828