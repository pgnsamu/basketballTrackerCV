import cv2
import sys 
import numpy as np
sys.path.append('../')

class Homography:
    def __init__(self, source_points, destination_points):
        self.source_points = source_points
        self.destination_points = destination_points
        #print("len of points to transform:", len(source_points))
        #print("len of destination points:", len(destination_points))
        
        # Remove invalid destination points (e.g., [-1, -1])
        valid_mask = ~((self.destination_points[:, 0] <= 0) & (self.destination_points[:, 1] <= 0))
        valid_indices = self.destination_points[valid_mask]
        self.source_points = self.source_points[valid_mask]
        if len(valid_indices) >= 4:
            self.homography_matrix, _ = cv2.findHomography(self.source_points, valid_indices)
            if self.homography_matrix is None:
                raise ValueError("Could not compute homography matrix.")
        
    def transform_points(self, points: np.ndarray, inverse: bool = False, offset=(0,0)) -> np.ndarray:
        """
        points: np.ndarray shape (N,2)
        returns: np.ndarray shape (N,2)
        """
        
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if inverse:
            homography_matrix = np.linalg.inv(self.homography_matrix)
        else:
            homography_matrix = self.homography_matrix
        transformed = cv2.perspectiveTransform(points, homography_matrix)
        return transformed.reshape(-1, 2)
    
    
    
        