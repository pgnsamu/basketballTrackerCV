import cv2
import sys 
sys.path.append('../')

class Homography:
    def __init__(self, source_points, destination_points):
        self.source_points = source_points
        self.destination_points = destination_points
        
        self.homography_matrix, _ = cv2.findHomography(self.source_points, self.destination_points)
        if self.homography_matrix is None:
            raise ValueError("Could not compute homography matrix.")
        
    def transpose_points(self, points):
        if len(points) == 0:
            return points
        
        points = points.reshape(-1, 1, 2).astype('float32')
        transformed_points = cv2.perspectiveTransform(points, self.homography_matrix)
        return transformed_points.reshape(-1, 2).astype('float32')
    
        