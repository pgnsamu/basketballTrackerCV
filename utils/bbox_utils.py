"""
A module providing utility functions for bounding box calculations and measurements.

This module contains helper functions for working with bounding boxes, including
calculations for centers, widths, and distances between points.
"""

def get_center_of_bbox(bbox):
    """
    Calculate the center coordinates of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates in format (x1, y1, x2, y2).

    Returns:
        tuple: Center coordinates (x, y) of the bounding box.
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates in format (x1, y1, x2, y2).

    Returns:
        int: Width of the bounding box.
    """
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (tuple): First point coordinates (x, y).
        p2 (tuple): Second point coordinates (x, y).

    Returns:
        float: Euclidean distance between the two points.
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def check_side(p1: tuple[float, float], pc: tuple[float, float], id_p1: int, id_frame: int) -> int:
    '''
    Check if point p1 is on the side of the court where it is supposed to be.
    Args:
        p1 (tuple): Point coordinates (x, y).
        pc (tuple): Center point coordinates (x, y). e.g., midpoint points 6 and 7.
        id_p1 (int): ID of the point to check.
        id_frame (int): Frame index (for debugging purposes).
    Returns:
        int: 1 if the point is on the correct side, 0 otherwise.
    '''
    left_ids  = {0,1,2,3,4,5,8,9}
    right_ids = {10,11,12,13,14,15,16,17}

    if p1[0] > pc[0]:          # a destra
        return 1 if id_p1 in right_ids else 0
    else:                      # a sinistra (o uguale)
        return 1 if id_p1 in left_ids else 0

def measure_xy_distance(p1,p2):
    """
    Calculate the separate x and y distances between two points.

    Args:
        p1 (tuple): First point coordinates (x, y).
        p2 (tuple): Second point coordinates (x, y).

    Returns:
        tuple: The (x_distance, y_distance) between the points.
    """
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    """
    Calculate the position of the bottom center point of a bounding box.

    Args:
        bbox (tuple): Bounding box coordinates in format (x1, y1, x2, y2).

    Returns:
        tuple: Coordinates (x, y) of the bottom center point.
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)