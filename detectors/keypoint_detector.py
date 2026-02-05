from ultralytics import YOLO
import sys 
sys.path.append('../')
import numpy as np
import torch
from utils import read_stub, save_stub


class CourtKeypointDetector:
    """
    The CourtKeypointDetector class uses a YOLO model to detect court keypoints in image frames. 
    It also provides functionality to draw these detected keypoints on the frames.
    """
    def __init__(self, model_path, conf_threshold=0.6):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
    
    def get_court_keypoints(self, frames,read_from_stub=False, stub_path=None, label=None) -> list[np.ndarray]:
        """
        Detect court keypoints for a batch of frames using the YOLO model. If requested, 
        attempts to read previously detected keypoints from a stub file before running the model.

        Args:
            frames (list of numpy.ndarray): A list of frames (images) on which to detect keypoints.
            read_from_stub (bool, optional): Indicates whether to read keypoints from a stub file 
                instead of running the detection model. Defaults to False.
            stub_path (str, optional): The file path for the stub file. If None, a default path may be used. 
                Defaults to None.
            label (str, optional): The label for the video processed. If None, a default path may be used. 
                Defaults to None.

        Returns:
            list: A list of detected keypoints for each input frame.
        """
        court_keypoints = read_stub(read_from_stub,stub_path, label)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):
                print("-------------------------------------")
                return court_keypoints
        
        batch_size=20
        court_keypoints = []
        
        cache_keypoints = []
        
        
        frame = 0
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=self.conf_threshold)
            # detections_batch is a list of Results objects, one per frame in the batch
            # Each Results object contains:
            # - keypoints: Keypoints object with detected keypoints (xy, xyn, conf)
            # - boxes: Boxes object with bounding boxes (if any)
            # - Other prediction data from YOLO
            
            counter = 0
            for detection in detections_batch:
                if detection.keypoints is None:
                    court_keypoints.append(None)
                    continue
                frame += 1
                # tensor: (n_instances, 18, 2)
                xy = detection.keypoints.xy
                if xy is None or xy.numel() == 0 or xy.shape[1] == 0:
                    # qui sei nel caso: tensor([], size=(1,0,2)) oppure comunque vuoto
                    court_keypoints.append(None)
                    continue

                if xy is None or xy.shape[0] == 0:
                    court_keypoints.append(None)
                    continue

                # prendi la prima detection (assumendo 1 campo)
                pts = xy[0].cpu().numpy().astype(np.float32)   # (18,2)
                court_keypoints.append(pts)
                counter += 1

        save_stub(stub_path,court_keypoints, label)
        
        return court_keypoints
    
    
def is_zero_point(p) -> bool:
    if isinstance(p, torch.Tensor):
        return bool((p == 0).all().item())
    p = np.asarray(p)
    return bool((p == 0).all())

