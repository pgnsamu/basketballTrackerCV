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
            
            left_ids  = [0,1,2,3,4,5,8,9]
            right_ids = [15,14,13,12,11,10,16,17]
            
            
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
                
                
                
                if frame == 68 or frame == 69:
                    for index, left_index  in enumerate(left_ids):
                        if is_zero_point(xy[0][left_index]):
                            continue
                        if is_zero_point(xy[0][right_ids[index]]):
                            continue
                        print( left_ids[index], xy[0][left_ids[index]], " - ", right_ids[index], xy[0][right_ids[index]], " distance",  np.linalg.norm(xy[0][left_ids[index]] - xy[0][right_ids[index]])) # 3 tensor([353.4377, 682.8314])  -  12 tensor([349.6709, 689.6677])  distance 7.8052964 
                    
                    
                # TODO: detecting overlap of keypoints between frames surely it happen around frame 68-69 in Alessio's video still need a fix like leave the last valid keypoints but for how many frames?
                # ------------- IMPORTANT --------------
                # cases:
                # 1) first frame with no cache -> skip
                # 2) first frame with cache -> use cache
                # 2.1) check the mirror keypoints
                # 2.2) check distance in 2 cases
                # 2.2.1) left point new and right point cached
                # 2.2.2) right point new and left point cached
                # 3) measure distance between new points and cached points if too close discard new point
                # the threshold can is set to 15 pixels for now but can be tuned
                if counter > 0:
                    for index, left_index  in enumerate(left_ids):
                        
                        # left new right cached
                        point = xy[0][left_ids[index]]
                        
                        xy_prev = cache_keypoints.keypoints.xy
                        point2 = xy_prev[0][right_ids[index]]
                        # OLD (Causing Error)
                        # distance1 = np.linalg.norm(point - point2)

                        # NEW (Correct)
                        distance1 = torch.linalg.norm(point - point2).item()
                        if distance1 < 15:  
                            #print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
                            #point2
                            #continue
                            pass
                        # right new left cached    
                        point = xy[0][right_ids[index]]
                        
                        xy_prev = cache_keypoints.keypoints.xy
                        point2 = xy_prev[0][left_ids[index]]
                        
                                                # OLD (Causing Error)
                        # distance1 = np.linalg.norm(point - point2)

                        # NEW (Correct)
                        distance = torch.linalg.norm(point - point2).item()            
                        if distance < 15: 
                            #print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
                            #point1
                            #continue
                            pass


                if xy is None or xy.shape[0] == 0:
                    court_keypoints.append(None)
                    continue

                # prendi la prima detection (assumendo 1 campo)
                pts = xy[0].cpu().numpy().astype(np.float32)   # (18,2)
                court_keypoints.append(pts)
                cache_keypoints = detection
                counter += 1

        save_stub(stub_path,court_keypoints, label)
        
        return court_keypoints
    
    
def is_zero_point(p) -> bool:
    if isinstance(p, torch.Tensor):
        return bool((p == 0).all().item())
    p = np.asarray(p)
    return bool((p == 0).all())

