from rfdetr import RFDETRNano
import supervision as sv
import numpy as np
import torch
import cv2
from utils import read_stub, save_stub, DetectedObject

#TODO: change the name of the class to PlayerBallDetector  
class PlayerDetector:
    def __init__(self, model_path: str, optimize: bool = True):
        self.model = RFDETRNano(
            pretrain_weights=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # NOTE: `optimize_for_inference()` uses `torch.jit.trace`. If the model's forward
        # returns non-tensor outputs (e.g., custom objects / dicts), tracing can fail with:
        # "Only tensors, lists, tuples of tensors, or dictionary of tensors can be output..."
        # In that case we just skip optimization and run eager inference.
        self.optimized = False
        if optimize:
            try:
                self.model.optimize_for_inference()
                self.optimized = True
            except RuntimeError as e:
                print(
                    f"[WARN] optimize_for_inference() failed; running without TorchScript optimization. Error: {e}"
                )
        self.INFERENCE_SIZE = 1280 # Resize the input image to this size for inference
        self.NMS_THRESHOLD = 0.45
        self.ALLOWED_CLASS_IDS = {1, 4}  # 1=ball, 4=player
        self.EXCLUDED_CLASS_IDS = {9}    # 9=referee
        self.CLASS_THRESHOLDS = {1: 0.25, 4: 0.70, "default": 0.30}
    
    def filter_detections(self, dets: sv.Detections):
        """
        Filter detections based on class IDs and confidence thresholds.
        Args:
            dets (sv.Detections): The detections to filter.
        Returns:
            sv.Detections: The filtered detections.
        """
        if dets is None or len(dets) == 0:
            return sv.Detections.empty()

        keep = []
        for i, (cid, conf) in enumerate(zip(dets.class_id, dets.confidence)):
            cid = int(cid)
            if cid in self.EXCLUDED_CLASS_IDS:
                continue
            if cid not in self.ALLOWED_CLASS_IDS:
                continue
            thr = self.CLASS_THRESHOLDS.get(cid, self.CLASS_THRESHOLDS["default"])
            if float(conf) < thr:
                continue
            keep.append(i)

        if not keep:
            return sv.Detections.empty()
        return dets[np.array(keep)]
    
    def getDetections(self, frame, read_from_stub=False, stub_path=None) -> sv.Detections:
        """
        Get filtered detections for a given frame using the RFDETR model.
        Args:
            frame (numpy.ndarray): The input image frame.
            read_from_stub (bool, optional): Indicates whether to read detections from a stub file. Defaults to False.
            stub_path (str, optional): The file path for the stub file. Defaults to None.
        Returns:
            sv.Detections: The filtered detections in the frame.
        """
        with torch.no_grad():
            dets = self.model.predict(frame, threshold=0.10, imgsz=self.INFERENCE_SIZE)
        
        dets = dets.with_nms(threshold=self.NMS_THRESHOLD)
        dets = self.filter_detections(dets)
        
        
        return dets
    
    def getPlayersPosition(self, frame, dets, read_from_stub=False, stub_path=None) -> list[DetectedObject]:
        """
        Detect player positions in a given frame using the RFDETR model.
        Args:
            frame (numpy.ndarray): The input image frame.
            read_from_stub (bool, optional): Indicates whether to read detections from a stub file. Defaults to False.
            stub_path (str, optional): The file path for the stub file. Defaults to None.
        Returns:
            list[DetectedObject]: A list of detected player positions in the frame.
        """
        

        players = dets[dets.class_id == 4] if len(dets) else sv.Detections.empty()
        player_list: list[DetectedObject] = [
            DetectedObject(xyxy=box.copy(), conf=float(conf), class_id=4) for box, conf in zip(players.xyxy, players.confidence)
        ]
        
        return player_list
        
    def getBallposition(self, frame, dets, read_from_stub=False, stub_path=None) -> list[DetectedObject]:
        """
        Detect ball position in a given frame using the RFDETR model.
        Args:
            frame (numpy.ndarray): The input image frame.
            read_from_stub (bool, optional): Indicates whether to read detections from a stub file. Defaults to False.
            stub_path (str, optional): The file path for the stub file. Defaults to None.
        Returns:
            list[DetectedObject]: A list of detected ball positions in the frame.
        """

        balls = dets[dets.class_id == 1] if len(dets) else sv.Detections.empty() 
        
        ball_list: list[DetectedObject] = [
            DetectedObject(xyxy=box.copy(), conf=float(conf), class_id=1) for box, conf in zip(balls.xyxy, balls.confidence)
        ]
        
        return ball_list
    
    def getBallPlayersPositions(self, frames: list[np.ndarray], read_from_stub=False, stub_path=None) -> tuple[list[list[DetectedObject]], list[list[DetectedObject]]]:
        """
        Detect players positions in the full video
        Args:
            frames (list[np.ndarray]): The frames of the video.
            read_from_stub (bool, optional): Indicates whether to read detections from a stub file. Defaults to False.
            stub_path (str, optional): The file path for the stub file. Defaults to None. (should refer to players)
        
        Returns:
            list[list[DetectedObject]]: A list of detected player positions in the frame.
        """
        
        players_positions = read_stub(read_from_stub,stub_path)
        ball_positions = read_stub(read_from_stub,stub_path.replace('players','balls'))
        if players_positions is not None and len(players_positions) == len(frames):
            if ball_positions is not None and len(ball_positions) == len(frames):
                return (players_positions, ball_positions)
        
        players_positions = []
        ball_positions = []
        for frame in frames:
            dets = self.getDetections(frame)
            
            players = self.getPlayersPosition(frame, dets)
            ball = self.getBallposition(frame, dets)

            if players is not None:
                players_positions.append(players)
            else:
                players_positions.append([])
            if ball is not None:
                ball_positions.append(ball)
            else:
                ball_positions.append([])

        #TODO: change in a better way
        save_stub(stub_path,players_positions)
        save_stub(stub_path.replace('players','balls'),ball_positions)
        
        return (players_positions, ball_positions)
        
        
if __name__ == "__main__":
    detector = PlayerDetector('models/bestEMA.pth', optimize=False)
    test_frame = cv2.imread('images/uno.png')
    dets = detector.getDetections(test_frame)
    player_positions = detector.getPlayersPosition(test_frame, dets)
    annotated_frame = test_frame.copy()
    for player in player_positions:
        x1, y1, x2, y2 = player.as_int_tuple()
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated_frame, (int(player.center[0]), int(player.center[1])), 5, (0, 0, 255), -1)

    cv2.imshow("Players Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print("Detected player positions:", player_positions)
    
    
