from ultralytics import YOLO
from rfdetr import RFDETRNano
import supervision as sv
import numpy as np
import torch
import cv2
from utils import read_stub, save_stub, Player, Ball, detections_to_players, players_to_detections
from typing import Optional

class PlayerBallDetector:
    def __init__(self, model_path: str, yolo: bool = True, optimize: bool = True):
        """
        Inizializza il rilevatore con il modello YOLO11 o RFDETRNano.
        """
        
        self.using_yolo = yolo   
        # 1. CARICAMENTO MODELLO
        try:
            if self.using_yolo:
                self.model = YOLO(model_path)
            else:
                self.model = RFDETRNano(
                    pretrain_weights=model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
        except Exception as e:
            raise FileNotFoundError(f"Impossibile caricare il modello: {e}")

        # 2. SETUP GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # 3. OTTIMIZZAZIONE (Fusing)
        # Unisce i layer Conv2d + BatchNorm per velocizzare l'inferenza su GPU
        try:
            if self.using_yolo and optimize:
                self.model.fuse()
            elif not self.using_yolo and optimize:
                self.model.optimize_for_inference()
                self.optimized = True
        except Exception as e:
             f"[WARN] optimize_for_inference() failed; running without TorchScript optimization. Error: {e}"

        # 4. TRACKER
        # ByteTrack è ottimo per gestire le occlusioni dei giocatori
        try:
            self.TRACKER = sv.ByteTrack()
        except Exception:
            self.TRACKER = sv.Sort()
            
            
        # 5. PARAMETRI INFERENZA
        self.INFERENCE_SIZE = 1280  # FONDAMENTALE: Hai allenato a 1280px!
        self.CONF_THRESHOLD = 0.45  # Confidenza base per accettare una detection
        
        if self.using_yolo:
            self.CLASS_ID_BALL = 0
            self.CLASS_ID_PLAYER = 3
        else:
            self.CLASS_ID_BALL = 1
            self.CLASS_ID_PLAYER = 4
        
        self.NMS_THRESHOLD = 0.45  # Non-Maximum Suppression threshold

        # Logica di gioco (Possesso e Smoothing)
        self.BALL_EMA_ALPHA = 0.75
        self.BALL_KEEP_FRAMES = 12
        self.POSSESSION_DIST_PX = 110
        self.STABLE_FRAMES = 5
        
        
        # Detection filtering
        self.ALLOWED_CLASS_IDS = {self.CLASS_ID_BALL, self.CLASS_ID_PLAYER}
        self.EXCLUDED_CLASS_IDS = {9}    # 9=referee can be deleted
        self.CLASS_THRESHOLDS = {self.CLASS_ID_BALL: 0.25, self.CLASS_ID_PLAYER: 0.70, "default": 0.30}
        
        
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

        if not self.using_yolo: 
            with torch.no_grad():
                dets = self.model.predict(frame, threshold=0.10, imgsz=self.INFERENCE_SIZE)
                
            dets = dets.with_nms(threshold=self.NMS_THRESHOLD)
            dets = self.filter_detections(dets)
        else:
            results = self.model.predict(
                frame, 
                imgsz=self.INFERENCE_SIZE, 
                conf=0.10, # Teniamo basso qui, filtriamo dopo
                #iou=self.NMS_THRESHOLD,
                device=self.device,
                verbose=False # Silenzia l'output nella console
            )[0] # Prendi il primo risultato (singolo frame)  
            
            # 2. Conversione in Supervision
            detections = sv.Detections.from_ultralytics(results) 
            dets = self.filter_detections(detections)
            
        
        return dets

    def getPlayersPosition(self, frame, dets, read_from_stub=False, stub_path=None) -> list[Player]:
        """
        Detect player positions in a given frame using the RFDETR model.
        Args:
            frame (numpy.ndarray): The input image frame.
            read_from_stub (bool, optional): Indicates whether to read detections from a stub file. Defaults to False.
            stub_path (str, optional): The file path for the stub file. Defaults to None.
        Returns:
            list[Player]: A list of detected player positions in the frame.
        """
        

        players = dets[dets.class_id == self.CLASS_ID_PLAYER] if len(dets) else sv.Detections.empty()
        player_list: list[Player] = [
            Player(xyxy=box.copy(), conf=float(conf), class_id=self.CLASS_ID_PLAYER, track_id=None) for box, conf in zip(players.xyxy, players.confidence)
        ]
        
        return player_list
        
    def getBallposition(self, frame, dets, read_from_stub=False, stub_path=None) -> list[Ball]:
        """
        Detect ball position in a given frame using the RFDETR model.
        Args:
            frame (numpy.ndarray): The input image frame.
            read_from_stub (bool, optional): Indicates whether to read detections from a stub file. Defaults to False.
            stub_path (str, optional): The file path for the stub file. Defaults to None.
        Returns:
            list[Ball]: A list of detected ball positions in the frame.
        """

        balls = dets[dets.class_id == self.CLASS_ID_BALL] if len(dets) else sv.Detections.empty() 
        
        ball_list: list[Ball] = [
            Ball(xyxy=box.copy(), conf=float(conf), class_id=self.CLASS_ID_BALL) for box, conf in zip(balls.xyxy, balls.confidence)
        ]
        
        return ball_list
    
    def getBallPlayersPositions(self, frames: list[np.ndarray], read_from_stub=False, stub_path=None) -> tuple[list[list[Player]], list[list[Ball]]]:
        """
        Detect players positions in the full video
        Args:
            frames (list[np.ndarray]): The frames of the video.
            read_from_stub (bool, optional): Indicates whether to read detections from a stub file. Defaults to False.
            stub_path (str, optional): The file path for the stub file. Defaults to None. (should refer to players)
        
        Returns:
            list[list[Ball]]: A list of detected ball positions in the frame.
        """
        
        players_positions = read_stub(read_from_stub,stub_path)
        ball_positions = read_stub(read_from_stub,stub_path.replace('players','balls'))
        if players_positions is not None and len(players_positions) == len(frames):
            if ball_positions is not None and len(ball_positions) == len(frames):
                return (players_positions, ball_positions)
        
        players_positions = []
        ball_positions = []
        
        poss_candidate = None
        poss_streak = 0
        current_possessor = None
        
        last_ball = None
        last_keep = self.BALL_KEEP_FRAMES
        
        for frame in frames:
            dets = self.getDetections(frame)
            
            players = self.getPlayersPosition(frame, dets)
            
            if len(players) > 0:
                tracked_dets = self.TRACKER.update_with_detections(players_to_detections(players))
                tracked_players: list[Player] = detections_to_players(tracked_dets)
            else:
                tracked_players = players
                

            balls = self.getBallposition(frame, dets)
            

            
            smoothed_ball, last_keep, ball_object = self.pick_ball_and_smooth(balls, last_ball, last_keep)
            
            candidate = self.find_possessor(tracked_players, smoothed_ball)
            if candidate == poss_candidate:
                poss_streak += 1
            else:
                poss_candidate = candidate
                poss_streak = 1

        # TODO: this seems like to not working
            if poss_candidate is not None and poss_streak >= self.STABLE_FRAMES:
                tracked_players[int(poss_candidate)].class_id = 99  # Mark possessor with special class_id 
            

            if tracked_players is not None:
                players_positions.append(tracked_players)
            else:
                players_positions.append([])
            if ball_object is not None:
                ball_positions.append(ball_object)
            else:
                ball_positions.append(None)

        save_stub(stub_path,players_positions)
        save_stub(stub_path.replace('players','balls'),ball_positions)
        
        return (players_positions, ball_positions)
    
    def pick_ball_and_smooth(self, balls: list[Ball], last_ball: Optional[Ball], last_keep: int) -> tuple[Optional[tuple[float, float]], int, Optional[Ball]]:
        '''
        Pick the most confident ball detection and apply EMA smoothing.
        Args:
            balls (list[Ball]): List of ball detections.
            last_ball (Ball): Last smoothed ball position.
            last_keep (int): Number of frames since last valid ball detection.
        Returns:
            tuple: (smoothed ball position [centers], updated frames last_keep, ball Object or None)
        '''
        if balls is not None and len(balls) > 0:
            idx = int(np.argmax([b.conf for b in balls]))

            if last_ball is None:
                smoothed = balls[idx].center
            else:
                smoothed = (
                    self.BALL_EMA_ALPHA * balls[idx].center[0] + (1 - self.BALL_EMA_ALPHA) * last_ball.center[0],
                    self.BALL_EMA_ALPHA * balls[idx].center[1] + (1 - self.BALL_EMA_ALPHA) * last_ball.center[1]
                )
            
            return smoothed, 0, balls[idx]
            

        if last_ball is not None and last_keep < self.BALL_KEEP_FRAMES:
            return last_ball.center, last_keep + 1, None

        return None, self.BALL_KEEP_FRAMES, None
        
       
    def find_possessor(self, players: list[Player], ball_center: tuple[float, float]) -> int:
        '''
        Find the player in possession of the ball based on proximity.
        Args:
            players (list[Player]): Detected players.
            ball_center (tuple[float, float]): Ball coordinates (x, y).
        Returns:
        int or None: Tracker ID of the player in possession, or None if no player is close enough.
        '''
        if ball_center is None or players is None or len(players) == 0:
            return None

        bx, by = ball_center
        best_track_id = None
        best_dist = 1e9

        for player in players:
            if player.track_id is None:
                continue
            
            track_id = int(player.track_id)
            player_x1, player_y1, player_x2, player_y2 = player.as_int_tuple()
            
            inside = (bx >= player_x1 and bx <= player_x2 and by >= player_y1 and by <= player_y2)
            if inside:
                dist = 0.0
            else:
                dist = float(np.hypot(bx - player.center[0], by - player.center[1]))

            if dist < best_dist:
                best_dist = dist
                best_track_id = track_id

        if best_dist > self.POSSESSION_DIST_PX:
            return None
        return best_track_id
    
if __name__ == "__main__":
    detector = PlayerBallDetector('models/bestEMA.pth', optimize=False)
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
    
    