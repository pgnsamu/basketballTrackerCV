import os
# --- FIX PER ERRORE OMP: Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------------------
import cv2
import numpy as np
from utils import read_video, save_video
from detectors.keypoint_detector import CourtKeypointDetector
from tactical_view_converter.tactical_view_converter import TacticalViewConverter
from homography.homography import Homography
from drawers.drawWindow import DrawWindow
from detectors.player_ball_detector import PlayerBallDetector
from detectors.player_tracker import PlayerTracker
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Basketball Tracker CV - Video Analysis')
    
    parser.add_argument('--video', type=str, default='video_1.mp4',
                        help='Path del video da processare (default: video_1.mp4)')
    parser.add_argument('--output-path', type=str, default='outputVideo/output_video.mp4',
                        help='Path del video di output (default: outputVideo/output_video.mp4)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='FPS del video di output (default: 30.0)')
    
    parser.add_argument('--keypoint-model', type=str, default='models/BEST2.pt', required=True,
                        help='Path del modello per rilevamento keypoints (default: models/BEST2.pt)')
    parser.add_argument('--player-model', type=str, default='models/PlayerDet.pt', required=True,
                        help='Path del modello per rilevamento giocatori (default: models/PlayerDet.pt)')
    
    parser.add_argument('--keypoint-stub', type=str, default='stubs/court_key_points_stub.pkl',
                        help='Path dello stub per keypoints (default: stubs/court_key_points_stub.pkl)')
    parser.add_argument('--player-stub', type=str, default='stubs/players_positions_stub.pkl',
                        help='Path dello stub per posizioni giocatori (default: stubs/players_positions_stub.pkl)')
    parser.add_argument('--no-stub', action='store_true',
                        help='Disabilita lettura da stub e ricalcola tutto')
    
    parser.add_argument('--court-image', type=str, default='./images/basketball_court.png',
                        help='Path immagine campo tattico (default: ./images/basketball_court.png)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Abilita modalità debug con visualizzazione frame')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Read Video
    video_frames = read_video(args.video)
    if video_frames == []:
        print("Error: Could not read video file.")
        return

    ## Initialize Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector(args.keypoint_model)
    
    ## Run KeyPoint Extractor
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
        video_frames,
        read_from_stub=not args.no_stub,
        stub_path=args.keypoint_stub, 
        label=args.video
    )
    print("Keypoint detection completed.")
    
    player_ball_detector = PlayerBallDetector(args.player_model, yolo=True)
    
    players_positions_per_frame, ball_positions_per_frame = player_ball_detector.getBallPlayersPositions(
        video_frames,
        read_from_stub=not args.no_stub,
        stub_path=args.player_stub,
        label=args.video
    )
    print("Player and ball detection completed.")
    
    tracker = PlayerTracker()
    ball_positions_per_frame = tracker.interpolate_ball_positions(ball_positions_per_frame)
    players_positions_per_frame = tracker.interpolate_player_positions(players_positions_per_frame)
    print("Interpolation of missing player and ball positions completed.")
    
    # Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path=args.court_image,
        video_width=video_frames[0].shape[1],
        video_height=video_frames[0].shape[0]
    )

    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    print("Validation of court keypoints completed.")
    
    if args.debug:
        with open('court_keypoints.txt', 'w') as f:
            for frame_kpts in court_keypoints_per_frame:
                if frame_kpts is None:
                    f.write("[]\n")
                    continue
                detected_indices = [i for i, pt in enumerate(frame_kpts) if pt[0] != 0 or pt[1] != 0]
                f.write(f"{detected_indices}\n")
    
    tactical_players_per_frame = tactical_view_converter.transform_players_to_tactical_view(
        court_keypoints_per_frame, 
        players_positions_per_frame
    )
    print("Transformation of player positions to tactical view completed.")
        
    drawWindow = DrawWindow("Output Video")
    tactical_court = cv2.imread(args.court_image)
    drawWindow.drawAllFrames(
        frames=video_frames, 
        small=tactical_court, 
        point_per_small=tactical_view_converter.getKeypointsForOpencv(), 
        points_per_frame=court_keypoints_per_frame,
        players_per_frame=players_positions_per_frame,
        tactical_players_per_frame=tactical_players_per_frame,
        ball_per_frame=ball_positions_per_frame,
        out_path=args.output_path,
        fps=args.fps,
        debug=args.debug
    )

if __name__ == '__main__':
    main()
