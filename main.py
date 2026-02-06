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

DEBUG = False


def main():
    
    video_name = "video_4.mp4"  
    # Read Video
    video_frames = read_video('input_video/'+video_name)
    if video_frames == []:
        print("Error: Could not read video file.")
        return

    ## Initialize Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector('models/BEST2.pt')
    
    ## Run KeyPoint Extractor
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                    read_from_stub=True,
                                                                    stub_path='stubs/court_key_points_stub_copia.pkl', 
                                                                    label=video_name
                                                                    )
    # [1, 18, 2]
    '''
    [ # 1
        [ #18
            [x,y], #2
            ...
        ]
    ]
    '''
    
    player_ball_detector = PlayerBallDetector('models/PlayerDet.pt', yolo=True)
    
    players_positions_per_frame, ball_positions_per_frame = player_ball_detector.getBallPlayersPositions(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/players_positions_stub.pkl',
        label=video_name
    )
    '''
    # using old model for comparison
    player_ball_detector2 = PlayerBallDetector('models/bestEMA.pth', yolo=False)
    
    players_positions_per_frame2, ball_positions_per_frame2 = player_ball_detector2.getBallPlayersPositions(
        video_frames,
        read_from_stub=False,
        stub_path='stubs/players_positions_stub_v2.pkl'
    )
    '''
    print("Interpolating Data...")
    tracker = PlayerTracker()
    ball_positions_per_frame = tracker.interpolate_ball_positions(ball_positions_per_frame)
    players_positions_per_frame = tracker.interpolate_player_positions(players_positions_per_frame)
    
    # Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png",
        video_width=video_frames[0].shape[1],
        video_height=video_frames[0].shape[0]
    )

    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    
    tactical_players_per_frame = tactical_view_converter.transform_players_to_tactical_view(court_keypoints_per_frame, players_positions_per_frame)
    
    
    if DEBUG:
        for frame_idx, frame in enumerate(video_frames):
            homography = Homography(
                source_points=tactical_view_converter.getKeypointsForOpencv(),
                destination_points=court_keypoints_per_frame[frame_idx]
            )
            # DEBUG display every 10 frames
            # Problems on 40, 50 (few kpts), 60, 70 (only 3 kpts detected),
            if frame_idx % 10 == 0:  
                print(f"Calculated homography for frame {frame_idx}")
                drawWindow = DrawWindow(str(frame_idx), homography)
                frameSpec = video_frames[frame_idx].copy()
                frameSpec = drawWindow.drawPointsOnFrame(frameSpec, court_keypoints_per_frame[frame_idx])
                
                drawWindow.realtimeDisplaying(frameSpec, court_keypoints_per_frame[frame_idx])
                frameImg = cv2.imread("images/basketball_court.png")
                frameImg = drawWindow.drawPointsOnFrame(frameImg, tactical_view_converter.getKeypointsForOpencv())
                drawWindow.realtimeDisplaying(frameImg, tactical_view_converter.getKeypointsForOpencv())
                frame = drawWindow.composeFrame(frameSpec, frameImg, pos=(10,10), scale=0.3)
                drawWindow.realtimeDisplaying(frame)
        
    drawWindow = DrawWindow("Output Video")
    tactical_court = cv2.imread("images/basketball_court.png")
    drawWindow.drawAllFrames(
        frames=video_frames, 
        small=tactical_court, 
        point_per_small=tactical_view_converter.getKeypointsForOpencv(), 
        points_per_frame=court_keypoints_per_frame,
        players_per_frame=players_positions_per_frame,
        tactical_players_per_frame=tactical_players_per_frame,
        ball_per_frame=ball_positions_per_frame,
        out_path="outputVideo/output_video5validated3_33_2.mp4",
        fps=30.0
    )

    ## Draw KeyPoints
    #output_video_frames = court_keypoint_drawer.draw(video_frames, court_keypoints_per_frame, cv=True)
    '''
    ## Draw Frame Number
    output_video_frames = frame_number_drawer.draw(output_video_frames)
    '''
    # Save video
    # save_video(output_video_frames, 'outputVideo/output_video5validated3_33.mp4')

if __name__ == '__main__':
    main()
    