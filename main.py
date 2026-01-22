import os
import cv2
import numpy as np
from utils import read_video, save_video
from detectors.keypoint_detector import CourtKeypointDetector
from tactical_view_converter.tactical_view_converter import TacticalViewConverter
from homography.homography import Homography
from drawers.drawWindow import DrawWindow
from detectors.player_detector import PlayerDetector

DEBUG = False


def main():
    
    # Read Video
    video_frames = read_video('input_video/video_1.mp4')

    ## Initialize Keypoint Detector
    court_keypoint_detector = CourtKeypointDetector('models/BEST2.pt')
    
    ## Run KeyPoint Extractor
    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                    read_from_stub=True,
                                                                    stub_path='stubs/court_key_points_stub.pkl'
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
    
    player_detector = PlayerDetector('models/bestEMA.pth', optimize=False)
    
    players_positions_per_frame, ball_positions_per_frame = player_detector.getBallPlayersPositions(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/players_positions_stub.pkl'
    )
    
    '''
    annotated_frame = video_frames[0].copy()
    for player in players_positions_per_frame[0]:
        x1, y1, x2, y2 = player.as_int_tuple()
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated_frame, (int(player.center[0]), int(player.center[1])), 5, (0, 0, 255), -1)

    cv2.imshow("Players Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    
    # Tactical View
    tactical_view_converter = TacticalViewConverter(
        court_image_path="./images/basketball_court.png"
    )

    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(court_keypoints_per_frame)
    
    if DEBUG:
        for frame_idx, frame in enumerate(video_frames):
            homography = Homography(
                source_points=tactical_view_converter.getKeypointsForOpencv(),
                destination_points=court_keypoints_per_frame[frame_idx]
            )
            # DEBUG display every 10 frames
            # Problems on 40, 50 (few kpts), 60, 70 (only 3 kpts detected),
            #if frame_idx % 10 == 0:
            if frame_idx in [40,60]:    
                print(f"Calculated homography for frame {frame_idx}")
                drawWindow = DrawWindow(str(frame_idx), homography)
                frameSpec = video_frames[frame_idx].copy()
                frameSpec = drawWindow.drawPointsOnFrame(frameSpec, court_keypoints_per_frame[frame_idx])
                
                #drawWindow.realtimeDisplaying(frameSpec, court_keypoints_per_frame[frame_idx])
                frameImg = cv2.imread("images/basketball_court.png")
                frameImg = drawWindow.drawPointsOnFrame(frameImg, tactical_view_converter.getKeypointsForOpencv())
                #drawWindow.realtimeDisplaying(frameImg, tactical_view_converter.getKeypointsForOpencv())
                frame = drawWindow.composeFrame(frameSpec, frameImg, pos=(10,10), scale=0.3)
                drawWindow.realtimeDisplaying(frame)
        
    drawWindow = DrawWindow("Output Video")
    tactical_court = cv2.imread("images/basketball_court.png")
    output_video_frames = drawWindow.drawAllFrames(
        frames=video_frames,
        small=tactical_court,
        point_per_small=tactical_view_converter.getKeypointsForOpencv(),
        players_boxes_per_frame=players_positions_per_frame,
        points_per_frame=court_keypoints_per_frame
    )

    ## Draw KeyPoints
    #output_video_frames = court_keypoint_drawer.draw(video_frames, court_keypoints_per_frame, cv=True)
    '''
    ## Draw Frame Number
    output_video_frames = frame_number_drawer.draw(output_video_frames)
    '''
    # Save video
    save_video(output_video_frames, 'outputVideo/output_video5validated.mp4')

if __name__ == '__main__':
    main()
    