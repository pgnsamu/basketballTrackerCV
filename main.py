import os
import cv2
import numpy as np
from utils import read_video, save_video
from detectors.keypoint_detector import CourtKeypointDetector
from drawers.courtKeypointDrawer import CourtKeypointDrawer
from tactical_view_converter.tactical_view_converter import TacticalViewConverter
from homography.homography import Homography
from drawers.drawWindow import DrawWindow

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
            if frame_idx % 10 == 0:
                print(f"Calculated homography for frame {frame_idx}")
                drawWindow = DrawWindow("Frame", homography)
                frameSpec = video_frames[frame_idx].copy()
                frameSpec = drawWindow.drawOnFrame(frameSpec, court_keypoints_per_frame[frame_idx])
                
                #drawWindow.realtimeDisplaying(frameSpec, court_keypoints_per_frame[frame_idx])
                frameImg = cv2.imread("images/basketball_court.png")
                frameImg = drawWindow.drawOnFrame(frameImg, tactical_view_converter.getKeypointsForOpencv())
                #drawWindow.realtimeDisplaying(frameImg, tactical_view_converter.getKeypointsForOpencv())
                frame = drawWindow.composeFrame(frameSpec, frameImg, pos=(10,10), scale=0.3)
                drawWindow.realtimeDisplaying(frame)
        
    drawWindow = DrawWindow("Output Video")
    tactical_court = cv2.imread("images/basketball_court.png")
    output_video_frames = drawWindow.drawAllFrames(
        frames=video_frames,
        small=tactical_court,
        point_per_small=tactical_view_converter.getKeypointsForOpencv(),
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
    