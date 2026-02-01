import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from utils.video_utils import read_video, save_video 
from detectors.keypoint_detector import CourtKeypointDetector
from tactical_view_converter.tactical_view_converter import TacticalViewConverter
from detectors.player_ball_detector import PlayerBallDetector
from detectors.player_tracker import PlayerTracker 
from drawers.drawWindow import DrawWindow

def main():
    print("=== BASKETBALL TRACKER AI (FINAL v2) ===")
    
    # 1. Caricamento
    input_path = 'input_video/video_2.mp4'
    video_frames, fps = read_video(input_path) 
    if not video_frames: return

    # 2. Detection
    print("Detecting Court Keypoints...")
    court_detector = CourtKeypointDetector('models/BEST2.pt')
    court_keypoints = court_detector.get_court_keypoints(
        video_frames, 
        read_from_stub=True, 
        stub_path='stubs/court_key_points_stub_copia.pkl' 
    )

    print("Detecting Players & Ball...")
    player_detector = PlayerBallDetector('models/best_player.pt', yolo=True)
    players_pos, ball_pos = player_detector.getBallPlayersPositions(
        video_frames,
        read_from_stub=True, 
        stub_path='stubs/players_positions_stub.pkl'
    )

    # 3. Interpolazione
    print("Interpolating Data...")
    tracker = PlayerTracker()
    ball_pos = tracker.interpolate_ball_positions(ball_pos)
    players_pos = tracker.interpolate_player_positions(players_pos)

    # 4. Vista Tattica (ORA INCLUDE LA PALLA)
    print("Converting to Tactical View...")
    tactical_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")
    
    # NUOVO: Ritorna anche la palla tattica
    tactical_players, tactical_ball = tactical_converter.transform_players_to_tactical_view(
        court_keypoints, players_pos, ball_list=ball_pos
    )

    # 5. Rendering
    print("Rendering...")
    drawer = DrawWindow("Output")
    output_frames = drawer.drawAllFrames(
        frames=video_frames,
        small=None,
        point_per_small=tactical_converter.getKeypointsForOpencv(),
        points_per_frame=court_keypoints,
        players_per_frame=players_pos,
        tactical_players_per_frame=tactical_players,
        ball_per_frame=ball_pos,
        tactical_ball_per_frame=tactical_ball # Passiamo la palla tattica
    )

    # 6. Salvataggio
    save_video(output_frames, 'outputVideo/output_final_v2.mp4', fps=fps) 
    print("Done!")

if __name__ == '__main__':
    main()