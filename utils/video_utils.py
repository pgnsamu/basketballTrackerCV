import cv2
import os

def read_video(video_path):
    """
    Read all frames from a video file into memory and return FPS.
    
    Returns:
        tuple: (list of frames, fps of the video)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS) # Leggiamo gli FPS originali
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps

def save_video(output_video_frames, output_video_path, fps=24):
    """
    Save a sequence of frames as a video file using the correct FPS.
    """
    if not output_video_frames:
        print("No frames to save.")
        return

    # If folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    # Usa 'mp4v' che è più compatibile di XVID per i browser/player moderni
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    height, width = output_video_frames[0].shape[:2]
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved at {output_video_path} with {fps} FPS")