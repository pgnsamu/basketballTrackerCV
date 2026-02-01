import cv2
import numpy as np

def overlay_videos(video1_path, video2_path, output_path, opacity=0.5):
    """
    Sovrappone due video frame per frame.
    
    Args:
        video1_path: percorso primo video
        video2_path: percorso secondo video
        output_path: percorso video output
        opacity: opacità del secondo video (0.0-1.0)
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1:
            break
        
        if ret2:
            frame2 = cv2.resize(frame2, (width, height))
            frame = cv2.addWeighted(frame1, 1 - opacity, frame2, opacity, 0)
        else:
            frame = frame1
        
        out.write(frame)
    
    cap1.release()
    cap2.release()
    out.release()
    print(f"Video salvato in: {output_path}")

# Utilizzo
overlay_videos("video1.mp4", "video2.mp4", "output.mp4", opacity=0.5)


if __name__ == "__main__":
    overlay_videos("../outputVideo/output_video5validated3.1.mp4", "../outputVideo/output_video5validated3.mp4", "../outputVideo/output_sovrapposto.mp4", opacity=0.5)