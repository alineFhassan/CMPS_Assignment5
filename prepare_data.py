# prepare_data.py

import os
import json
import cv2
import whisper
from tqdm import tqdm
import yt_dlp

def download_video(youtube_url, output_path="video.mp4"):
    ydl_opts = {'outtmpl': output_path}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def transcribe_audio(video_path):
    model = whisper.load_model("small")  
    result = model.transcribe(video_path)
    return result['segments']

def save_transcript(segments, output_path="data/transcript.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)

def extract_frames(video_path, frame_folder="data/frames", every_n_seconds=5):
    os.makedirs(frame_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * every_n_seconds)

    count = 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(frame_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        count += 1
    cap.release()

if __name__ == "__main__":
    YOUTUBE_URL = "https://www.youtube.com/watch?v=dARr3lGKwk8"
    
    print("Downloading video...")
    download_video(YOUTUBE_URL)

    print("Transcribing audio...")
    segments = transcribe_audio("video.mp4")
    save_transcript(segments)

    print("Extracting frames...")
    extract_frames("video.mp4")

    print("Done!")
