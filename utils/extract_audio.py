import subprocess
import numpy as np
import librosa

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio track from video using ffmpeg"""
    command = f"ffmpeg -y -i {video_path} -ac 1 -ar 16000 {output_audio_path}"
    subprocess.run(command, shell=True, check=True, 
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_mfcc(audio_path, n_mfcc=39):
    """Extract MFCC features from audio file"""
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)