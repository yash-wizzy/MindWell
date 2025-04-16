import os
import librosa
import numpy as np
import tempfile
import moviepy.editor as mp

def extract_mfcc_from_video(video_path, sr=22050, n_mfcc=13):
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
   
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_path, logger=None)
    video.reader.close()
    video.audio.reader.close_proc()
   
    y, _ = librosa.load(temp_audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
   
    os.remove(temp_audio_path)

    return mfcc.T

