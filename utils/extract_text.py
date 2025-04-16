import speech_recognition as sr
import numpy as np
import gensim.downloader as api
from pydub import AudioSegment

glove = api.load("glove-wiki-gigaword-300")
recognizer = sr.Recognizer()

def get_text_embedding(audio_path):
    """Convert speech to text then to GloVe embeddings"""
    if not audio_path.endswith('.wav'):
        sound = AudioSegment.from_file(audio_path)
        audio_path = audio_path.replace('.mp4', '.wav')
        sound.export(audio_path, format="wav")
    
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = ""
    
    words = text.lower().split()
    vectors = [glove[word] for word in words if word in glove]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)