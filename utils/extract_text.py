import numpy as np
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-300")

def get_glove_embedding(text):
    words = text.lower().split()
    vectors = [glove[word] for word in words if word in glove]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)
