from flask import Flask, request, render_template, jsonify
import os
from utils.extract_audio import extract_mfcc_from_video
from utils.extract_text import get_glove_embedding
from utils.extract_visual import run_openface
import torch
import numpy as np
from model import DepressionModel, weights_init

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = DepressionModel()
model.apply(weights_init)
model.load_state_dict(torch.load("depression_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        text = request.form['text']

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)

        audio_feat = extract_mfcc_from_video(video_path)
        text_feat = get_glove_embedding(text)
        visual_feat = run_openface(video_path)
       
        audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0)   
        visual_tensor = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0)   
        text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0)


        with torch.no_grad():
           prediction = model(audio_tensor, text_tensor, visual_tensor).item()
        

        return jsonify({'depression_score': prediction})

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
