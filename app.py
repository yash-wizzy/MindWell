from flask import Flask, request, render_template, jsonify
import os
from utils.extract_audio import extract_audio_from_video, extract_mfcc  
from utils.extract_text import get_text_embedding  
from utils.extract_visual import run_openface
import torch
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
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_audio.wav")
        extract_audio_from_video(video_path, audio_path)        
        
        audio_feat = extract_mfcc(audio_path)
        transcribed_text = get_text_embedding(audio_path) 
        visual_feat = run_openface(video_path)
      
        audio_tensor = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0)
        text_tensor = torch.tensor(transcribed_text, dtype=torch.float32).unsqueeze(0)
        visual_tensor = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = model(audio_tensor, text_tensor, visual_tensor).item()

        os.remove(audio_path)
        
        return jsonify({
            'depression_score': prediction
        })

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)