import subprocess
import os
import numpy as np
import pandas as pd
import tempfile
from sklearn.preprocessing import StandardScaler
import shutil

def run_openface(video_path, openface_path="C:/OpenFace/OpenFace_2.2.0_win_x64/FeatureExtraction.exe"):
    output_dir = tempfile.mkdtemp()

    try:
        subprocess.run([openface_path, "-f", video_path, "-out_dir", output_dir], check=True)

        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV output found from OpenFace in: " + output_dir)

        csv_path = os.path.join(output_dir, csv_files[0])
        df = pd.read_csv(csv_path, encoding='ISO-8859-1', on_bad_lines='skip')
        facial_features = df.iloc[:, 4:136].astype(np.float32)

        scaler = StandardScaler()
        return scaler.fit_transform(facial_features)[-50:]

    finally:
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        try:
            os.rmdir(output_dir)
        except Exception as e:
            print(f"Failed to delete output directory. Reason: {e}")
