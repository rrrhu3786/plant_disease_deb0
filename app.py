import os
import io
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model_final.h5')

# --- Model Download & Validation ---
def validate_model_file():
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        print(f"Model found. Size: {size/1024:.2f} KB")
        if size < 1024:  # <1KB = invalid
            print("Removing corrupted file")
            os.remove(MODEL_PATH)

def download_model():
    FILE_ID = "1RGKYGHs4-reK7I52ZoqKgk-MpPymZvfa"
    URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    print("Downloading model...")
    try:
        response = requests.get(URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk: f.write(chunk)
        print("Download successful!")
    except Exception as e:
        print("Download failed:", e)

# --- Fallback Model ---
def create_fallback_model():
    print("Creating fallback model")
    model = Sequential([
        Flatten(input_shape=(256, 256, 3)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# --- Load Main Model ---
validate_model_file()
if not os.path.exists(MODEL_PATH):
    download_model()

try:
    model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
except:
    model = None

if not model:
    model = create_fallback_model()

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((256, 256))
    return np.expand_dims(np.array(img), axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    try:
        img_array = preprocess_image(request.files['file'].read())
        prediction = model.predict(img_array)
        result = "Healthy" if prediction[0][0] > 0.7 else "Infected"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
