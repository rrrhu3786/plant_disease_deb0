from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown
import gc

# 🔒 تقييد استهلاك الذاكرة على GPU (إن وُجد)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # أو لتحديد كمية معينة بالميغابايت (مثال: 1024MB = 1GB)
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        # )
    except RuntimeError as e:
        print("GPU memory limitation error:", e)

app = Flask(__name__)

# إعداد المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'plant_disease_model_final.h5')

# تحميل النموذج من Google Drive إذا لم يكن موجودًا
if not os.path.exists(MODEL_PATH):
    # ضع هنا ID الخاص بملف النموذج في Google Drive
    FILE_ID = "1RGKYGHs4-reK7I52ZoqKgk-MpPymZvfa"  
    GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# تحميل النموذج
model = load_model(MODEL_PATH)
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((256, 256))
    img_array = np.expand_dims(np.array(img), axis=0)
    return img_array

@app.route('/')
def index():
    print("Current working directory:", os.getcwd())
    print("Templates folder exists:", os.path.exists(os.path.join(os.getcwd(), 'templates', 'index.html')))
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    img_array = preprocess_image(file.read())
    prediction = model.predict(img_array)

    result = "Infected" if prediction[0][0] > 0.7 else "Healthy"
    return jsonify({"result": result})


         


