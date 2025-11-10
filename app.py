from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io
import base64
from PIL import Image

app = Flask(__name__)
model = load_model("model.h5")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Label-to-emotion mapping
emotion_map = {
    "label_0": "Angry",
    "label_1": "Disgust",
    "label_2": "Fear",
    "label_3": "Happy",
    "label_4": "Neutral",
    "label_5": "Sad",
    "label_6": "Surprise",
    "label_7": "Calm",
    "label_8": "Content",
    "label_9": "Confused",
    "label_10": "Bored",
    "label_11": "Excited",
    "label_12": "Frustrated",
    "label_13": "Tired",
    "label_14": "Relaxed",
    "label_15": "Anxious",
    "label_16": "Pleased",
    "label_17": "Disappointed",
    "label_18": "Curious"
}


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    img_path = None

    # Handle webcam (base64)
    if request.form.get("image_data"):
        image_data = request.form["image_data"].split(",")[1]
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img_path = os.path.join(UPLOAD_FOLDER, "webcam_image.jpg")
        img.save(img_path)

    # Handle uploaded image
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No image uploaded"})
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)
    else:
        return jsonify({"error": "No image provided"})

    # Predict
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)[0]
    top_index = np.argmax(preds)
    confidence = float(preds[top_index]) * 100

    top_label = f"label_{top_index}"
    emotion = emotion_map.get(top_label, top_label)

    # ✅ Match what frontend expects
    return jsonify({
        "emotion": emotion,
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
