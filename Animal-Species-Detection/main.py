from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import base64
import os

app = Flask(__name__)
mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()

# Create a folder to temporarily store uploaded images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def detect_animal(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    preprocessed_frame = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(resized_frame, axis=0))
    predictions = mobile.predict(preprocessed_frame)
    results = imagenet_utils.decode_predictions(predictions)
    label = results[0][0][1]
    return label

def process_uploaded_image(file):
    nparr = np.frombuffer(file.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detected_label = detect_animal(img_np)
    return detected_label, img_np

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' in request.files:
        uploaded_file = request.files['file']
        detected_label, img_np = process_uploaded_image(uploaded_file)
        # Save uploaded image temporarily for display (optional)
        image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
        cv2.imwrite(image_path, cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        return jsonify({'label': detected_label, 'image_path': image_path})

    elif request.data:
        # Process image captured from webcam (as blob data)
        blob_data = request.get_data()
        nparr = np.frombuffer(blob_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detected_label = detect_animal(img_np)
        return jsonify({'label': detected_label})

    else:
        return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run(debug=True)
