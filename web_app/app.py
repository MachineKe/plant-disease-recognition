from flask import Flask, request, render_template, jsonify, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

from flask_cors import CORS
import sys
print(f"Python executable: {sys.executable}")
print(f"sys.path: {sys.path}")

app = Flask(__name__)
CORS(app)  # Allow all origins for development

# Store last prediction info for visualization endpoint
last_prediction = {
    'img_path': None,
    'predicted_class': None,
    'confidence': None
}

model = tf.keras.models.load_model('../plant_disease_model.keras')
class_labels = ['Healthy', 'Powdery', 'Rust']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_labels[tf.argmax(predictions[0])]
    confidence = tf.reduce_max(predictions[0]) * 100
    return predicted_class, confidence.numpy()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image file provided'}), 400
    img_path = 'static/uploaded_image.jpg'
    file.save(img_path)
    predicted_class, confidence = predict_image(img_path)
    # Store for visualization endpoint
    last_prediction['img_path'] = img_path
    last_prediction['predicted_class'] = predicted_class
    last_prediction['confidence'] = confidence
    # Optionally provide image URL
    image_url = '/static/uploaded_image.jpg'
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence,
        'image_url': image_url
    })

@app.route('/visualization', methods=['GET'])
def visualization():
    # Example: return gradcam.png if exists
    gradcam_path = '../gradcam.png'
    if os.path.exists(gradcam_path):
        return send_file(gradcam_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Visualization not available'}), 404

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = 'static/uploaded_image.jpg'
            file.save(img_path)
            predicted_class, confidence = predict_image(img_path)
            return render_template('index.html', predicted_class=predicted_class, confidence=confidence, image_path=img_path)
    return render_template('index.html')
