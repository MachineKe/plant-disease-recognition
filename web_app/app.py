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

# Configure CORS without dotenv
cors_allowed_origins = os.environ.get('CORS_ALLOWED_ORIGINS', '*')
if cors_allowed_origins and cors_allowed_origins != '*':
    origins = [origin.strip() for origin in cors_allowed_origins.split(',')]
    CORS(app, origins=origins)
else:
    CORS(app)  # Allow all origins if not set

# Store last prediction info for visualization endpoint
last_prediction = {
    'img_path': None,
    'predicted_class': None,
    'confidence': None
}

# Load model with error handling
try:
    model = tf.keras.models.load_model('../plant_disease_model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class_labels = ['Healthy', 'Powdery', 'Rust']

def predict_image(img_path):
    if model is None:
        return "Model not loaded", 0
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = class_labels[tf.argmax(predictions[0])]
        confidence = tf.reduce_max(predictions[0]) * 100
        return predicted_class, confidence.numpy()
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    img_path = 'static/uploaded_image.jpg'
    file.save(img_path)
    
    # Predict and get probabilities for all classes
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = class_labels[tf.argmax(predictions[0])]
        confidence = float(tf.reduce_max(predictions[0]) * 100)
        class_breakdown = [
            {
                'label': class_labels[i],
                'confidence': float(predictions[0][i] * 100)
            }
            for i in range(len(class_labels))
        ]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'Error during prediction'}), 500
    
    # Store for visualization endpoint
    last_prediction['img_path'] = img_path
    last_prediction['predicted_class'] = predicted_class
    last_prediction['confidence'] = confidence
    
    # Optionally provide image URL
    image_url = '/static/uploaded_image.jpg'
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence,
        'image_url': image_url,
        'class_breakdown': class_breakdown
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
        file = request.files.get('image')
        if file and file.filename:
            os.makedirs('static', exist_ok=True)
            img_path = 'static/uploaded_image.jpg'
            file.save(img_path)
            predicted_class, confidence = predict_image(img_path)
            return render_template('index.html', 
                                 predicted_class=predicted_class, 
                                 confidence=confidence, 
                                 image_path=img_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)