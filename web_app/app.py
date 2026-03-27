from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

print("="*50)
print("Starting Plant Disease Recognition App")
print("="*50)
print(f"Base directory: {BASE_DIR}")

app = Flask(__name__)
CORS(app)

# Load model
model_path = BASE_DIR / 'plant_disease_model.keras'
print(f"Looking for model at: {model_path}")

if model_path.exists():
    try:
        model = tf.keras.models.load_model(str(model_path))
        print("✓ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = None
else:
    print(f"✗ Model not found at {model_path}")
    model = None

class_labels = ['Healthy', 'Powdery', 'Rust']

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['image']
    if not file:
        return jsonify({'error': 'No image file'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Save uploaded image
    static_dir = BASE_DIR / 'static'
    static_dir.mkdir(exist_ok=True)
    img_path = static_dir / 'uploaded_image.jpg'
    file.save(str(img_path))
    
    try:
        # Load and preprocess image
        img = image.load_img(str(img_path), target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class = class_labels[tf.argmax(predictions[0])]
        confidence = float(tf.reduce_max(predictions[0]) * 100)
        
        # Class breakdown
        class_breakdown = [
            {'label': class_labels[i], 'confidence': float(predictions[0][i] * 100)}
            for i in range(len(class_labels))
        ]
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_breakdown': class_breakdown,
            'image_url': '/static/uploaded_image.jpg'
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            static_dir = BASE_DIR / 'static'
            static_dir.mkdir(exist_ok=True)
            img_path = static_dir / 'uploaded_image.jpg'
            file.save(str(img_path))
            
            # Make prediction
            if model:
                img = image.load_img(str(img_path), target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = tf.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array, verbose=0)
                predicted_class = class_labels[tf.argmax(predictions[0])]
                confidence = float(tf.reduce_max(predictions[0]) * 100)
                
                return render_template('index.html',
                                     predicted_class=predicted_class,
                                     confidence=confidence,
                                     image_path=str(img_path))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)