from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS
from pathlib import Path
import sys

# Get the directory where this file is located
BASE_DIR = Path(__file__).resolve().parent

print("="*60)
print("Plant Disease Recognition App - Starting...")
print("="*60)
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Base directory: {BASE_DIR}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in BASE_DIR: {[f.name for f in BASE_DIR.iterdir() if f.is_file()]}")
print(f"Directories in BASE_DIR: {[d.name for d in BASE_DIR.iterdir() if d.is_dir()]}")

# Look for model files
print("\n" + "="*60)
print("Looking for model files:")
keras_files = list(BASE_DIR.glob("*.keras"))
h5_files = list(BASE_DIR.glob("*.h5"))
print(f".keras files found: {[f.name for f in keras_files]}")
print(f".h5 files found: {[f.name for f in h5_files]}")

if keras_files:
    for kf in keras_files:
        size_mb = kf.stat().st_size / (1024 * 1024)
        print(f"  - {kf.name}: {size_mb:.2f} MB")
else:
    print("No .keras files found!")
print("="*60)

app = Flask(__name__)
CORS(app)

# Load model
model = None
model_path = BASE_DIR / 'plant_disease_model.keras'

print(f"\nAttempting to load model from: {model_path}")

if model_path.exists():
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model file found: {model_path.name} ({file_size_mb:.2f} MB)")
    
    try:
        print("Loading model...")
        model = tf.keras.models.load_model(str(model_path))
        print("✓ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Number of layers: {len(model.layers)}")
        
        # Test model with dummy input
        import numpy as np
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        dummy_output = model.predict(dummy_input, verbose=0)
        print(f"✓ Model test successful! Output shape: {dummy_output.shape}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
else:
    print(f"✗ Model file NOT found at {model_path}")
    print(f"Available files in {BASE_DIR}:")
    for f in BASE_DIR.iterdir():
        if f.is_file():
            print(f"  - {f.name}")

class_labels = ['Healthy', 'Powdery', 'Rust']

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_path': str(model_path) if model_path.exists() else 'Not found',
        'files_in_dir': [f.name for f in BASE_DIR.iterdir() if f.is_file()][:10]
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
    print("\nStarting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)