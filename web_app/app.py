from flask import Flask, request, render_template, jsonify, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS
import sys
from pathlib import Path

# Get the directory where this file is located
BASE_DIR = Path(__file__).resolve().parent

print(f"Python executable: {sys.executable}")
print(f"Application base directory: {BASE_DIR}")
print(f"Current working directory: {os.getcwd()}")

app = Flask(__name__)

# Configure CORS
cors_allowed_origins = os.environ.get('CORS_ALLOWED_ORIGINS', '*')
if cors_allowed_origins and cors_allowed_origins != '*':
    origins = [origin.strip() for origin in cors_allowed_origins.split(',')]
    CORS(app, origins=origins)
else:
    CORS(app)

# Store last prediction info for visualization endpoint
last_prediction = {
    'img_path': None,
    'predicted_class': None,
    'confidence': None
}

# Model path configuration - works for both local and Docker
def find_model():
    """Try to find model in multiple locations"""
    possible_paths = [
        BASE_DIR / 'plant_disease_model.keras',           # Same directory as app.py
        BASE_DIR / 'models' / 'plant_disease_model.keras', # In models subdirectory
        BASE_DIR.parent / 'plant_disease_model.keras',     # One level up (original structure)
        Path('/app/plant_disease_model.keras'),            # Docker container root
        Path('/app/models/plant_disease_model.keras'),     # Docker with models folder
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found model at: {path}")
            return str(path)
    
    print("✗ Model not found in any expected location")
    print("Searched paths:")
    for path in possible_paths:
        print(f"  - {path}")
    return None

# Load model with error handling
model_path = find_model()
if model_path:
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
        # Print model architecture summary
        model.summary()
    except Exception as e:
        print(f"✗ Error loading model from {model_path}: {e}")
        model = None
else:
    print("✗ No model file found. Please ensure plant_disease_model.keras is in the correct location.")
    model = None

class_labels = ['Healthy', 'Powdery', 'Rust']

def predict_image(img_path):
    """Predict disease from image path"""
    if model is None:
        return "Model not loaded", 0
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)
        predicted_class = class_labels[tf.argmax(predictions[0])]
        confidence = float(tf.reduce_max(predictions[0]) * 100)
        return predicted_class, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No image file provided'}), 400
    
    # Create static directory if it doesn't exist
    static_dir = BASE_DIR / 'static'
    static_dir.mkdir(exist_ok=True)
    
    img_path = static_dir / 'uploaded_image.jpg'
    file.save(str(img_path))
    
    # Predict and get probabilities for all classes
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        img = image.load_img(str(img_path), target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)
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
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    
    # Store for visualization endpoint
    last_prediction['img_path'] = str(img_path)
    last_prediction['predicted_class'] = predicted_class
    last_prediction['confidence'] = confidence
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence,
        'image_url': '/static/uploaded_image.jpg',
        'class_breakdown': class_breakdown
    })

@app.route('/visualization', methods=['GET'])
def visualization():
    """Return GradCAM visualization if available"""
    possible_paths = [
        BASE_DIR / 'gradcam.png',
        BASE_DIR / 'static' / 'gradcam.png',
        BASE_DIR.parent / 'gradcam.png',
    ]
    
    for path in possible_paths:
        if path.exists():
            return send_file(str(path), mimetype='image/png')
    
    return jsonify({'error': 'Visualization not available'}), 404

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_path': model_path if model_path else 'None',
        'base_directory': str(BASE_DIR)
    })

@app.route('/model-status', methods=['GET'])
def model_status():
    """Debug endpoint to check model status and file locations"""
    files_in_base = [f.name for f in BASE_DIR.iterdir() if f.is_file()]
    files_in_parent = [f.name for f in BASE_DIR.parent.iterdir() if f.is_file()] if BASE_DIR.parent.exists() else []
    
    # Check for model files
    model_files = []
    for ext in ['*.keras', '*.h5', '*.hdf5']:
        model_files.extend(list(BASE_DIR.glob(ext)))
        model_files.extend(list(BASE_DIR.parent.glob(ext)))
    
    return jsonify({
        'model_loaded': model is not None,
        'model_path': model_path if model_path else 'Not found',
        'base_directory': str(BASE_DIR),
        'current_working_directory': os.getcwd(),
        'files_in_app_directory': files_in_base[:20],  # First 20 files
        'model_files_found': [str(f) for f in model_files],
        'python_version': sys.version,
        'tensorflow_version': tf.__version__
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main web interface"""
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            static_dir = BASE_DIR / 'static'
            static_dir.mkdir(exist_ok=True)
            img_path = static_dir / 'uploaded_image.jpg'
            file.save(str(img_path))
            predicted_class, confidence = predict_image(str(img_path))
            return render_template('index.html', 
                                 predicted_class=predicted_class, 
                                 confidence=confidence, 
                                 image_path=str(img_path))
    return render_template('index.html')

# For local development
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Plant Disease Recognition App")
    print("="*50)
    print(f"Base directory: {BASE_DIR}")
    print(f"Model loaded: {model is not None}")
    if model:
        print(f"Model path: {model_path}")
    print("="*50)
    print("\nAccess the app at: http://localhost:5000")
    print("API endpoint: http://localhost:5000/predict")
    print("Health check: http://localhost:5000/health")
    print("\n")
    app.run(debug=True, host='0.0.0.0', port=5000)