from flask import Flask, request, render_template, jsonify, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS
import sys
import traceback
from pathlib import Path

# Get the directory where this file is located
BASE_DIR = Path(__file__).resolve().parent

print("="*60)
print("Plant Disease Recognition App - Starting...")
print("="*60)
print(f"Base directory: {BASE_DIR}")
print(f"Current working directory: {os.getcwd()}")
print(f"TensorFlow version: {tf.__version__}")

# List all files in base directory
print("\nFiles in base directory:")
for item in BASE_DIR.iterdir():
    if item.is_file():
        file_size = item.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"  📄 {item.name} ({file_size:.2f} MB)")
    elif item.is_dir():
        print(f"  📁 {item.name}/")

# Look for model files specifically
print("\nLooking for model files:")
keras_files = list(BASE_DIR.glob("*.keras"))
h5_files = list(BASE_DIR.glob("*.h5"))
print(f"  .keras files: {[f.name for f in keras_files]}")
print(f"  .h5 files: {[f.name for f in h5_files]}")

# Check model file size and integrity
if keras_files:
    model_file = keras_files[0]
    file_size = model_file.stat().st_size
    print(f"\nModel file details:")
    print(f"  Path: {model_file}")
    print(f"  Size: {file_size / (1024*1024):.2f} MB")
    
    # Try to read first few bytes to verify it's not corrupted
    try:
        with open(model_file, 'rb') as f:
            header = f.read(100)
            print(f"  File header (first 100 bytes): {header[:50]}...")
    except Exception as e:
        print(f"  Error reading file: {e}")
print("="*60)

app = Flask(__name__)

# Configure CORS
cors_allowed_origins = os.environ.get('CORS_ALLOWED_ORIGINS', '*')
if cors_allowed_origins and cors_allowed_origins != '*':
    origins = [origin.strip() for origin in cors_allowed_origins.split(',')]
    CORS(app, origins=origins)
else:
    CORS(app)

# Store last prediction info
last_prediction = {
    'img_path': None,
    'predicted_class': None,
    'confidence': None
}

# Load model with detailed error handling
def load_model():
    """Load the model with detailed error logging"""
    possible_model_names = [
        'plant_disease_model.keras',
        'plant_disease_model.h5',
        'model.keras',
        'model.h5'
    ]
    
    # Search in BASE_DIR
    search_dirs = [BASE_DIR, BASE_DIR / 'models', BASE_DIR.parent]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        print(f"\nSearching in: {search_dir}")
        for model_name in possible_model_names:
            model_path = search_dir / model_name
            if model_path.exists():
                print(f"✓ Found model at: {model_path}")
                print(f"  File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
                
                try:
                    print(f"  Attempting to load model...")
                    # Try loading with different methods
                    try:
                        model = tf.keras.models.load_model(str(model_path))
                        print(f"✓ Model loaded successfully!")
                        return model, str(model_path)
                    except Exception as e1:
                        print(f"  Standard load failed: {e1}")
                        
                        # Try with custom objects if needed
                        try:
                            print(f"  Trying with custom_objects...")
                            model = tf.keras.models.load_model(
                                str(model_path),
                                custom_objects={'tf': tf}
                            )
                            print(f"✓ Model loaded successfully with custom_objects!")
                            return model, str(model_path)
                        except Exception as e2:
                            print(f"  Custom objects load failed: {e2}")
                            
                            # Try with compile=False
                            try:
                                print(f"  Trying with compile=False...")
                                model = tf.keras.models.load_model(
                                    str(model_path),
                                    compile=False
                                )
                                print(f"✓ Model loaded successfully with compile=False!")
                                return model, str(model_path)
                            except Exception as e3:
                                print(f"  Compile=False load failed: {e3}")
                                print(f"\n  Full traceback:")
                                traceback.print_exc()
                                continue
                                
                except Exception as e:
                    print(f"✗ Error loading model: {e}")
                    traceback.print_exc()
                    continue
    
    print("\n✗ No model found or model failed to load in any location")
    return None, None

# Load the model
model, model_path = load_model()

if model:
    print("\n✓ Model loaded successfully!")
    print("\nModel summary:")
    try:
        model.summary()
    except:
        print("  Could not print model summary")
    
    # Test with a dummy input to verify it works
    try:
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        dummy_output = model.predict(dummy_input, verbose=0)
        print(f"\n✓ Model test prediction successful! Output shape: {dummy_output.shape}")
    except Exception as e:
        print(f"\n⚠️ Model test prediction failed: {e}")
else:
    print("\n⚠️  WARNING: Model not loaded. Predictions will not work.")
    print("Check the logs above for detailed error information.")

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
        traceback.print_exc()
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
    
    # Predict
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs for details.'}), 500
    
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
        traceback.print_exc()
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

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_path': model_path if model_path else 'Not found',
        'base_directory': str(BASE_DIR),
        'tensorflow_version': tf.__version__
    })

@app.route('/model-status', methods=['GET'])
def model_status():
    """Debug endpoint to check model status"""
    # List all files in base directory with sizes
    files = []
    for item in BASE_DIR.iterdir():
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            files.append(f"{item.name} ({size_mb:.2f} MB)")
    
    # Find all model files
    model_files = []
    for ext in ['*.keras', '*.h5', '*.hdf5']:
        for f in BASE_DIR.glob(ext):
            size_mb = f.stat().st_size / (1024 * 1024)
            model_files.append(f"{f.name} ({size_mb:.2f} MB)")
    
    return jsonify({
        'model_loaded': model is not None,
        'model_path': model_path,
        'base_directory': str(BASE_DIR),
        'files_in_directory': files[:30],
        'model_files_found': model_files,
        'tensorflow_version': tf.__version__,
        'python_version': sys.version
    })

@app.route('/visualization', methods=['GET'])
def visualization():
    """Return GradCAM visualization if available"""
    possible_paths = [
        BASE_DIR / 'gradcam.png',
        BASE_DIR / 'static' / 'gradcam.png',
    ]
    
    for path in possible_paths:
        if path.exists():
            return send_file(str(path), mimetype='image/png')
    
    return jsonify({'error': 'Visualization not available'}), 404

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

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Flask App...")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Model loaded: {model is not None}")
    if model:
        print(f"Model path: {model_path}")
    print("="*60)
    print("\nAccess the app at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("Model status: http://localhost:5000/model-status")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)