from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from flask_cors import CORS
from pathlib import Path
import cv2
import sys

# Get the directory where this file is located
BASE_DIR = Path(__file__).resolve().parent

print("="*60)
print("Plant Disease Recognition App - Starting...")
print("="*60)
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Base directory: {BASE_DIR}")

# List files
print("\nFiles in base directory:")
for item in BASE_DIR.iterdir():
    if item.is_file():
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  📄 {item.name} ({size_mb:.2f} MB)")

app = Flask(__name__)
CORS(app)

# Load model - look for .h5 file
model = None
model_path = BASE_DIR / 'plant_disease_model.h5'

print(f"\nLooking for model at: {model_path}")

if model_path.exists():
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model file found: {model_path.name} ({file_size_mb:.2f} MB)")
    
    try:
        print("Loading model...")
        model = tf.keras.models.load_model(str(model_path))
        print("✓ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Test with dummy input
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

class_labels = ['Healthy', 'Powdery', 'Rust']

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_path': str(model_path) if model_path.exists() else 'Not found',
        'model_file_size_mb': model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
    })

def generate_gradcam(model, img_array, class_idx, save_path):
    try:
        # Get last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            return False

        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        heatmap = np.uint8(255 * heatmap)
        # Apply colormap for better visualization
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        img = img_array[0].numpy()
        img = np.uint8(255 * img)
        heatmap_colored = cv2.resize(heatmap_colored, (img.shape[1], img.shape[0]))
        # Overlay heatmap on image
        superimposed_img = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)
        plt.figure(figsize=(4, 4))
        plt.axis('off')
        plt.imshow(superimposed_img)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True
    except Exception as e:
        print(f"GradCAM generation error: {e}")
        return False

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
        img = image.load_img(str(img_path), target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(tf.argmax(predictions[0]))
        predicted_class = class_labels[predicted_class_idx]
        confidence = float(tf.reduce_max(predictions[0]) * 100)

        class_breakdown = [
            {'label': class_labels[i], 'confidence': float(predictions[0][i] * 100)}
            for i in range(len(class_labels))
        ]

        gradcam_image_url = None
        try:
            gradcam_path = static_dir / 'gradcam_image.jpg'
            success = generate_gradcam(model, img_array, predicted_class_idx, str(gradcam_path))
            if success:
                gradcam_image_url = request.host_url.rstrip('/') + '/static/gradcam_image.jpg'
        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            gradcam_image_url = None

        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_breakdown': class_breakdown,
            'image_url': '/static/uploaded_image.jpg'
        }
        if gradcam_image_url is not None:
            response['gradcam_image_url'] = gradcam_image_url
        return jsonify(response)
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