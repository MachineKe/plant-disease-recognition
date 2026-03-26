import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = tf.keras.models.load_model('../plant_disease_model_fixed2.keras')
print("✅ Model loaded")

# Test with a dummy image
img_path = 'test.jpg'
if not os.path.exists(img_path):
    # Create a dummy test image
    import matplotlib.pyplot as plt
    dummy_img = np.random.rand(224, 224, 3)
    plt.imsave(img_path, dummy_img)
    print("Created dummy test image")

# Load and preprocess
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array, verbose=0)
print(f"✅ Prediction shape: {pred.shape}")
print(f"Prediction values: {pred[0]}")

class_labels = ['Healthy', 'Powdery', 'Rust']
pred_class = class_labels[np.argmax(pred[0])]
conf = float(np.max(pred[0])) * 100
print(f"Predicted: {pred_class} with {conf:.2f}% confidence")
