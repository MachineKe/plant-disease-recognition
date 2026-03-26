import tensorflow as tf
import zipfile
import json
import numpy as np
import os
import tempfile
import shutil

print("TensorFlow version:", tf.__version__)

# Step 1: Extract the model files
model_path = 'plant_disease_model.keras'
extract_dir = '/tmp/model_extracted'

print(f"Extracting {model_path} to {extract_dir}...")
with zipfile.ZipFile(model_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Step 2: Load the configuration
print("Loading model configuration...")
with open(os.path.join(extract_dir, 'config.json'), 'r') as f:
    config = json.load(f)

# Step 3: Recreate the model from config
print("Recreating model from config...")
model = tf.keras.models.model_from_json(json.dumps(config))

# Step 4: Load the weights
print("Loading weights...")
weights_path = os.path.join(extract_dir, 'model.weights.h5')

# Load weights into the model
model.load_weights(weights_path)

print("✅ Model successfully reconstructed!")

# Step 5: Verify the model works
print("\nModel summary:")
model.summary()

print(f"\nModel input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Step 6: Test with a dummy input
print("\nTesting with dummy input...")
dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
predictions = model.predict(dummy_input, verbose=0)
print(f"✅ Prediction successful! Output shape: {predictions.shape}")

# Step 7: Save in a more compatible format
print("\nSaving model in SavedModel format...")
model.save('plant_disease_model_fixed', save_format='tf')
print("✅ Model saved to 'plant_disease_model_fixed'")

print("\nSaving model as H5...")
model.save('plant_disease_model_fixed.h5')
print("✅ Model saved to 'plant_disease_model_fixed.h5'")

# Cleanup
shutil.rmtree(extract_dir)
print("\n✅ All done! You can now use the fixed model files.")
