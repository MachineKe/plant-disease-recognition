import tensorflow as tf
import zipfile
import json
import numpy as np
import os
import h5py
import tempfile
import shutil

print("TensorFlow version:", tf.__version__)

# Step 1: Extract the model files
model_path = 'plant_disease_model.keras'
extract_dir = '/tmp/model_debug'

print(f"Extracting {model_path} to {extract_dir}...")
with zipfile.ZipFile(model_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Step 2: Load the configuration
print("\n--- Loading model configuration ---")
with open(os.path.join(extract_dir, 'config.json'), 'r') as f:
    config = json.load(f)

# Print model architecture summary
print(f"Model name: {config.get('name', 'Unknown')}")
print(f"Number of layers: {len(config.get('layers', []))}")

# Step 3: Inspect weights file
print("\n--- Inspecting weights file ---")
weights_path = os.path.join(extract_dir, 'model.weights.h5')

with h5py.File(weights_path, 'r') as f:
    print("Keys in weights file:")
    for key in f.keys():
        print(f"  - {key}")
    
    # Try to list all weight groups
    def print_weights(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"    Weight: {name} - shape: {obj.shape}")
    
    print("\nAll weights in file:")
    f.visititems(print_weights)

# Step 4: Try to manually load weights
print("\n--- Attempting manual weight loading ---")

# First, recreate the model
model = tf.keras.models.model_from_json(json.dumps(config))
print("Model recreated from config")

# Get layer names from model
print("\nModel layers:")
for i, layer in enumerate(model.layers):
    print(f"  Layer {i}: {layer.name}")
    if layer.weights:
        for w in layer.weights:
            print(f"    Weight: {w.name} - shape: {w.shape}")

# Try to load weights with a different approach
print("\n--- Attempting to load weights manually ---")

with h5py.File(weights_path, 'r') as f:
    # Try different possible weight group names
    possible_groups = ['model_weights', 'weights', 'layer_weights', '']
    
    for group_name in possible_groups:
        if group_name and group_name in f:
            print(f"Found group: {group_name}")
            weight_group = f[group_name]
            break
        elif not group_name:
            # Use root
            weight_group = f
            print("Using root group")
    
    # Try to assign weights manually
    for layer in model.layers:
        layer_name = layer.name
        print(f"\nProcessing layer: {layer_name}")
        
        # Look for weights in different possible paths
        found = False
        
        # Try different possible paths for this layer's weights
        for path in [layer_name, f"layer_with_weights/{layer_name}", f"layers/{layer_name}"]:
            if path in weight_group:
                layer_weights_group = weight_group[path]
                print(f"  Found weights at: {path}")
                
                # Get the weight names from the layer
                weight_names = [w.name.split('/')[-1].split(':')[0] for w in layer.weights]
                
                # Try to load each weight
                for i, weight_name in enumerate(weight_names):
                    # Try different possible names for the weight
                    for possible_name in [weight_name, f"{layer_name}/{weight_name}", f"kernel:0", f"bias:0"]:
                        if possible_name in layer_weights_group:
                            weight_data = layer_weights_group[possible_name][()]
                            print(f"    Loaded {possible_name} with shape {weight_data.shape}")
                            layer.set_weights([weight_data] + layer.get_weights()[1:])
                            found = True
                            break
                    else:
                        print(f"    Could not find weight: {weight_name}")
        
        if not found:
            print(f"  ⚠️ No weights found for layer: {layer_name}")

# Step 5: Save the model if we succeeded
print("\n--- Attempting to save model ---")
try:
    model.save('plant_disease_model_fixed2.keras')
    print("✅ Model saved as 'plant_disease_model_fixed2.keras'")
except Exception as e:
    print(f"❌ Failed to save model: {e}")

# Cleanup
shutil.rmtree(extract_dir)
print("\nDebug complete!")
