from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

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
