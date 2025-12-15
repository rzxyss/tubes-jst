from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/pepaya_classifier_MobileNetV2.h5'
model = keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = ['mature', 'partiallymature', 'unmature']

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocess
            processed_image = preprocess_image(image)
            
            # Predict
            predictions = model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx]) * 100
            
            # Get all probabilities
            all_predictions = {
                CLASS_NAMES[i]: float(predictions[0][i]) * 100 
                for i in range(len(CLASS_NAMES))
            }
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions
            })
        
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
