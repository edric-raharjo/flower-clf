from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('flower_classifier.h5')

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Create feedback storage
FEEDBACK_FILE = 'feedback_data.json'

def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feedback_data(data):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def preprocess_image(image):
    """Preprocess image for model prediction"""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Add route to serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Generate unique prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        return jsonify({
            'prediction_id': prediction_id,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        is_correct = data.get('is_correct')
        correct_class = data.get('correct_class', '')
        
        # Load existing feedback
        feedback_data = load_feedback_data()
        
        # Add new feedback
        feedback_entry = {
            'prediction_id': prediction_id,
            'is_correct': is_correct,
            'correct_class': correct_class,
            'timestamp': datetime.now().isoformat()
        }
        
        feedback_data.append(feedback_entry)
        save_feedback_data(feedback_data)
        
        return jsonify({'message': 'Feedback received successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback-stats', methods=['GET'])
def get_feedback_stats():
    try:
        feedback_data = load_feedback_data()
        
        total_feedback = len(feedback_data)
        correct_predictions = sum(1 for f in feedback_data if f['is_correct'])
        accuracy = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0
        
        return jsonify({
            'total_feedback': total_feedback,
            'correct_predictions': correct_predictions,
            'accuracy': round(accuracy, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': class_names})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
