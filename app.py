from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io


# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # This allows your HTML file to communicate with the server

# Define constants
IMG_SIZE = (224, 224)
class_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Load the pre-trained model once when the server starts
try:
    model = load_model('blood_group_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Create an API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        # Read the image data from the file
        img = image.load_img(io.BytesIO(file.read()), target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_blood_group = class_labels[predicted_class_index]
        
        # Prepare the response
        response = {
            'prediction': predicted_blood_group,
            'probabilities': {label: float(prob) for label, prob in zip(class_labels, predictions[0])}
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
