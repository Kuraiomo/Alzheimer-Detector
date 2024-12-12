from __future__ import division, print_function
# coding=utf-8
import sys
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np

# Define a flask app
app = Flask(__name__)

# Path to the model file
MODEL_PATH = 'alzheimer_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# Define the class dictionary that maps indices to labels (modify it according to your model's classes)
class_dict = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(image_path, model, class_dict):
    """
    Classify an input image using the trained model.

    Parameters:
    - model: Trained TensorFlow/Keras model
    - image_path: Path to the image file
    - class_dict: Dictionary mapping class indices to labels

    Returns:
    - str: Predicted classification name
    """
    # Load the image
    img = load_img(image_path, target_size=(128, 128))
    # Convert the image to an array
    img_array = img_to_array(img)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict using the model
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])  # Get the index of the highest probability

    # Map index to class name
    predicted_class = class_dict[class_index]

    return predicted_class

@app.route('/', methods=['GET'])
def index():
    # Main page for uploading files
    return render_template('index.html')

# Define the class dictionary
class_dict = {
    0: 'Mild Demented',
    1: 'Moderate Demented',
    2: 'Non Demented',
    3: 'Very Mild Demented'
}

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        f = request.files['file']

        # Ensure the 'uploads' directory exists
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the file to the uploads folder
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction and pass the class_dict
        result = model_predict(file_path, model, class_dict)

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
