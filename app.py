#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions


model_path = "Inception_ModelCheck_256_d_0.h5"
model = tf.keras.models.load_model(model_path, compile=False)


def process_image(image_path):
    # Load and prepare the image
    fig_size=256
    image = load_img(image_path, target_size=(fig_size, fig_size))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Scale the image if your model expects pixel values in [0, 1]
    return image

def predict_class(image):
    prediction = model.predict(image)
    # Assuming a model with two outputs: plant type and disease
    # Adjust the indices [0] and [1] based on your model's output structure
    plant_type_pred = np.argmax(prediction[0], axis=1)
    print(prediction[0])
    disease_pred = np.argmax(prediction[1], axis=1)

    return plant_type_pred, disease_pred



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if request.method != 'POST':
        return jsonify({'error': 'Method not allowed'}), 405

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(file.__class__)
        # Construct the URL of the uploaded image
        image_url = url_for('uploaded_file', filename=filename, _external=True)
        image = load_img(file_path, target_size=(256, 256))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add the batch dimension
        image = image / 255.0  # Assuming the model expects pixel values in [0, 1]

        # Now the image is ready to be passed to the model
        plant_type_pred, disease_pred = predict_class(image)
        print("plant_type_pred", plant_type_pred, "disease_pred", disease_pred)

        return jsonify({'message': 'File uploaded successfully', 'image_url': image_url}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['GET'])
def disallow_get_request():
    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, port=8000, use_reloader=False)

