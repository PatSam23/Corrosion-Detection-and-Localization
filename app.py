from flask import Flask, request, render_template, redirect, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from image_utils import predict  # Assuming predict function for localization

app = Flask(__name__)

# Configure file uploads
images = UploadSet('images', IMAGES)
app.config['UPLOADED_IMAGES_DEST'] = 'static/img'
configure_uploads(app, images)

# Load trained model for rust detection
model = load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST' and 'image' in request.files:
        # Save uploaded image
        filename = images.save(request.files['image'])
        return redirect(url_for('check_rust', filename=filename))
    return render_template('index.html')

@app.route('/check_rust/<filename>', methods=['GET', 'POST'])
def check_rust(filename):
    img_path = os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename)
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    test_x = img_to_array(img_resized)
    test_x = test_x.reshape((1,) + test_x.shape)
    test_x = test_x.astype('float32') / 255
    prediction = model.predict(test_x)
    rust_detected = prediction > 0.5

    if request.method == 'POST':
        return redirect(url_for('localize_rust', filename=filename))

    return render_template('result.html', rust_detected=rust_detected, filename=filename)

@app.route('/localize_rust/<filename>', methods=['GET', 'POST'])
def localize_rust(filename):
    img_path = os.path.join(app.config['UPLOADED_IMAGES_DEST'], filename)
    img = cv2.imread(img_path)
    result_img_path = predict(img)  # Assuming predict function saves the result image and returns the path
    return render_template('localization_result.html', result_img_path=result_img_path)

if __name__ == '__main__':
    app.run(debug=True)
