#  Image Upload and Prediction Application

This is a Flask web application that allows users to upload images and uses a trained model to make predictions.

## Features

- Image upload functionality
- Image prediction using a trained TensorFlow/Keras model

## Dependencies

- Flask
- Flask-Uploads
- TensorFlow
- Keras
- OpenCV
- numpy

## Usage

1. Navigate to the root URL of the web application.
2. Use the upload form to upload an image.
3. The application will save the image and use a trained model to make a prediction.


## Model

The application uses a trained model stored in a file named `model.h5`. Make sure this file is in the same directory as `app.py`.

## Image Storage

Uploaded images are stored in the `static/img` directory. Make sure this directory exists and the application has write permissions to it.
