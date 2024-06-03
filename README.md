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

## Setup and Installation

1. Clone the repository to your local machine.
2. Install the required dependencies using pip:
    ```
    pip install -r requirements_1.txt
    ```
    ```
    pip install -r requirements_2.txt
    ```
3. Run the application:
    ```
    python app.py
    ```

## Usage

1. Navigate to the root URL of the web application.
2. Use the upload form to upload an image.
3. The application will save the image and use a trained model to make a prediction.

## Note

The application assumes the existence of a function named `predict` for localization in a module named `image_utils`. Make sure this function is properly defined and working.

## Model

The application uses a trained model stored in a file named `model.h5`. Make sure this file is in the same directory as `app.py`.

## Image Storage

Uploaded images are stored in the `static/img` directory. Make sure this directory exists and the application has write permissions to it.
