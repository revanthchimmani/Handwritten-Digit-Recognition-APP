import flask
import io
import time
import os
from keras.models import load_model
import numpy as np
import cv2
from flask import Flask, jsonify, request, flash
from PIL import Image
from werkzeug.utils import secure_filename, redirect
from flask import Response
app = Flask(__name__)
UPLOAD_FOLDER = 'MC_images'
model = load_model('final_model.h5')


def predict_digit(img):
    img = Image.open(io.BytesIO(img))
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


def get_digit(img_bytes):
    digit, acc = predict_digit(img_bytes)
    # print("Recognized as: ", digit, " with accuracy: ", acc * 100)
    return digit, acc


@app.route('/predict', methods=['POST','GET'])
def predict():
    print("predict enter")
    if flask.request.method == 'POST':
        if 'image' not in flask.request.files:
            flash('No image uploaded')
            return redirect(flask.request.url)

        input_file = flask.request.files['image']

        if not input_file:
            print("not file Please try again. The Image doesn't exist")
            return
        img_bytes = input_file.read()
        digit, acc = get_digit(img_bytes)
        print("Recognized as: ", digit, " with accuracy: ", acc * 100)
        # print("Recognized as: ", digit)
        target_folder = f"{UPLOAD_FOLDER}/{digit}"

        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)

        image_filename = secure_filename(input_file.filename)
        time_str = time.strftime("%Y%m%d-%H%M%S")
        input_file.seek(0)
        input_file.save(os.path.join(target_folder, time_str + '_' + image_filename))

        print("Image upload success!")
        return 'Image recognized as ' + str(digit) + ' successfully!'


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
