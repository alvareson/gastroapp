from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import json
import h5py
from PIL import Image

# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
app.secret_key = 'secretkey'

# Model saved with Keras model.save()
MODEL_PATH = '/home/alister/flask_apps/gastroapp/gastromodel.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        im = Image.open(file_path)
        (width, height) = im.size
        print(width, height)

        if width < 224:
            flash('Размер изображения ниже порогового значения. Пожалуйста загрузите другое изображение')

        # Make prediction
        preds = model_predict(file_path, model)


        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        # return result

        def decode_my_predictions(preds, top=5, class_list_path=None):
            if len(preds.shape) != 2 or preds.shape[1] != 8:
                raise ValueError('`decode_predictions` expects '
                                 'a batch of predictions '
                                 '(i.e. a 2D array of shape (samples, 1000)). '
                                 'Found array with shape: ' + str(preds.shape))
            index_list = json.load(open(class_list_path))
            results = []
            for pred in preds:
                top_indices = pred.argsort()[-top:][::-1]
                result = [tuple(index_list[str(i)]) + (pred[i],) for i in top_indices]
                result.sort(key=lambda x: x[2], reverse=True)
                results.append(result)
            return results

        most_likely_labels = decode_my_predictions(preds, top=5, class_list_path='/home/alister/flask_apps/gastroapp/class_gastro.json')
        print(most_likely_labels)
        # result = str(most_likely_labels[0][0][1]) + str(float(most_likely_labels[0][0][2])*100)
        result = str(most_likely_labels[0][0][1:3])
        return result

    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()