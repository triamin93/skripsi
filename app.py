from flask import Flask, render_template, flash, redirect, url_for, session, request, jsonify, Response
from flask_mysqldb import MySQL
from wtforms import Form, StringField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from wtforms.fields.html5 import EmailField
import os
import numpy as np
import re
import base64
import cv2
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils


app = Flask(__name__)

app.secret_key = os.urandom(24)

# Mengambil Model
quality_model = load_model('model/model_coba2.h5')

# mengaktifkan webcam
# camera = cv2.VideoCapture(0)

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

# Fungsi untuk prediksi gambar
def model_predict(image, model):
    image = image.resize((224, 224))           
    image = image_utils.img_to_array(image)
    image = image.reshape(-1, 224, 224, 3)
    image = image.astype('float32')
    image = image / 255.0
    preds = model.predict(image)
    return preds

# Routing
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/image")
def image_classify():
    return render_template("image.html")

@app.route("/webcam")
def webcam_classify():
    return render_template("webcam.html")

@app.route('/prediction-image', methods=['GET','POST'])
def prediction_image():
    if request.method=='POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        # Make prediction
        preds = model_predict(img, quality_model)
        target_names = ['Busuk', 'Setengah Segar', 'Segar']     
        hasil_label = target_names[np.argmax(preds)]
        hasil_prob = "{:.2f}".format(100 * np.max(preds))
        return jsonify(result=hasil_label, probability=hasil_prob)

    return render_template('image.html')

def prediction_webcam(): 
    camera = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            print("[INFO] loading and preprocessing image...")
            image = Image.fromarray(frame, 'RGB')
            image = image.resize((224,224))
            image = image_utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Classify the image
            print("[INFO] classifying image...")
            preds = quality_model.predict(image)
            target_names = ['Busuk', 'Setengah Segar', 'Segar']     
            hasil_label = target_names[np.argmax(preds)]
            hasil_prob = "{:.2f}".format(100 * np.max(preds))
            # prediction = f'{hasil_prob}% {hasil_label}'

            # cv2.putText(frame, "Label: {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (209, 80, 0, 255), 2)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concat frame one by one and show result
    camera.release()
        
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(prediction_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)

