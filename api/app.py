from flask import Flask, request,render_template, json

#import pickle
import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf
#from tensorflow.keras.preprocessing import image
from  keras.preprocessing import image

app = Flask(__name__)

model = tf.keras.models.load_model('ml/my_model')

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/predict',methods=["Get","POST"])
def predict():
    valorenv=""
    try:
        races=['Black','East Asian','Indian','Latino_Hispanic','Middle Eastern','Southeast Asian','White']
        new_file = request.files['file']
        valorenv="1 "
        target_path = os.path.join("upload",new_file.filename)
        valorenv=" 2 "+target_path
        new_file.save(target_path)
        valorenv=valorenv+" 3 "
        #image = cv2.imread(target_path, 0)
        #image = data_validation(image)
        batch_size = 32
        img_height = 224
        img_width = 224
        input_shape=(img_height, img_width, 3)
        valorenv=valorenv+" 4 "
        new_img = image.load_img(target_path, target_size=input_shape[:2])
        valorenv=valorenv+" 5 "
        new_img = image.img_to_array(new_img)
        valorenv=valorenv+" 6 "
        new_img = np.expand_dims(new_img, axis=0)
        valorenv=valorenv+" 7 "
        new_img = new_img / 255.0
        valorenv=valorenv+" 8 "
        #prediction = model.predict(image)
        prediction = model.predict(new_img)
        valorenv=valorenv+" 9 "
        #predicted_label = np.argmax(prediction, 1)
        #return f"Es un {predicted_label[0]} ;) !!"
        laPrediccion=np.argmax(prediction)
        valorenv=valorenv+"10 "
    except Exception as e:
        valorenv=valorenv+" "+e.args[0]
        return f"el error {valorenv}"
    return f"This person is a  {races[laPrediccion]} ;) !!"

def data_validation(image):
    # image = image.flatten()
    image = np.expand_dims(image, axis=0)
    # Convertir la imagen a float32 para usar valores decimales en el tensor
    image = tf.cast(image, tf.float32)
    # Dividir el tensor entre el nivel de intensidad mas alto en la imagen
    return image

if __name__ == '__main__':
     app.run(debug=True, port=5002)