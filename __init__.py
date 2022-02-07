# -*- coding: utf-8 -*-
from io import BytesIO
import cv2
import flask
import os
from flask import json, request
import urllib.request
import matplotlib.image as mpimg
from werkzeug.exceptions import HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image



# Flask initialization:
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Handling requests:
@app.route('/document-type', methods=['POST'])
def home():
        if request.method == 'POST':
                
                if flask.request.files.get("file") == None:
                    return {"result":"FILE not present", "status":400}, 400

                img_requested = flask.request.files["file"].read()
                img = Image.open(BytesIO(img_requested))
                img = img.resize((150, 150))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                
                classnames=["CEDULA_UY","PASAPORTE"]
                model = load_model('final.h5')
                try:
                    prediction = model.predict(img) 
                except:
                    return {"result":["NO_PREDICTION"], "status":400}, 400

                return {"result":classnames[int(prediction[0][0])], "status":200}, 200


def download_image(url, file_path, file_name):
    full_path = file_path + file_name + '.jpg'
    urllib.request.urlretrieve(url, full_path)


@app.route('/document-type-url', methods=['POST'])
def with_url():
        if request.method == 'POST':
                
                if flask.request.form.get("url") == None:
                    return {"result":"URL not present", "status":400}, 400

                

                url = flask.request.form.get("url") #prompt user for img url
                file_name = 'file' #prompt user for file_name

                download_image(url, 'static/', file_name)
                    
                basedir = os.path.abspath(os.path.dirname(__file__))
                img_requested = os.path.join(basedir, 'static/file.jpg')                
                img = Image.open(img_requested)
                img = img.resize((150, 150))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                
                classnames=["CEDULA_UY","PASAPORTE"]
                model = load_model('final.h5')
                try:
                    prediction = model.predict(img) 
                except:
                    return {"result":["NO_PREDICTION"], "status":400}, 400

                try:
                    os.remove(img_requested)
                except:
                    print("Warning: the file downloaded can't be removed")

                return {"result":classnames[int(prediction[0][0])], "status":200}, 200

import math
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

import face_recognition
import numpy as np

import PIL
import PIL.Image


@app.route('/validate-document', methods=['POST'])
def other():
        if request.method == 'POST':
                #default threshold
                threshold = 0.6

                if flask.request.files.get("document") == None:
                    return {"result":"DOCUMENT file not present", "status":400}, 400

                if flask.request.files.get("face") == None:
                    return {"result":"FACE file not present", "status":400}, 400

                if flask.request.form.get("threshold"):
                    threshold = float(flask.request.form["threshold"])

                img_document = flask.request.files["document"].read()
                img_face = flask.request.files["face"].read()

                im_doc = PIL.Image.open(BytesIO(img_document))
                im_doc = im_doc.convert('RGB')
                im_doc = np.array(im_doc)

                im_face = PIL.Image.open(BytesIO(img_face))
                im_face = im_face.convert('RGB')
                im_face = np.array(im_face)
                
                try:

                    document_encoding = face_recognition.face_encodings(im_doc)[0]
                    unknown_encoding = face_recognition.face_encodings(im_face)[0]

                    results = face_recognition.face_distance([document_encoding], unknown_encoding)   
                    match_percentage = face_distance_to_conf(results,threshold)
                          
                    if(match_percentage[0]>threshold):
                        return {"result":{"MATCH_PERCENTAGE":match_percentage[0]}, "status":200}
                    else:
                        return {"result":"NO_COINCIDENCE", "status":200}
                except:
                    return {"result":["NO_FACES_DETECTED"], "status":400}, 400
                    

@app.route('/validate-document-url', methods=['POST'])
def validate_document():
        if request.method == 'POST':
                #default threshold
                threshold = 0.6

                if flask.request.form.get("document_url") == None:
                    return {"result":"DOCUMENT_URL not present", "status":400}, 400

                if flask.request.form.get("face_url") == None:
                    return {"result":"FACE_URL not present", "status":400}, 400

                if flask.request.form.get("threshold"):
                    threshold = float(flask.request.form["threshold"])

                document_url = flask.request.form.get("document_url") 
                document_file_name = 'document' 

                face_url = flask.request.form.get("face_url") 
                face_file_name = 'face' 

                download_image(document_url, 'static/', document_file_name)
                download_image(face_url, 'static/', face_file_name)

                basedir = os.path.abspath(os.path.dirname(__file__))

                img_document = os.path.join(basedir, 'static/document.jpg')
                img_face = os.path.join(basedir, 'static/face.jpg')                


                im_doc = PIL.Image.open(img_document)
                im_doc = im_doc.convert('RGB')
                im_doc = np.array(im_doc)

                im_face = PIL.Image.open(img_face)
                im_face = im_face.convert('RGB')
                im_face = np.array(im_face)

                try:
                    os.remove(img_document)
                    os.remove(img_face)
                except:
                    print("Warning: the file downloaded can't be removed")
                
                try:

                    document_encoding = face_recognition.face_encodings(im_doc)[0]
                    unknown_encoding = face_recognition.face_encodings(im_face)[0]

                    results = face_recognition.face_distance([document_encoding], unknown_encoding)   
                    match_percentage = face_distance_to_conf(results,threshold)
                          
                    if(match_percentage[0]>threshold):
                        return {"result":{"MATCH_PERCENTAGE":match_percentage[0]}, "status":200}
                    else:
                        return {"result":"NO_COINCIDENCE", "status":200}
                except:
                    return {"result":["NO_FACES_DETECTED"], "status":400}, 400

            
# Handling errors:

# 404
@app.errorhandler(404)
def page_not_found(e):
    return {"result:":"Not found", "status":"404"}, 404

# Mapping JSON instead of HTML for HTTP errors:
@app.errorhandler(HTTPException)
def handle_exception(e):    
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


if __name__ == "__main__":
    app.run()
    #app.run(host='0.0.0.0', port=80)
    


