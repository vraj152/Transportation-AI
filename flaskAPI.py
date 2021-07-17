from flask import Flask, render_template, request
from flask_cors import CORS
import json
from werkzeug.utils import secure_filename

import cv2
import torch

import os
import shutil

app = Flask(__name__)
CORS(app)

outputfile_path = "OutputImages"
inputfile_path = "InputImages"

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")

@app.route('/objectDetect', methods=['POST'])
def object_detection():
    input_files = request.files.getlist('files[]')
    
    preprocess_directory(outputfile_path)
    preprocess_directory(inputfile_path)
    
    images = []
    
    for file in input_files:
        filename = secure_filename(file.filename) # save file 
        filepath = os.path.join(inputfile_path, filename);
        file.save(filepath)
        
        images.append(cv2.imread(filepath)[..., ::-1])
    
    meta_data = run_pretrained_model(images)
    res = prepare_response(meta_data)
    
    return json.dumps(res)

def preprocess_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        
    if not os.path.exists(path):
        os.makedirs(path)

def run_pretrained_model(img_arr):
    responses = []
    
    results = model(img_arr, size=640)  # includes NMS

    results.print() 
    results.save(save_dir = outputfile_path)
    
    for i in range(len(img_arr)):
        responses.append(results.pandas().xyxy[i].to_dict())
        
    return responses
    
def prepare_response(data):
    res = []
    
    for ind, x in enumerate(data):
        curr = {}
        curr["path"] = outputfile_path + "/image" + str(ind) + ".jpg"
        curr["stats"] = count_cat(x["name"])
        
        res.append(curr)
    
    return res

def count_cat(dic):
    temp = {}
    
    for key, value in dic.items():
        if(value not in temp):
            temp[value] = 0
            
        temp[value] += 1
        
    return temp

if __name__ == "__main__":
    app.run(host='0.0.0.0')