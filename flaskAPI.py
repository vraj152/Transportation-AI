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

outputfile_path = "static/OutputImages"
inputfile_path = "InputImages"

"""
Load the model before server gets launched.
This will make the subsequent requests to the server faster.

Model's being used: Yolov5 (https://github.com/ultralytics/yolov5)
"""

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


"""
Endpoint: Whenever the application will be launched, this endpoint will
serve the request. It will render "home.html"

Request Type: GET

"""

@app.route('/', methods=['GET'])
def home():
    return render_template("home.html")


"""
Endpoint -- "objectDetect"
Request type: POST

This method will accept the list of images uploaded by user and perform
object detection on them. Will return the appropriate JSON response to 
client.

Steps:
    1. Delete if exists - Input image directory
    2. Delete if exists - Output image directory
    3. Will iterate through list of files and store cv2 objects of each
       files in array.
    4. Run pretrained model on these images.
    5. Prepare the appropriate response which will be sent to client.
    6. Return response.
"""

@app.route('/objectDetect', methods=['POST'])
def object_detection():
    input_files = request.files.getlist('files[]')
    
    preprocess_directory(outputfile_path)
    preprocess_directory(inputfile_path)
    
    images = []
    height_needed = 0
    
    for file in input_files:
        filename = secure_filename(file.filename) # save file 
        filepath = os.path.join(inputfile_path, filename);
        file.save(filepath)
        
        curr_img = cv2.imread(filepath)[..., ::-1]
        
        height_needed = max(height_needed, curr_img.shape[0])
        images.append(curr_img)
    
    meta_data = run_pretrained_model(images)
    ls = prepare_response(meta_data)
    
    print(ls)
    
    res = {
        "height_needed" : height_needed + 50,
        "ls" : ls
    }
    
    return json.dumps(res)


"""
Helper function: This function will check if directory exists. And if it
    does, it will delete it.

Accepts:
    Path : String - which specifies the path of the directory
"""

def preprocess_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        
    if not os.path.exists(path):
        os.makedirs(path)

"""
Helper function: This function will run pretrained model on all the images.
    And will save the processed images in output image directory.

Accepts:
    img_arr : Array of cv2 objects
    
Returns:
    List of dictionary of results produced by Yolov5.
"""

def run_pretrained_model(img_arr):
    responses = []
    
    results = model(img_arr, size=640)  # includes NMS

    #results.print() 
    results.save(save_dir = outputfile_path)
    
    for i in range(len(img_arr)):
        responses.append(results.pandas().xyxy[i].to_dict())
        
    return responses
    

"""
Helper function: This function will prepare the response which will be later
    sent to client.

Accepts:
    data = list of dictionaries 

Returns:
    res = List of dictionaries with only the values which will be rendered
    on frontend.
    
Sample output:
    [{
      'path': 'static/OutputImages/image0.jpg', 
      'stats': {
             'person': 13, 
             'handbag': 1
        }
   }]
    
    where path specifies the path of output image
    stats specifies the frequency of each identified objects.
    
"""
def prepare_response(data):
    res = []
    
    for ind, x in enumerate(data):
        curr = {}
        
        temp = outputfile_path + "/image" + str(ind) + ".jpg"
        resized = cv2.resize(cv2.imread(temp), (300, 270))
        cv2.imwrite(temp, resized)

        curr["path"] = temp
        curr["stats"] = count_cat(x["name"])
        
        res.append(curr)
    
    return res

"""
Helper function: This function will accept the dictionary of all the objects
    identified by the model. It will return the frequency map of this 
    dictionary.
    
Accepts: Dictionary
Returns: Refined dictionary.
"""

def count_cat(dic):
    temp = {}
    
    for key, value in dic.items():
        if(value not in temp):
            temp[value] = 0
            
        temp[value] += 1
        
    return temp

if __name__ == "__main__":
    app.run(host='0.0.0.0')
