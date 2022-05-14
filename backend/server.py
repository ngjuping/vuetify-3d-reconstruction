from flask import Flask, jsonify, request
from flask import send_file # for downloading ply file
from flask import send_from_directory
from flask_cors import CORS

import os
import subprocess as sp

GPU_AVAILABLE_MEMORY_THRESHOLD = 2000

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

# [nhead, nlayer, feedforward, positionEncoding, distanceAugmented]
def get_model_configs():
    return {
    "room_transformer_4head_128ff_k4": [4,2,128,"none",False],
    "room_transformer_dist_augmented_128ff_4head_k4": [4,2,128,"none",True],
    "room_transformer_interp_4head_128ff_k4": [4,2,128,"interp",False],
    "room_transformer_dxyz_128ff_4head_k4": [4,2,128,"dxyz",False]
    }

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

def _CORS(res):
    res.headers.add('Access-Control-Allow-Origin', '*')
    return res

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    model_selected = data["model"]
    asset_selected = data["ply"]
    print(data)
    free_gpu_memory = get_gpu_memory()
    free_gpu = -1
    for index, gpu in enumerate(free_gpu_memory):
        if gpu > GPU_AVAILABLE_MEMORY_THRESHOLD:
            free_gpu = index
            break
    if free_gpu == -1:
        return jsonify("No available GPU(s) found... Please try later.")

    bashCommand = f"source activate conv_onet && CUDA_VISIBLE_DEVICES={free_gpu} python generate.py configs/pointcloud_crop/{model_selected}.yaml"
    process = sp.Popen(bashCommand.split(), stdout=sp.PIPE, cwd='/home/huangzubin/conv_onet')
    output, error = process.communicate()

    return jsonify("Job submitted")


@app.route("/retrieve")
def retrieve():
    return "<p>Get your fucking mesh... NOW</p>"

@app.route("/list")
def ply_list():
    ply_files = []
    for file in os.listdir("./ply_files"):
        if file.endswith(".ply"):
            ply_files.append({
                'name': file.split('.ply')[0],
                'link': file
            })
    return _CORS(jsonify({
        'plys': ply_files
    }))

@app.route("/models")
def model_list():
    models = get_model_configs()
    parsed = []
    for model in models:
        temp = {}
        temp["name"] = model
        model = models[model]
        temp["transformer"] = {}
        temp["transformer"]["nhead"] = model[0]
        temp["transformer"]["nlayer"] = model[1]
        temp["transformer"]["feedforward"] = model[2]
        temp["positionEncoding"] = model[3]
        temp["distanceAugmented"] = model[4]
        parsed.append(temp)

    return _CORS(jsonify(parsed))
    

@app.route('/ply/<path:path>')
def send_ply(path):
    return _CORS(send_from_directory('./ply_files', path))

@app.route('/download')
def downloadFile ():
    pass
    # return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(port=5000,debug=True) 