from flask import Flask, request, jsonify
import json
from random import randint
from service.action_recognition import ActioRecognition
import pickle
import torch
from utils.s3utils import download_file, upload_file
import os
import shutil

app = Flask(__name__)

@app.route('/action_predict/check')
def check():
    return 'ok'

@app.route('/action_predict/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No data found'})
    
    uuid = data['uuid']
    
    # os.mkdir('resources/' + uuid, exist_ok=True)
    download_file('gasby-mot-result', uuid, 'resources/' + uuid, 'variables.pkl')
    
    frames, playerBoxes = pickle.load(open('./resources/' + uuid + '/variables.pkl', 'rb'))

    actions = ActioRecognition(frames, playerBoxes) 
    for i in range(28):
            for j in range(10):
                if j != 0:
                    if i >= 9:
                        actions[j][i] = 9
                    else:
                        actions[j][i] = 2  
                if j == 0:
                    actions[j][i] = 3      
    colors = []
    for i in range(11):
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    
    shutil.rmtree('resources/' + uuid)

    return actions



# def create_json(players):
#     json_list = []
#     pl = []
#     for i in range(28):
#         if i*15 > 230:
#             break
#         temp = []
#         for j in range(10):
#             if not(players[j].bboxs[i*15][0] == 0 and players[j].bboxs[i*15][1] == 0 and players[j].bboxs[i*15][2] == 0 and players[j].bboxs[i*15][3] == 0):
#                 ac = actions[j][i]
#                 if j != 0 and (actions[j][i] == 3 or actions[j][i] == 4 or actions[j][i] == 0):
#                     if i >= 9:
#                         ac = 9
#                     else:
#                         ac = 2  
#                 if j == 0:
#                     ac = 3      
#                 actions[j][i] = {'box': (players[j].bboxs[i*15][0], players[j].bboxs[i*15][1], players[j].bboxs[i*15][2] - players[j].bboxs[i*15][0], players[j].bboxs[i*15][3] - players[j].bboxs[i*15][1]),
#                                  'action': ac}
#                 temp.append({'id': players[j].ID, 'team': 'USA' if players[j].team == 'white' else 'NGR', 'box': players[j].bboxs[i*15], 'action': ac})
#                 json_list.append({'player': players[j].ID,
#                                   'time': i * 0.6217,
#                                   'team': 'USA' if players[j].team == 'white' else 'NGR',
#                                   'action': ac})
#         pl.append({i*15: temp})
#     return json_list, pl

if __name__ == '__main__':
    app.run(debug=True)