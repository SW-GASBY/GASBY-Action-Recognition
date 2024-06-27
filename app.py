from flask import Flask, request, jsonify
import json
from random import randint
from service.action_recognition import ActioRecognition, create_json
import pickle
import torch
from utils.s3utils import download_file, upload_file
import os
import shutil
from entity.player import Player
from flask_cors import CORS

app = Flask(__name__)
CORS

@app.route('/action-predict')
def check():
    return 'ok'

@app.route('/action-predict/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No data found'})
    
    uuid = data['uuid']
    
    download_file('gasby-mot-result', uuid, 'resources/' + uuid, 'variables.pkl')
    
    frames, playerBoxes = pickle.load(open('./resources/' + uuid + '/variables.pkl', 'rb'))

    players = []
    for i in range(1, 6):
        players.append(Player(i, 'white', (255, 255, 255)))
        players.append(Player(i, 'black', (255, 255, 255)))
    players.append(Player(0, 'referee', (0, 0, 0)))
    
    for player in players:
        for i in range(len(frames)):
            player.bboxs[i] = (playerBoxes[i][player.ID-1])
    
    actions = ActioRecognition(frames, playerBoxes) 
    json_list = create_json(players, actions, frame_len=len(frames))

    # 행동 제한하는 부분
    # for i in range(28):
    #         for j in range(10):
    #             if j != 0:
    #                 if i >= 9:
    #                     actions[j][i] = 9
    #                 else:
    #                     actions[j][i] = 2  
    #             if j == 0:
    #                 actions[j][i] = 3    
    
    if not os.path.exists('outputs/' + uuid):
        os.mkdir('outputs/' + uuid)
    with open('outputs/' + uuid + '/action.json', 'w') as json_file:
        json.dump(json_list, json_file, indent=4)
    
    upload_file('gasby-actrecog-result', uuid, 'action.json')
    
    shutil.rmtree('resources/' + uuid)
    shutil.rmtree('outputs/' + uuid)

    return json_list

if __name__ == '__main__':
    app.run(debug=True)