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
    
    download_file('gasby-mot-result', uuid, 'resources/' + uuid, 'frames.pkl')
    download_file('gasby-mot-result', uuid, 'resources/' + uuid, 'player_positions_filtered.json')
    
    frames = pickle.load(open('./resources/' + uuid + '/frames.pkl', 'rb'))
    
    with open('./resources/' + uuid + '/player_positions_filtered.json', 'r') as f:
        mot_results = json.load(f)
    
    
    # mot results -> Player (box -> 행동인식 바운딩 박스 양식 맞춰야함)
    # team, color는 우선 생략
    playerBoxes = []
    players = []
    for r in mot_results:
        player = Player(r['player_id'], 'USA', 'white')
        bboxs = []
        position_names = []
        for pos in r['position']:
            box = pos['box']
            pos_name = pos['position_name']
            # 행동인식 모델에서 사용하는 바운딩 박스에 맞게 변경 (x1, y1, x2, y2) ->   
            # act_bbox = 
            bboxs.append(box)
            position_names.append(pos_name)
           
        player.bboxs = bboxs
        player.positions = position_names
        playerBoxes.append(bboxs)
        players.append(player)
    
    actions = ActioRecognition(frames, playerBoxes)
    
    # For Debugging
    # if not os.path.exists('outputs/' + uuid):
    #     os.mkdir('outputs/' + uuid)
    # pickle.dump(actions, open('./outputs/' + uuid + '/actions.pkl', 'wb'))
    # actions = pickle.load(open('./outputs/' + uuid + '/actions.pkl', 'rb'))
    
    # for i in range(len(players)):
    #     action = actions[i]
    #     for j in range(len(players[i].bboxs)):
    #         players[i].actions.append(action[j // 16])
    


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