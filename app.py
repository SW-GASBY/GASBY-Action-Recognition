import cv2
from flask import Flask, request, jsonify
import json
from random import randint

import imageio
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
    
    download_file('gasby-req', uuid, 'resources/' + uuid, uuid+".mp4")
    download_file('gasby-mot-result', uuid, 'resources/' + uuid, uuid+'.json')
    
    video = cv2.VideoCapture('./resources/' + uuid + '/' + uuid + '.mp4')
    
    frames = []
    frame_count = 0
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            break
        if frame_count % 3 != 0:
            frame_count += 1
            continue
        
        frames.append(frame)
        frame_count += 1

    with open('./resources/' + uuid + '/'+uuid+'.json', 'r') as f:
        mot_results = json.load(f)
    
    # team, color는 우선 생략
    players = []
    for r in mot_results:
        player = Player(r['player_id'], r['team'])
        bboxs = {}
        position_names = {}
        for pos in r['positions']:
            frame = pos['frame']
            box = pos['box']
            pos_name = pos['position_name']
            # 행동인식 모델에서 사용하는 바운딩 박스에 맞게 변경 (x1, y1, x2, y2) ->   
            # act_bbox = 
            bboxs[frame] = box
            position_names[frame] = pos_name
           
        player.bboxs = bboxs
        player.positions = position_names
        players.append(player)
    
    players, actions = ActioRecognition(frames, players)
    
    # For Debugging (행동인식 결과 저장하고, 불러옴)
    # if not os.path.exists('outputs/' + uuid):
    #     os.mkdir('outputs/' + uuid)
    # pickle.dump(actions, open('./outputs/' + uuid + '/actions.pkl', 'wb'))
    
    # actions = pickle.load(open('./outputs/' + uuid + '/actions.pkl', 'rb'))
    
    # 행동 제한하는 부분
    for player in players:
        for action in player.actions:
            if action == "8":
                player.actions[action] = "9"
    
    labels = {"0" : "block", "1" : "pass", "2" : "run", "3" : "dribble", "4" : "shoot", "5" : "ball in hand", "6" : "defense", "7" : "pick" , "8" : "no_action" , "9" : "walk" , "10" : "discard"}
    for i in range(len(players)):
        action = actions[i]
        for j in range(len(players[i].bboxs)):
            frame_nums = list(players[i].bboxs.keys())
            action_idx = j // 16
            if action_idx >= len(action):
                action_idx = len(action) - 1
            players[i].actions[frame_nums[j]] = labels[str(action[action_idx])]

    # # 동영상에 선수 바운딩박스와 행동 입력
    # # 행동 결과 확인용 gif 생성 부분
    for frame_idx, frame in enumerate(frames):
        for player in players:
            if frame_idx in player.bboxs:
                bbox = player.bboxs[frame_idx]
                action = player.actions.get(frame_idx, 'No Action')
                # 바운딩박스 그리기
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                # 행동 라벨 표시
                cv2.putText(frame, action, (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        frames[frame_idx] = frame
        
    with imageio.get_writer('outputs/output.gif', mode='I', fps=10) as writer:
        for frame in frames:
                writer.append_data(frame)

    json_list = create_json(players, actions, frame_len=len(frames))
    
    if not os.path.exists('outputs/' + uuid):
        os.mkdir('outputs/' + uuid)
    with open('outputs/' + uuid + '/' + uuid + '.json', 'w') as json_file:
        json.dump(json_list, json_file, indent=4)
    
    upload_file('gasby-actrecog-result', uuid, uuid+'.json', uuid)
    
    shutil.rmtree('resources/' + uuid)
    shutil.rmtree('outputs/' + uuid)

    return json_list

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000, debug=True)