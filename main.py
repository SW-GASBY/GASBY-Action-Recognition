import json
from random import randint
from service.action_recognition import ActioRecognition
import pickle
import torch

def create_json(players):
    json_list = []
    pl = []
    for i in range(28):
        if i*15 > 230:
            break
        temp = []
        for j in range(10):
            if not(players[j].bboxs[i*15][0] == 0 and players[j].bboxs[i*15][1] == 0 and players[j].bboxs[i*15][2] == 0 and players[j].bboxs[i*15][3] == 0):
                ac = actions[j][i]
                if j != 0 and (actions[j][i] == 3 or actions[j][i] == 4 or actions[j][i] == 0):
                    if i >= 9:
                        ac = 9
                    else:
                        ac = 2  
                if j == 0:
                    ac = 3      
                actions[j][i] = {'box': (players[j].bboxs[i*15][0], players[j].bboxs[i*15][1], players[j].bboxs[i*15][2] - players[j].bboxs[i*15][0], players[j].bboxs[i*15][3] - players[j].bboxs[i*15][1]),
                                 'action': ac}
                temp.append({'id': players[j].ID, 'team': 'USA' if players[j].team == 'white' else 'NGR', 'box': players[j].bboxs[i*15], 'action': ac})
                json_list.append({'player': players[j].ID,
                                  'time': i * 0.6217,
                                  'team': 'USA' if players[j].team == 'white' else 'NGR',
                                  'action': ac})
        pl.append({i*15: temp})
    return json_list, pl

frames, playerBoxes = pickle.load(open('./resources/variables.pkl', 'rb'))

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
print(actions)

# json_list, pl = create_json(players)
# json_file_path = 'action.json'
# with open(json_file_path, 'w') as json_file:
#     lable = {"0" : "block", "1" : "pass", "2" : "run", "3" : "dribble", "4" : "shoot", "5" : "ball in hand", "6" : "defense", "7" : "pick" , "8" : "no_action" , "9" : "walk" , "10" : "discard"}
#     for i in json_list:
#         i['action'] = lable[str(i['action'])]
#     json.dump(json_list, json_file, indent=4)

        
