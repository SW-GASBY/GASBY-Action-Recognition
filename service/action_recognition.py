from __future__ import print_function
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np
from easydict import EasyDict
from random import randint
import sys
from imutils.video import FPS

import torch
import torch.nn as nn
from torchvision import models

from utils.checkpoints import load_weights

args = EasyDict({ 

    'detector': "tracker",

    # Path Params
    'videoPath': "videos/Short4Mosaicing.mp4",

    # Player Tracking
    'classes': ["person"],
    'tracker': "CSRT",
    'trackerTypes': ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'],
    'singleTracker': False,

    # Court Line Detection
    'draw_line': False,

    # YOLOV3 Detector
    'weights': "yolov3.weights",
    'config': "yolov3.cfg",

    'COLORS': np.random.uniform(0, 255, size=(1, 3)),

    # Action Recognition
    'base_model_name': 'r2plus1d_multiclass',
    'pretrained': True,
    'lr': 0.0001,
    'start_epoch': 1,
    'num_classes': 10,
    'labels': {"0" : "block", "1" : "pass", "2" : "run", "3" : "dribble", "4" : "shoot", "5" : "ball in hand", "6" : "defense", "7" : "pick" , "8" : "no_action" , "9" : "walk" , "10" : "discard"},
    'model_path': "model_checkpoints/r2plus1d_augmented-2/",
    'history_path': "histories/history_r2plus1d_augmented-2.txt",
    'seq_length': 16,
    'vid_stride': 16,
    'output_path': "output_videos/"

})

def cropVideo(clip, crop_window,  max_w, max_h, player=0):
    
    video = []
    #print(len(clip))
    
    for i, frame in enumerate(clip):
        x = int(crop_window[player][i][0])
        y = int(crop_window[player][i][1])
        w = int(crop_window[player][i][2] - crop_window[player][i][0])
        h = int(crop_window[player][i][3] - crop_window[player][i][1])

        cropped_frame = frame[y:y+h, x:x+w]
        # max_w또는 max_h보다 작은 경우 padding
        if cropped_frame.shape[0] < max_h:
            cropped_frame = np.pad(cropped_frame, ((0, int(max_h - cropped_frame.shape[0])), (0, 0), (0, 0)), mode='constant')
        if cropped_frame.shape[1] < max_w:
            cropped_frame = np.pad(cropped_frame, ((0, 0), (0, int(max_w - cropped_frame.shape[1])), (0, 0)), mode='constant')
            
        # video.append(cropped_frame)
        
        # resize to 128x176
        try:
            resized_frame = cv2.resize(
                cropped_frame,
                dsize=(int(128),
                       int(176)),
                interpolation=cv2.INTER_NEAREST
            )
        except:
            # Use previous frame
            if len(video) == 0:
                resized_frame = np.zeros((int(176), int(128), 3), dtype=np.uint8)
            else:
                resized_frame = video[i-1]
        assert resized_frame.shape == (176, 128, 3)
        video.append(resized_frame)

    return video

def cropWindows(vidFrames, playerBoxes, seq_length=16, vid_stride=8):
    
    player_count = len(playerBoxes)
    player_frames = {}
    for player in range(player_count):
        player_frames[player] = []

    # How many clips in the whole video
    # n_clips = len(vidFrames) // vid_stride
    player_n_clips = [ len(playerBoxes[player]) // vid_stride for player in range(player_count)]
    # 각 플레이어별 최대 크기의 바운딩 박스 w와 h 찾기
    player_max_w = [0 for _ in range(player_count)]
    player_max_h = [0 for _ in range(player_count)]
    for player in range(player_count):
        for box in playerBoxes[player]:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w > player_max_w[player]:
                player_max_w[player] = w
            if h > player_max_h[player]:
                player_max_h[player] = h
    
    # print(playerBoxes.shape)

    for n_clips in player_n_clips:
        continue_clip = 0
        for clip_n in range(n_clips):
            seq_boxes = [playerBoxes[player][clip_n*vid_stride : clip_n*vid_stride + seq_length] for player in range(player_count)]
            for player in range(player_count):
                if seq_boxes[player] == []:
                    continue
                if len(seq_boxes[player]) != seq_length:
                    continue
                if clip_n*vid_stride + seq_length < len(vidFrames):
                    clip = vidFrames[clip_n*vid_stride: clip_n*vid_stride + len(seq_boxes[player])]
                    #print(" length of clip ", len(clip))
                    #print(np.asarray(cropVideo(clip, crop_window, player)).shape)
                    player_frames[player].append(cropVideo(clip, seq_boxes, player_max_w[player], player_max_h[player], player))
                else:
                    continue_clip = clip_n
                    break
            if continue_clip != 0:
                break

    # # Append to list after padding
    # for i in range(continue_clip, n_clips):
    #     for player in range(player_count):
    #         crop_window = playerBoxes[vid_stride*i:]
    #         frames_remaining = len(vidFrames) - vid_stride * i
    #         clip = vidFrames[vid_stride*i:]
    #         player_frames[player].append(np.asarray(cropVideo(clip, crop_window, player) + [
    #         np.zeros((int(176), int(128), 3), dtype=np.uint8) for x in range(seq_length-frames_remaining)
    #     ]))

    # Check if number of clips is expected
    # assert(len(player_frames[0]) == n_clips)

    return player_frames

def inference_batch(batch):
    # (batch, t, h, w, c) --> (batch, c, t, h, w)
    # (t, c, batch, h, w)
    batch = batch.permute(1, 4, 0, 2, 3)
    return batch

def ActioRecognition(videoFrames, playerBoxes):
    frames = cropWindows(videoFrames, playerBoxes, seq_length=args.seq_length, vid_stride=args.vid_stride)
    print("Number of players tracked: {}".format(len(frames)))
    print("Number of windows: {}".format(len(frames[0])))
    print("# Frames per Clip: {}".format(len(frames[0][0])))
    print("Frame Shape: {}".format(frames[0][0][0].shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Initialize R(2+1)D Model
    model = models.video.r2plus1d_18(pretrained=args.pretrained, progress=True)
    # input of the next hidden layer
    num_ftrs = model.fc.in_features
    # New Model is trained with 128x176 images
    # Calculation:
    model.fc = nn.Linear(num_ftrs, args.num_classes, bias=True)

    model = load_weights(model, args)

    if torch.cuda.is_available():
        # Put model into device after updating parameters
        model = model.to(device)

    model.eval()

    predictions = {}
    for player in range(len(playerBoxes)):
        input_frames_np = np.array(frames[player])
        input_frames_tensor = torch.tensor(input_frames_np, dtype=torch.float).to(device)
        input_frames = inference_batch(input_frames_tensor)
        print('player ', player, ' input_frames ', input_frames.shape)

        input_frames = input_frames.to(device=device)
 
        with torch.no_grad():
            outputs = model(input_frames)
            _, preds = torch.max(outputs, 1)

        # print(preds.cpu().numpy().tolist())
        predictions[player] = preds.cpu().numpy().tolist()

    print('predictions ', predictions)
    
    return predictions


def create_json(players, actions, frame_len):
    json_list = []
    pl = []
    for i in range(frame_len // 16):
        temp = []
        for j in range(len(players)):
            if not(players[j].bboxs[i][0] == 0 and players[j].bboxs[i][1] == 0 and players[j].bboxs[i][2] == 0 and players[j].bboxs[i][3] == 0):
                ac = actions[j][i]
                position = max(players[j].positions,key=players[j].positions[i * 16: (i + 1) * 16].count)    
                actions[j][i] = {'box': (players[j].bboxs[i], players[j].bboxs[i][1], players[j].bboxs[i][2] - players[j].bboxs[i][0], players[j].bboxs[i][3] - players[j].bboxs[i][1]),
                                 'action': ac}
                temp.append({'id': players[j].ID, 'team': 'USA' if players[j].team == 'white' else 'NGR', 'box': players[j].bboxs[i], 'action': ac})
                json_list.append({'player': players[j].ID,
                                  'frame': (i * 16, (i + 1) * 16),
                                  'team': 'USA' if players[j].team == 'white' else 'NGR',
                                  'position' : position,
                                  'action': ac})
    return json_list