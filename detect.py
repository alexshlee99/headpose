from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from RetinaFace_Torch.data import cfg_mnet, cfg_re50
from RetinaFace_Torch.layers.functions.prior_box import PriorBox
from RetinaFace_Torch.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from RetinaFace_Torch.models.retinaface import RetinaFace
from RetinaFace_Torch.utils.box_utils import decode, decode_landm
import time

from Arcface_Torch.models import *


from config import Config

# from decord import VideoReader
# from decord import cpu, gpu

from filter.crop_utils import alignment_procedure
from detect_engine import detect_Retina


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def main(cap, det_model, rec_model, track_which): 
           
    # Loop over video. 
    i = 0
    while (cap.isOpened()):
        
        ret, frame = cap.read()
        if not ret: 
            break    
        
        im_height, im_width,_ = frame.shape

        # Detect face. 
        dets = detect_Retina(frame, device, det_model, cfg, args)

        # Processing based on detection (crop, align, ID track, Euler angles).
        if len(dets) == 0: 
            print("No face detected...")
            continue
        
        # Iterate through detections, and get all cropped faces. 
        faces = []
        for det in dets: 
            if det[4] < args.vis_thres:  # Check threshold. 
                continue
            
            det = list(map(int, det))
            
            # Bounding box. 
            x_min = max(0, det[0])
            y_min = max(0, det[1])
            x_max = min(det[2], im_width)
            y_max = min(det[3], im_height)
            
            # Crop face. 
            face = frame[y_min:y_max, x_min:x_max]
            
            # Calculate face embedding. 
            face_emb = rec_model(cv2.resize(face, (128, 128)))  # Check dimensions. 
            
            # Append to detection details. 
            bbox_emb_score = [x_min, y_min, x_max, y_max, face_emb, det[4]]
                    
            # Add to array containing all detected "faces".
            faces.append(bbox_emb_score)


        # Sort faces by confidence score (descending order).
        faces.sort(key=lambda x: x[5], reverse=True)
        
        # If first frame, initialize embeddings. 
        if i == 0:
            if track_which == "none":  # If no instructions given, choose most confident face. 
                tracked_faces = [faces[0]]
                curr_face = [faces[0]]
            
            else: # Choose top2, and select by x-position (both, left, right).
                face1, face2 = faces[0], faces[1]
                
                face1_x = int((face1[0]+face1[2])/2)
                face2_x = int((face2[0]+face2[2])/2)
                
                if face1_x < face2_x: 
                    left_and_right = [face1, face2]
                else: 
                    left_and_right = [face2, face1]
                
                # Choose number of faces by given instruction. 
                if track_which == "both":
                    tracked_faces = left_and_right
                    curr_face = left_and_right

                elif track_which == "left": 
                    tracked_faces = [left_and_right[0]]
                    curr_face = [left_and_right[0]]
                    
                else: # "right"
                    tracked_faces = [left_and_right[1]]
                    curr_face = [left_and_right[1]]

        # If not first frame, calculate similarity with tracked face embeddings. 
        # Then, choose the most similar face and continue. 
        else: 
            for prev_face in tracked_faces: 
                for curr_face in faces: 
                    sim = cosin_metric(prev_face[-2], curr_face[-2])
                    
        
        # On first frame, find the faces with max confidence scores (top 2, if "both").
        # On consequent frames, compare embeddings and choose the one with highest similarity. 
        # 

        
        
        
        
        for det in dets: 
            if det[4] < args.vis_thres:  # confidence threshold check. 
                continue
            
            det = list(map(int, det))
            
            # Landmark coordinates. 
            left_eye = (det[5], det[6])   # from our perspective... 
            right_eye = (det[7], det[8])
            nose = (det[9], det[10])
            mouth_left = (det[11], det[12])
            mouth_right = (det[13], det[14])

            # Bounding box. 
            x_min = max(0, det[0])
            y_min = max(0, det[1])
            x_max = min(det[2], im_width)
            y_max = min(det[3], im_height)
            
            # landmarks.
            cv2.circle(frame, left_eye, 2, (0, 0, 255), 4)
            cv2.circle(frame, right_eye, 2, (0, 255, 255), 4)
            cv2.circle(frame, nose, 2, (255, 0, 255), 4)
            cv2.circle(frame, mouth_left, 2, (0, 255, 0), 4)
            cv2.circle(frame, mouth_right, 2, (255, 0, 0), 4)
            
            """
            ### Shift bounding box (to center face). 
            # Use nose to find left/right. 
            # (eye_middle_x, eye_middle_y) = ((left_eye[0]+right_eye[0])/2, (left_eye[1]+right_eye[1])/2)
            # (mouth_middle_x, mouth_middle_y) = ((mouth_left[0]+mouth_right[0])/2, (mouth_left[1]+mouth_right[1])/2)
            
            # if eye_middle_x - nose[0] > 5: # If nose is rotated left by more than 5 pixels. 
            #     # Shift box right. 
            #     x_min += int(abs(mouth_middle_x - nose[0]))
            #     x_max += int(abs(mouth_middle_x - nose[0]))
            
            # elif eye_middle_x - nose[0] < -5: # If nose is rotated right by more than 5 pixels. 
            #     # Shift box left. 
            #     x_min += int(abs(mouth_middle_x - nose[0]))
            #     x_max += int(abs(mouth_middle_x - nose[0]))
        
            # Align.
            # face_aligned = alignment_procedure(face_img, left_eye, right_eye, nose)
            
            # Visualize. 
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            
            # If bounding box size changes drastically, we set that as a new starting point. 
            # And 
            # curr_bbox_size = int((x_max - x_min) * (y_max - y_min))
            # if abs(prev_bbox_size - curr_bbox_size) > 100: 
            #     prev_bbox = (cur_x_min, cur_y_min, cur_x_max, cur_y_max)  # Reset. 
                
            # else: # If not large pose change, then constrain by previous bbox.  
            #     x_min = int((curr_x_min - prev_x_min) / 2)
            #     y_min = max(0, det[1])
            #     x_max = min(det[2], im_width)
            #     y_max = min(det[3], im_height)
            
            """ 
            
            # Edit bbox coordinates based on previous frame.  
            # cur_box_area = int((x_max - x_min) * (y_max - y_min))
            # ratio = cur_box_area / prev_bbox[-1]
            # print(ratio)
            # if 0.95 < ratio < 1.05:  # If assumed to be jitter.
            #     if abs(nose[0] - prev_bbox[-2][0]) < 10 and abs(nose[1] - prev_bbox[-2][1]) < 10: 
            #         x_min = prev_bbox[0]
            #         y_min = prev_bbox[1]
            #         x_max = prev_bbox[2]
            #         y_max = prev_bbox[3]
            
            # Crop by face. 
            face_img = frame[y_min:y_max, x_min:x_max]
            
            # If cropped face is smaller than 224 x 224, resize. 
            cropped_area = int(face_img.shape[0] * face_img.shape[1])
            if cropped_area < int(224 * 224): 
                face_resized = cv2.resize(face_img, (224,224), interpolation=cv2.INTER_CUBIC)  # Enlarge.
            # Else: 
            else: 
                face_resized = cv2.resize(face_img, (224,224), interpolation=cv2.INTER_AREA)  # Shrink.
            
            # Save image.
            name = f"output/{i}.jpg"
            cv2.imwrite(name, face_resized)
            
            # Store previous results. 
            # prev_bbox = (x_min, y_min, x_max, y_max, nose, cur_box_area)
        
        i += 1 
        if i == 100: 
            break

    cap.release()
           

if __name__ == '__main__':
    
    # Initialize Torch. 
    torch.set_grad_enabled(False)
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    # Initialize configuration. 
    args = Config()

    # Face detection. 
    cfg = None
    if args.det_backbone == "mobile0.25":
        cfg = cfg_mnet
    elif args.det_backbone == "resnet50":
        cfg = cfg_re50
    # net and model
    det_model = RetinaFace(cfg=cfg, phase = 'test')
    det_model = load_model(det_model, args.det_trained_weights, not(torch.cuda.is_available()))
    det_model = det_model.to(device)
    det_model.eval()
    print('Finished loading [face detection] model!')
    print(det_model)

    resize = 1
    

    # Face recognition. 
    if args.rec_backbone == 'resnet18':
        rec_model = resnet_face18(args.use_se)
    elif args.rec_backbone == 'resnet34':
        rec_model = resnet34()
    elif args.rec_backbone == 'resnet50':
        rec_model = resnet50()
    rec_model.load_state_dict(torch.load(args.rec_trained_weights))
    rec_model.to(device)
    rec_model.eval()
    print('Finished loading [face recognition] model!')
    print(rec_model)


    # Input data: 
    multiple_faces = True
    cap = cv2.VideoCapture('/home/alex/바탕화면/Data/AffWild2_raw/6-30-1920x1080.mp4')
    main(cap, det_model, rec_model, multiple_faces)