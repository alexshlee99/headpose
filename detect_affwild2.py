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
import shutil

from filter.crop_utils import alignment_procedure
from detect_engine import detect_Retina

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='RetinaFace_Torch/weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')  # Default: 0.02 
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.9, type=float, help='visualization_threshold')  # Changed to exclude non-face objects. 
args = parser.parse_args()


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


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    
    #=====================================================================================
    
    left_and_right = ["6-30-1920x1080_left", "30-30-1920x1080_left", "46-30-484x360_left", "49-30-1280x720_left", 
                      "52-30-1280x720_left", "130-25-1280x720_left", "135-24-1920x1080_left", "video5_left", 
                      "video10_1_left", "video29_left", "video49_left", "video55_left"]
    left = [""]
    right = ["10-60-1280x720_right"]
    
    # DROP VIDEO2
    raw_data_pth = "/home/alex/바탕화면/Data/AffWild2/test"
    output_data_pth = "/home/alex/바탕화면/Data/AffWild2/result"
    
    # Loop through raw, unprocessed videos. 
    for vid in sorted(os.listdir(raw_data_pth)): 
        if not vid.startswith('.'):  
            
            vid_name = vid.split('.')[0]
            
            # Initialize output directory for current video. 
            output_vid_path = os.path.join(output_data_pth, vid_name)
            if os.path.exists(output_vid_path):
                shutil.rmtree(output_vid_path)
            os.makedirs(output_vid_path)
                        
            # Load raw video.  
            raw_vid_path = os.path.join(raw_data_pth, vid)
            cap = cv2.VideoCapture(raw_vid_path)
            i = 1

            # Loop through video, and manually crop. 
            while (cap.isOpened()):
                
                ret, frame = cap.read()
                if not ret: 
                    break
                
                im_height, im_width,_ = frame.shape

                # Detect face. 
                dets = detect_Retina(frame, device, net, cfg, args)
            
                # Processing based on detection (crop, align, ID track, Euler angles).
                if len(dets) == 0: 
                    print("No face detected...")
                    i += 1
                    continue
                
                # Force to choose only 1 face with highest confidence. 
                max_idx = 0
                max_score = 0
                for idx, det in enumerate(dets): 
                    if det[4] > args.vis_thres and det[4] > max_score: 
                        max_idx = idx
                        max_score = det[4]
                
                # Calculate bounding box, and crop. 
                det = list(map(int, dets[max_idx]))
                
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
                name = os.path.join(output_vid_path, f"{str(i).zfill(5)}.jpg")
                cv2.imwrite(name, face_resized)
                # print(f"{i}th frame of [{vid_name}] finished...")
                
                i += 1
                
            cap.release()
                        