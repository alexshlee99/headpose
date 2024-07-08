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


def detect_Retina(img_raw, device, net, cfg, args): 
    """
    Returns detection results based on RetinaFace (PyTorch implementation).
    
    Input: BGR image.
    Output: List of arrays for each face -> [bbox_coords, confidence, landmark_coords]
    """
    resize = 1
    
    img = np.float32(img_raw)
    
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    
    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    # print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    #=============================================================================
    # Personal Addition: constrained bbox
    # areas = np.asarray([int(box[2]-box[0]) * int(box[3]-box[1]) for box in boxes])
    
    # if prev_box_area: 
    #     inds2 = np.where(areas <= prev_box_area)[0]
    #     boxes = boxes[inds2]
    #     landms = landms[inds2]
    #     scores = scores[inds2]

    #=============================================================================

    # do NMS (Non-Maxmimum Suppression)
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    
    return dets