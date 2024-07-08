import config as conf
from .head_pose import PoseEstimator
from ID_recog.model import Backbone

import cv2
import numpy as np
import os
import mediapipe as mp
import csv

from decord import VideoReader
from decord import cpu, gpu
from deepface import DeepFace
from retinaface import RetinaFace

### 1) Brisque: check image quality.
# credit: (https://live.ece.utexas.edu/publications/2011/am_asilomar_2011.pdf)



### 2) Face recognition: check if constant ID in video. 
def init_id_recognizer(): 
    
    # Initialize model. 
    model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)

    # Load weights. 
    model.load_state_dict(torch.load(conf.save_path/'model_{}'.format(fixed_str)))
    
    # Set to eval. 
    model.eval()
    
    return model

def detect_face(detector, image): 
    
    # Landmark results. 
    results = detector.detect(image)
    face = results.face_landmarks[0]

    landmarks = []
    if face is not None: 
        for idx in conf.landmark_points_68: 
            x = int(face[idx].x * width)
            y = int(face[idx].y * height)
            # x = face[idx].x
            # y = face[idx].y
            landmarks.append([x,y])  # Add to list of landmarks. 
    
    else: # no face detected...
        return None
    
    landmarks = np.array(landmarks)  # 68 x 2 dim. 
    return landmarks

def detect_id(model, image): 
    """
    Assumed to be PIL image (rgb?)
    """
    
    # threshold = conf.threshold
    
    # test time augmentation (mirroring not necessary, just normalize). 
    emb = model(conf.test_transform(image).to(conf.device).unsqueeze(0))
    
    return emb

### 3) Head pose: check if face alignment is favorable. 
def predict_head_pose(landmarks, pose_estimator): 
    """
    Estimates head pose and associated Euler angles from single image. 
    For rotation, assumes X-Y-Z order. 
    """
    
    # Estimate head pose. 
    pose = pose_estimator.solve(landmarks)
    
    # Calculate Euler angle. 
    euler_deg = pose_estimator.euler_angles(pose)
    
    pitch = euler_deg[0]
    roll = euler_deg[2]
    yaw = euler_deg[1]
    print(f"Pitch (x rot): {pitch}")
    print(f"Roll (y rot): {roll}")
    print(f"Yaw (z rot): {yaw}")

    return pitch, roll, yaw


### 4) Head pose: check if face alignment is favorable. 



def main(root_path, face_detector, id_recognizer, pose_estimator): 
    
    csv_path = ""
    
    f = open(csv_path, 'w')
    writer = csv.writer(f)
    
    # Iterate through videos in database. 
    for vid_name in sorted(os.listdir(root_path)): 
        vid_path = os.path.join(root_path, vid_name)
        
        for img_name in sorted(os.listdir(vid_path)): 
            if not img_name.startswith('.'):
                
                row = []
                row.extend([vid_path, img_name])  # Add current video path and image name to row. 
                
                
                img_path = os.path.join(vid_path, img_name)
                
                # Load image. 
                img_mp = mp.Image.create_from_file(img_path)  # Using innate MediaPipe version of images. 
                # height, width, _ = img.numpy_view().shape


                ### Calculate image quality score. 
                # BRISQUE
                
                
                ### Check if face detected. 
                landmarks = detect_face(face_detector, img_mp)
                face_is_detected = landmarks is not None
                row.append(face_is_detected)
                
                if face_is_detected: 
                    
                    ### Check if ID is consistent. 
                    # HOW TO GET CLASS CENTER? 


                    ### Calculate head pose. 
                    pitch, roll, yaw = predict_head_pose(landmarks, pose_estimator)
                    row.append([pitch, roll, yaw])
                    
                    # if (abs(pitch) > 45) or (abs(roll) > 45) or (abs(yaw) > 45):  # if head angles are rotated beyond 45 degrees. 
                    
                                             


# CSV format: 
# vid path  /  image name ('00001.jpg')  /  face detected (bool)  /  ID (majority?)  /  euler angles (pitch, roll, yaw)  /  annotation labels (valence, arousal, AU) 
                
                
                        

if __name__ == "__main__":
    
    # root = "/home/alex/바탕화면/Data/Aff-Wild2/cropped_aligned"

    # # Initialize landmark detector & pose estimator.  
    # lm_detector = init_face_detector()
    
    # print('{}_{} model opened for face recognition!'.format(conf.net_mode, conf.net_depth))
    
    # pose_estimator = PoseEstimator(width, height)

    # # Run main func. 
    # main(root, lm_detector, pose_estimator)
    
    # Load video. 
    vr = VideoReader('/home/alex/바탕화면/Data/batch1/1-30-1280x720.mp4', ctx=cpu(0))
    fps = vr.get_avg_fps()
    
    # Loop through video and detect faces. 
    for idx in range(len(vr)): 
        
        frame = vr[idx].asnumpy()  # Format: RGB
            
        # Detect face. 
        try: 
            # Input: required to be BGR (https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py#L78)
            # Output: [face_cropped, bbox_coords, confidence]
            # face_objs = DeepFace.extract_faces(frame, target_size=(224, 224), detector_backend="retinaface", 
            #                     enforce_detection=True, align=True, grayscale=False)  # Seems like input is required to be BGR... 
           
            
        
            
        except ValueError: 
            # face not detected... 

        
        # Aligned, padded result. 
        aligned_img = face_objs[0]['face']
        
        # Cropped, resized result. 
        x = face_objs[0]['facial_area']['x']
        y = face_objs[0]['facial_area']['y']
        w = face_objs[0]['facial_area']['w']
        h = face_objs[0]['facial_area']['h']
        cropped_img = cv2.resize(frame[y:y+h, x:x+w], (112, 112))
        cropped_img = cropped_img[:,:,::-1]
        
        cv2.imshow('auto', aligned_img)
        cv2.imshow('manual', cropped_img)
        cv2.waitKey(0)
        print(0)

        # Extract representation. 
        embedding_objs = DeepFace.represent(frame)
        embedding = embedding_objs[0]["embedding"]