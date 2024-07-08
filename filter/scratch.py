from decord import VideoReader
from decord import cpu, gpu
from retinaface import RetinaFace
from crop_utils import alignment_procedure


import matplotlib.pyplot as plt
import cv2
import numpy as np


def align_crop_faces(img, obj, threshold=0.9, model = None, align = True, allow_upscaling = True):
    """
    Align detected face (mostly roll-based), and crop with conserved ratio. 
    Credit: DeepFace (GitHub)
    """

    resp = []

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]

            facial_area = identity["facial_area"]
            facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]  # Crop face. 
        
            if align == True:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                mouth_right = landmarks["mouth_right"]
                mouth_left = landmarks["mouth_left"]

                facial_img = alignment_procedure(facial_img, right_eye, left_eye, nose)

            resp.append(facial_img[:, :, ::-1])

    return resp


###================================================================================================
def test_video(): 
    # Load video. 
    cap = cv2.VideoCapture('/home/alex/바탕화면/Data/batch1/4-30-1920x1080.mp4')
    i = 0

    while (cap.isOpened()):
        
        ret, frame = cap.read()
        if not ret: 
            break

        # Detect face. 
        result = RetinaFace.detect_faces(frame, threshold=0.9, model = None, allow_upscaling = False)  # Input must be BGR!
        
        # if len(resp) == 1: 
            
        #     score = resp['face_1']['score']
        #     bbox_coord = resp['face_1']['facial_area']
        #     lmk = resp['face_1']['landmarks']
        
        #     bbox_coords.append(bbox_coord)
        #     lmks.append(lmk)
        
        faces = align_crop_faces(frame, result, threshold=0.9, align = True, allow_upscaling=True)
        
        if len(faces) == 1: # more than 1 face detected.
            face = faces[0]
        else: 
            break
        
        i += 1
        
        cropped_img = cv2.resize(face[:,:,::-1], (224,224))
        cv2.imwrite(f'output/{i}.jpg', cropped_img)
        print(i)
        
        if i == 100: 
            break 
    

def test_image(): 
    
    image = cv2.imread("/home/alex/바탕화면/AU/test_angle/yaw.jpeg")
    
    faces = RetinaFace.extract_faces(image, threshold=0.9, align = True, allow_upscaling=True)
    for face in faces:
        plt.imshow(face)
        plt.show()


if __name__ == "__main__":
    
    test_video()